#version 450 core
#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f)   uniform image2D color_prediction;
layout (binding = 1, rgba32f)   uniform image2D color_reference;
layout (binding = 2, rgba32f)   uniform image2D color_backprop;

#include "common.glsl"

uniform int bounces;
uniform int seed;
uniform int show_environment;
uniform int current_sample;
uniform ivec2 resolution;

// ---------------------------------------------------
// adjoint delta tracking

void add_density_gradient(const vec3 ipos, const float dx) {
    const ivec3 iipos = ivec3(floor(ipos));
    atomicAdd(gradients[iipos.z * grid_size.x * grid_size.y + iipos.y * grid_size.x + iipos.x], sanitize(dx));
}

void backward_real(const vec3 ipos, const float P_real, const vec3 dy) {
    if (P_real <= 1e-8f || any(lessThan(ipos, vec3(0))) || any(greaterThanEqual(ipos, grid_size))) return;
    const float dx = sum(dy) / (vol_majorant * P_real);
    add_density_gradient(ipos, dx);
}

void backward_null(const vec3 ipos, const float P_null, const vec3 dy) {
    if (P_null <= 1e-8f || any(lessThan(ipos, vec3(0))) || any(greaterThanEqual(ipos, grid_size))) return;
    const float dx = -sum(dy) / (vol_majorant * P_null);
    add_density_gradient(ipos, dx);
}

/*
void backward_tf_color(const float d, const vec3 rgb, const vec3 dy) {
    const float tc = (d - tf_window_left) / tf_window_width;
    const int idx = clamp(int(floor(tc * tf_size)), 0, int(tf_size) - 1);
    const vec3 dx = dy / rgb;
    // TODO: filter
    if (rgb.x > 0) atomicAdd(gradients[idx].x, sanitize(dx.x));
    if (rgb.y > 0) atomicAdd(gradients[idx].y, sanitize(dx.y));
    if (rgb.z > 0) atomicAdd(gradients[idx].z, sanitize(dx.z));
}

void backward_tf_extinction(const float tc, const float P_real, const vec3 dy) {
    const float tc_mapped = (tc - tf_window_left) / tf_window_width;
    if (tc_mapped < 0.f || tc_mapped >= 1.f || P_real <= 1e-6) return;
    const float dx = -sum(dy) / P_real;
    // TODO: filter
    const int idx = int(floor(tc_mapped * tf_size));
    if (P_real > 0) atomicAdd(gradients[idx].w, sanitize(dx));
}
*/

// ---------------------------------------------------
// path replay backprop

float transmittance_adjoint(const vec3 wpos, const vec3 wdir, inout uint seed, const vec3 grad, float t_max = FLT_MAX) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    near_far.y = min(t_max, near_far.y);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // ratio tracking
    float t = near_far.x - log(1 - rng(seed)) * vol_inv_majorant, Tr = 1.f;
    while (t < near_far.y) {
        const vec3 pos_jitter = ipos + t * idir + rng3(seed) - .5f;
#ifdef USE_TRANSFERFUNC
        const vec4 rgba = tf_lookup(lookup_density(pos_jitter) * vol_inv_majorant);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density(pos_jitter);
#endif
        // track ratio of real to null particles
        const float P_null = 1 - d * vol_inv_majorant;
        Tr *= P_null;
        backward_null(pos_jitter, P_null, grad);
        // russian roulette
        if (Tr < .1f) {
            const float prob = 1 - Tr;
            if (rng(seed) < prob) return 0.f;
            Tr /= 1 - prob;
        }
        // advance
        t -= log(1 - rng(seed)) * vol_inv_majorant;
    }
    return Tr;
}

bool sample_volume_adjoint(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, /*inout*/ vec3 L, inout uint seed, const vec3 grad) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // delta tracking
    t = near_far.x - log(1 - rng(seed)) * vol_inv_majorant;
    while (t < near_far.y) {
        const vec3 pos_jitter = ipos + t * idir + rng3(seed) - .5f;
#ifdef USE_TRANSFERFUNC
        const float tc = lookup_density(pos_jitter) * vol_inv_majorant;
        const vec4 rgba = tf_lookup(tc);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density(pos_jitter);
#endif
        const float P_real = d * vol_inv_majorant;
        // L -= throughput * (1.f - vol_albedo) * lookup_emission(ipos + t * idir, seed) * P_real;
        // classify as real or null collison
        if (rng(seed) < P_real) {
#ifdef USE_TRANSFERFUNC
            throughput *= rgba.rgb * vol_albedo;
#else
            throughput *= vol_albedo;
#endif
            backward_real(pos_jitter, P_real * vol_albedo, L * grad / vol_albedo);
            return true;
        }
        // advance
        t -= log(1 - rng(seed)) * vol_inv_majorant;
        backward_null(pos_jitter, P_real, L * grad);
    }
    return false;
}

vec3 path_replay_backprop(vec3 pos, vec3 dir, inout uint seed, vec3 L, const vec3 grad) {
    vec3 throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    // while (sample_volume_adjoint(pos, dir, t, throughput, L, seed, grad)) {
    while (sample_volume(pos, dir, t, throughput, L, seed)) {
        // advance ray
        pos += t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittance(pos, w_i, seed);
            const vec3 Li = throughput * mis_weight * f_p * Tr * Le_pdf.rgb / Le_pdf.w;
            L -= Li;
            {
                const uint saved_seed = seed;
                // TODO
                transmittance_adjoint(pos, w_i, seed, Li * grad);
                // transmittance_adjoint(pos, w_i, seed, L * grad);
                // transmittance_adjoint(pos, w_i, seed, Le_pdf.rgb / Le_pdf.w * grad));
                seed = saved_seed;
            }
        }

        {
            // backprop to scatter event
            const uint saved_seed = seed;
            const vec3 wpos = pos - t * dir;
            const float pdf_scatter = 1.f;//lookup_density(vec3(vol_inv_model * vec4(wpos, 1))) * transmittance(wpos, dir, seed, t);
            if (pdf_scatter > 0) transmittance_adjoint(wpos, dir, seed, L * grad / pdf_scatter, t);
            seed = saved_seed;
        }

        // early out?
        if (++n_paths >= bounces) { free_path = false; break; }
        // russian roulette
        const float rr_val = luma(throughput);
        if (rr_val < .1f) {
            const float prob = 1 - rr_val;
            if (rng(seed) < prob) { free_path = false; break; }
            throughput /= 1 - prob;
        }

        // scatter ray
        const vec3 scatter_dir = sample_phase_henyey_greenstein(dir, vol_phase_g, rng2(seed));
        f_p = phase_henyey_greenstein(dot(-dir, scatter_dir), vol_phase_g);
        dir = scatter_dir;
    }

    if (free_path && show_environment > 0) {
        const vec3 Le = lookup_environment(dir);
        const float mis_weight = n_paths > 0 ? power_heuristic(f_p, pdf_environment(dir)) : 1.f;
        const vec3 Li = throughput * mis_weight * Le;
        L -= Li;
        {
            const uint saved_seed = seed;
            // TODO
            transmittance_adjoint(pos, dir, seed, L * -grad); // TODO why negate?
            seed = saved_seed;
            // return 5 * L * grad;
        }
    } else {
        // TODO handle cancelled ray?
    }

    // return vec3(0);
    return abs(L);
    return luma(abs(L)) <= 1e-6 ? vec3(0) : vec3(1e10, 0, 1e10); // pink of doom
}

// ---------------------------------------------------
// direct volume rendering and irradiance cache

vec3 backprop_irradiance_cache(vec3 pos, vec3 dir, inout uint seed, vec3 L, const vec3 dy) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return show_environment > 0 ? L - lookup_environment(dir) : L;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(pos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(dir, 0)); // non-normalized!
    const float dt = (near_far.y - near_far.x) / float(RAYMARCH_STEPS);
    // jitter starting position
    const float jitter = rng(seed) * dt;
    near_far.x += jitter;
    float Tr = exp(-lookup_density(ipos + near_far.x * idir, seed) * jitter); // TODO: differentiate this?
    // ray marching
    for (int i = 0; i < RAYMARCH_STEPS; ++i) {
        const vec3 curr = ipos + min(near_far.x + i * dt, near_far.y) * idir + rng3(seed) - .5f;
#ifdef USE_TRANSFERFUNC
        const float d_real = lookup_density(curr);
        const vec4 rgba = tf_lookup(d_real * vol_inv_majorant);
        const float d = rgba.a * vol_majorant;
        const vec3 I = vol_albedo * rgba.rgb * irradiance_query(curr, seed);
#else
        const float d = lookup_density(curr);
        const vec3 I = vol_albedo * irradiance_query(curr, seed);
#endif
        //return I; // TODO: debug irradiance cache?
        const float dtau = d * dt;
        const float Tr_i = exp(-dtau * vol_extinction);
        // subtract emission from irradiance cache
        L -= I * dtau * vol_scattering * Tr * exp(-dtau * vol_extinction * 0.5);
        // dL/de+s
        add_density_gradient(curr, sum(dy * Tr * (-dt * Tr_i * L * vol_extinction + dt * I * vol_scattering)));
        // dL/de
        //add_density_gradient(curr, sum(dy * Tr * -dt * Tr_i * L * vol_extinction));
        // dL/ds
        //add_density_gradient(curr, sum(dy * Tr * dt * I * vol_scattering)); // TODO scattering derivative
        // update transmittance
        Tr *= Tr_i;
        //if (Tr <= 1e-5) return L;
    }
    // TODO: term regarding envmap contribution (along whole ray?)
    if (show_environment > 0) L -= lookup_environment(dir) * Tr;
    return L;
}

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, resolution))) return;

    // compute gradient of l2 loss between prediction and reference
    const vec3 L = imageLoad(color_prediction, pixel).rgb;
    if (sum(L) <= 1e-6) return;
    const vec3 L_ref = imageLoad(color_reference, pixel).rgb;
    const vec3 grad = 2 * (L - L_ref);

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * resolution.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir(pixel, resolution, rng2(seed));
    
    // replay path to backprop gradients
    //const vec3 Lr = path_replay_backprop(pos, dir, seed, L, grad);
    //const vec3 Lr = abs(L - vec3(transmittance_adjoint(pos, dir, seed, grad)));
    const vec3 Lr = backprop_irradiance_cache(pos, dir, seed, L, grad);

    // store result
    imageStore(color_backprop, pixel, vec4(mix(imageLoad(color_backprop, pixel).rgb, sanitize(Lr), 1.f / current_sample), 1));
}
