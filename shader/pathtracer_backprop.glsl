#version 450 core
#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f)   uniform image2D color_prediction;
layout (binding = 1, rgba32f)   uniform image2D color_reference;
layout (binding = 2, rgba32f)   uniform image2D color_backprop;

uniform int bounces;
uniform int seed;
uniform int show_environment;
uniform ivec2 resolution;
uniform int current_sample;

#include "common.glsl"

// ---------------------------------------------------
// adjoint delta tracking

void backward_real(const vec3 ipos, const float P_real, const vec3 dy) {
    if (P_real <= 0.f) return;
    const float dx = sum(dy) / (vol_majorant * P_real);
    // add_gradient(ipos, sanitize(dx));
}

void backward_null(const vec3 ipos, const float P_null, const vec3 dy) {
    if (P_null <= 0.f) return;
    const float dx = -sum(dy) / (vol_majorant * P_null);
    // add_gradient(ipos, sanitize(dx));
}

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

// ---------------------------------------------------
// path replay backprop

bool sample_volume_adjoint(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, inout vec3 L, inout uint seed, const vec3 grad) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // delta tracking
    t = near_far.x - log(1 - rng(seed)) * vol_inv_majorant;
    while (t < near_far.y) {
#ifdef USE_TRANSFERFUNC
        const float tc = lookup_density(ipos + t * idir, seed) * vol_inv_majorant;
        const vec4 rgba = tf_lookup(tc);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density(ipos + t * idir, seed);
#endif
        const float P_real = d * vol_inv_majorant;
        L -= throughput * (1.f - vol_albedo) * lookup_emission(ipos + t * idir, seed) * P_real;
        // classify as real or null collison
        if (rng(seed) < P_real) {
#ifdef USE_TRANSFERFUNC
            throughput *= rgba.rgb * vol_albedo;
            // TODO backprop
            // backward_tf_color(tc, rgba.rgb, L * grad);
            backward_tf_extinction(tc, P_real, L * grad);
            // TODO: optimize extinction with envmap
            // backward_tf_extinction(tc, P_real, lookup_environment(normalize(vec3(vol_model * vec4(idir, 0)))) * grad);
#else
            throughput *= vol_albedo;
#endif
            return true;
        }
        // advance
        t -= log(1 - rng(seed)) * vol_inv_majorant;
#ifdef USE_TRANSFERFUNC
        backward_tf_extinction(tc, (1.f - P_real), L * grad);
        // TODO: optimize extinction with envmap
        // backward_tf_extinction(tc, 1.f - P_real, lookup_environment(normalize(vec3(vol_model * vec4(idir, 0)))) * grad);
#endif
    }
    return false;
}

// DDA-based volume sampling
bool sample_volumeDDA_adjoint(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, inout vec3 L, inout uint seed, const vec3 grad) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    const vec3 ri = 1.f / idir;
    // march brick grid
    t = near_far.x + 1e-6f;
    float tau = -log(1.f - rng(seed)), mip = MIP_START;
    while (t < near_far.y) {
        const vec3 curr = ipos + t * idir;
#ifdef USE_TRANSFERFUNC
        const float majorant = vol_majorant * tf_lookup(lookup_majorant(curr, int(round(mip))) * vol_inv_majorant).a;
#else
        const float majorant = lookup_majorant(curr, int(round(mip)));
#endif
        const float dt = stepDDA(curr, ri, int(round(mip)));
        t += dt;
        tau -= majorant * dt;
        mip = min(mip + MIP_SPEED_UP, 3.f);
        if (tau > 0) continue; // no collision, step ahead
        t += tau / majorant; // step back to point of collision
        if (t >= near_far.y) break;
#ifdef USE_TRANSFERFUNC
        const float tc = lookup_density(ipos + t * idir, seed) * vol_inv_majorant;
        const vec4 rgba = tf_lookup(tc);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density(ipos + t * idir, seed);
#endif
        L -= throughput * (1.f - vol_albedo) * lookup_emission(ipos + t * idir, seed) * d * vol_inv_majorant;
        if (rng(seed) * majorant < d) { // check if real or null collision
            throughput *= vol_albedo;
#ifdef USE_TRANSFERFUNC
            throughput *= rgba.rgb;
            // TODO merge backprop
            // backward_tf_color(tc, rgba.rgb, L * grad);
            // backward_tf_extinction_real(tc, d / majorant, L * grad * (vol_majorant / majorant)); // TODO
#endif
            return true;
        }
        tau = -log(1.f - rng(seed));
        mip = max(0.f, mip - MIP_SPEED_DOWN);
#ifdef USE_TRANSFERFUNC
        // backward_tf_extinction_null(tc, 1 - d / majorant, L * grad * (1.f - vol_majorant / majorant)); // TODO
#endif
    }
    return false;
}

vec3 path_replay_backprop(vec3 pos, vec3 dir, inout uint seed, vec3 L, const vec3 grad) {
    vec3 throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    while (sample_volume_adjoint(pos, dir, t, throughput, L, seed, grad)) {
    // while (sample_volume(pos, dir, t, throughput, L, seed)) {
    // while (sample_volumeDDA_adjoint(pos, dir, t, throughput, L, seed, grad)) {
    // float pdf;
    // while (sample_volume_raymarch(pos, dir, t, throughput, pdf, seed)) {

        // advance ray
        pos += t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittance/*DDA*/(pos, w_i, seed);
            // const float Tr = transmittance_raymarch(pos, w_i, seed);
            const vec3 Li = throughput * f_p * mis_weight * Tr * Le_pdf.rgb / Le_pdf.w;
            // TODO: backprop along shadow ray
            L -= Li;
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
        L -= throughput * mis_weight * Le;
    }

    // TODO FIXME: check seeds and verify path replay
    return vec3(0);
    // return abs(L);
    return luma(abs(L)) <= 1e-6 ? vec3(0) : vec3(1e10, 0, 1e10); // pink of doom
}

// ---------------------------------------------------
// direct volume rendering and irradiance cache

vec3 direct_volume_rendering_irradiance_cache(vec3 pos, vec3 dir, inout uint seed, in vec3 dy) {
    vec3 L = vec3(0);
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return show_environment > 0 ? lookup_environment(dir) : vec3(0);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(pos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(dir, 0)); // non-normalized!
    const float dt = (near_far.y - near_far.x) / float(RAYMARCH_STEPS);
    // jitter starting position
    const float jitter = rng(seed) * dt;
    near_far.x += jitter;
    float Tr = exp(-lookup_density(ipos + near_far.x * idir, seed) * jitter);
    const bool partials = !all(equal(vec3(0), dy));
    // ray marching
    for (int i = 0; i < RAYMARCH_STEPS; ++i) {
        const vec3 curr = ipos + min(near_far.x + i * dt, near_far.y) * idir;
#ifdef USE_TRANSFERFUNC
        const float d_real = lookup_density(curr, seed);
        const vec4 rgba = tf_lookup(d_real * vol_inv_majorant);
        const float d = rgba.a * vol_majorant;
        const vec3 Le = vol_albedo * rgba.rgb * irradiance_query(curr, seed);
#else
        const float d = lookup_density(curr, seed);
        const vec3 Le = vol_albedo * irradiance_query(curr, seed);
#endif
        const float dtau = d * dt;
#ifdef USE_TRANSFERFUNC
        // partials
        if (partials) {
            backward_tf_color(d_real * vol_inv_majorant, rgba.rbg, L * dy);
            // TODO partials to extinction
            const float dx = (1 - dtau * i) * dt;
            // backward_tf_extinction(d_real * vol_inv_majorant, dtau, dx * L * dy);
        }
        // accum emission from irradiance cache with geom avg of transmittance along segment
        L += Le * dtau * Tr * exp(-dtau * 0.5);
#endif
        // update transmittance
        Tr *= exp(-dtau);
        if (Tr <= 1e-5) break;
    }
    if (show_environment > 0) L += lookup_environment(dir) * Tr;
    return L;
}

vec3 trace_path_cache(vec3 pos, vec3 dir, inout uint seed) {
    // trace path
    vec3 L = vec3(0), throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t; // t: end of ray segment (i.e. sampled position or out of volume)
    while (sample_volumeDDA(pos, dir, t, throughput, L, seed)) {
        // advance ray
        pos = pos + t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            const float f_p = phase_isotropic();
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittanceDDA(pos, w_i, seed);
            L += throughput * mis_weight * f_p * Tr * Le_pdf.rgb / Le_pdf.w;
        }

        // early out?
        if (++n_paths >= max(0, bounces-1)) { free_path = false; break; }
        // russian roulette
        const float rr_val = luma(throughput);
        if (rr_val < .1f) {
            const float prob = 1 - rr_val;
            if (rng(seed) < prob) { free_path = false; break; }
            throughput /= 1 - prob;
        }

        // scatter ray
        const vec3 scatter_dir = sample_phase_isotropic(rng2(seed));
        dir = scatter_dir;
    }

    // free path? -> add envmap contribution
    if (free_path && show_environment > 0) {
        const vec3 Le = lookup_environment(dir);
        const float f_p = phase_isotropic();
        const float mis_weight = power_heuristic(f_p, pdf_environment(dir));
        L += throughput * mis_weight * Le;
    }

    return L;
}

void update_cache(vec3 pos, vec3 dir, inout uint seed) {
    float t;
    vec3 Li = vec3(0);
    vec3 throughput = vec3(1);
    if (sample_volumeDDA(pos, dir, t, throughput, Li, seed)) {
        pos += t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            const float f_p = phase_isotropic();
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittanceDDA(pos, w_i, seed);
            Li += mis_weight * f_p * Tr * Le_pdf.rgb / Le_pdf.w;
        }

        const vec3 scatter_dir = sample_phase_isotropic(rng2(seed));
        Li += trace_path_cache(pos, scatter_dir, seed);
        irradiance_update(vec3(vol_inv_model * vec4(pos, 1)), Li);
    }
}

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, resolution))) return;

    // compute gradient of l2 loss between prediction and reference
    const vec3 L = imageLoad(color_prediction, pixel).rgb;
    const vec3 L_ref = imageLoad(color_reference, pixel).rgb;
    const vec3 grad = 2 * (L - L_ref);

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * resolution.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir(pixel, resolution, rng2(seed));
    
    // replay path to backprop gradients
    // seed = forward_seed;
    const vec3 Lr = path_replay_backprop(pos, dir, seed, L, grad);
    // const vec3 Lr = direct_volume_rendering_irradiance_cache(pos, dir, seed, grad);

    // store result
    imageStore(color_backprop, pixel, vec4(mix(imageLoad(color_backprop, pixel).rgb, sanitize(Lr), 1.f / current_sample), 1));
}
