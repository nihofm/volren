#version 450 core
#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f)   uniform image2D color_prediction;
layout (binding = 1, rgba32f)   uniform image2D color_reference;
layout (binding = 2, rgba32f)   uniform image2D color_backprop;

uniform int current_sample;
uniform int sppx;
uniform int bounces;
uniform int seed;
uniform int show_environment;
uniform ivec2 resolution;

#include "common.glsl"

// ---------------------------------------------------
// helper funcs

float sum(const vec3 x) { return x.x + x.y + x.z; }
float mean(const vec3 x) { return sum(x) / 3.f; }
float sanitize(const float x) { return isnan(x) || isinf(x) ? 0.f : x; }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }

void add_gradient(const vec3 ipos, const float dx) {
    // XXX TODO update to new optimization target
    // const ivec3 iipos = ivec3(floor(ipos));
    // const uint idx = iipos.z * vol_size.x * vol_size.y + iipos.y * vol_size.x + iipos.x;
    // atomicAdd(parameters[idx].y, dx);
}

void add_gradient(const float d, const vec3 dx) {
    const float tc = clamp((d - tf_window_left) / tf_window_width, 0.0, 1.0 - 1e-6);
    const int idx = int(floor(tc * n_parameters));
    atomicAdd(gradients[idx].x, sanitize(dx.x));
    atomicAdd(gradients[idx].y, sanitize(dx.y));
    atomicAdd(gradients[idx].z, sanitize(dx.z));
}

// ---------------------------------------------------
// forward path tracing

vec3 trace_path(vec3 pos, vec3 dir, inout uint seed) {
    vec3 L = vec3(0), throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    while (sample_volumeDDA(pos, dir, t, throughput, seed)) {
        // advance ray
        pos += t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = 1.f;//show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittanceDDA(pos, w_i, seed);
            // L += throughput * mis_weight * f_p * Tr * Le_pdf.rgb / Le_pdf.w;
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

    // free path? -> add envmap contribution
    if (free_path && show_environment > 0) {
        const vec3 Le = lookup_environment(dir);
        const float mis_weight = 1.f;//n_paths > 0 ? power_heuristic(f_p, pdf_environment(dir)) : 1.f;
        L += throughput * mis_weight * Le;
    }

    return L;
}

// ---------------------------------------------------
// adjoint ray marching

float integrate_density(const vec3 ipos, const vec3 idir, const vec2 near_far, inout uint seed, out float last_interpolant) {
    const int steps = 32;
    const float dt = (near_far.y - near_far.x) / float(steps);
    // first step
    const float t0 = near_far.x + rng(seed) * dt;
    float last_value = lookup_density(ipos + t0 * idir, seed),  tau = 0.f;
    // integrate density
    for (int i = 1; i < steps; ++i) {
        const vec3 curr_pos = ipos + min(t0 + i * dt, near_far.y) * idir;
        const float curr_value = lookup_density(curr_pos, seed);
        last_interpolant = (last_value + curr_value) * 0.5f;
        tau += last_interpolant * dt;
        last_value = curr_value;
    }
    return tau;
}

float distance_pdf(const vec3 wpos, const vec3 wdir, const float t, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 0.f;
    near_far.y = min(near_far.y, t);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    float last_interpolant;
    const float tau = integrate_density(ipos, idir, near_far, seed, last_interpolant);
    return last_interpolant * exp(-tau);
}

float raymarch_transmittance_adjoint(const vec3 wpos, const vec3 wdir, inout uint seed, const vec3 dy, const float t_max = FLT_MAX) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 0.f;
    near_far.y = min(near_far.y, t_max);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    float dummy;
    const float tau = integrate_density(ipos, idir, near_far, seed, dummy);
    // store gradients
    const int steps = 32;
    const float dt = (near_far.y - near_far.x) / float(steps);
    const float dx = -exp(-tau) * sum(dy) * dt;
    float t0 = near_far.x + rng(seed) * dt;
    for (int i = 0; i < steps; ++i) {
        const vec3 curr_p = ipos + min(t0 + i * dt, near_far.y) * idir;
        add_gradient(curr_p, dx);
    }
    return dx;
}

// ---------------------------------------------------
// radiative backprop

vec3 radiative_backprop(vec3 pos, vec3 dir, inout uint seed, const vec3 grad) {
    vec3 Lr = vec3(0), throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t;
    while (sample_volume(pos, dir, t, throughput, seed)) {
        // Term: Q2 * K * Li
        // Approximate Li to avoid recursion (biased)
        vec3 Li = vec3(1);
        // emitter sampling
        vec3 w_i;
        const vec4 env = sample_environment(rng2(seed), w_i);
        if (env.w > 0) {
            const float f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float Tr = transmittance(pos + t * dir, w_i, seed);
            Li += f_p * Tr * env.rgb / env.w;
        }
        // backprop transmittance
        const vec3 weight = throughput * Li * grad;
        Lr += vec3(raymarch_transmittance_adjoint(pos, dir, seed, sanitize(weight), t));

        // early out?
        if (++n_paths >= bounces) { free_path = false; break; }
        // russian roulette
        const float rr_val = luma(throughput);
        if (rr_val < .1f) {
            const float prob = 1 - rr_val;
            if (rng(seed) < prob) { free_path = false; break; }
            throughput /= 1 - prob;
        }

        // advance and scatter ray
        pos += t * dir;
        dir = sample_phase_henyey_greenstein(dir, vol_phase_g, rng2(seed));
    }

    // Term: Q2 * Le
    if (free_path && show_environment > 0) {
        const vec3 Le = lookup_environment(dir);
        const vec3 weight = throughput * Le * grad;
        Lr += vec3(raymarch_transmittance_adjoint(pos, dir, seed, sanitize(weight)));
    }

    return Lr;
}

// ---------------------------------------------------
// adjoint delta tracking

void backward_real(const vec3 ipos, const float P_real, const vec3 dy) {
    if (P_real <= 0.f) return;
    const float dx = sum(dy) / (vol_majorant * P_real);
    add_gradient(ipos, sanitize(dx));
}

void backward_real(const float P_real, const vec3 dy) {
    if (P_real <= 0.f) return;
    // TODO actually differentiate
    add_gradient(P_real, dy);
}

void backward_null(const vec3 ipos, const float P_null, const vec3 dy) {
    if (P_null <= 0.f) return;
    const float dx = -sum(dy) / (vol_majorant * P_null);
    add_gradient(ipos, sanitize(dx));
}

// TODO check gradients
float transmittance_adjoint(const vec3 wpos, const vec3 wdir, inout uint seed, const vec3 grad) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // ratio tracking
    float t = near_far.x - log(1 - rng(seed)) * vol_inv_majorant, Tr = 1.f;
    while (t < near_far.y) {
        // apply stochastic filter once for replayability
        const vec3 curr = ipos + t * idir;// + rng3(seed) - .5f; // XXX DEBUG
        const float d = lookup_density(curr);
        const float P_null = 1.f - d * vol_inv_majorant;
        Tr *= P_null;
        backward_null(curr, P_null, grad);
        // russian roulette
        /* TODO
        if (Tr < .1f) {
            const float prob = 1 - Tr;
            if (rng(seed) < prob) return 0.f;
            Tr /= 1 - prob;
        }
        */
        // advance
        t -= log(1 - rng(seed)) * vol_inv_majorant;
    }
    return Tr;
}

// TODO check gradients
bool sample_volume_adjoint(const vec3 wpos, const vec3 wdir, out float t, inout uint seed, const vec3 grad) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // delta tracking
    t = near_far.x - log(1 - rng(seed)) * vol_inv_majorant;
    while (t < near_far.y) {
        // apply stochastic filter once for replayability
        const vec3 curr = ipos + t * idir;// + rng3(seed) - .5f; // XXX DEBUG
        const float d = lookup_density(curr);
        const float P_real = d * vol_inv_majorant;
        if (rng(seed) < P_real) {
            // real collision
            backward_real(curr, P_real, grad); // TODO include albedo
            return true;
        }
        // null collision
        // const float P_null = 1.f - P_real;
        // backward_null(curr, P_null, grad);
        // advance
        t -= log(1 - rng(seed)) * vol_inv_majorant;
    }
    return false;
}

// ---------------------------------------------------
// path replay backprop

vec3 path_replay_backprop(vec3 pos, vec3 dir, inout uint seed, vec3 L, const vec3 grad) {
    vec3 throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    // while (sample_volume_adjoint(pos, dir, t, seed, L * grad)) {
    while (sample_volumeDDA(pos, dir, t, throughput, seed)) {
        // backprop transmittance
        {
            const uint saved_seed = seed;
            // transmittance_adjoint(pos, dir, seed, throughput * lookup_environment(dir) * grad);
            seed = saved_seed;
        }

        // advance ray
        pos += t * dir;
        // throughput *= vol_albedo;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = 1.f;//show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittanceDDA(pos, w_i, seed);
            const vec3 Li = throughput * f_p * mis_weight * Tr * Le_pdf.rgb / Le_pdf.w;
            // L -= Li;
            // backprop transmittance
            {
                const uint saved_seed = seed;
                // transmittance_adjoint(pos, dir, seed, Li * grad / max(1e-8, Tr));
                seed = saved_seed;
            }
        }

        // TODO backprop scattering
        {
            const uint saved_seed = seed;
            const vec3 ipos = vec3(vol_inv_model * vec4(pos, 1));
            const float d = lookup_density(ipos, seed);
            const float P_real = d * vol_inv_majorant;
            const vec3 tf = tf_lookup(P_real).rgb;
            // TODO differentiate and include vol_albedo
            backward_real(P_real, L * grad / tf);
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
        // backprop
        // transmittance_adjoint(pos, dir, seed, L * grad);
        
        const vec3 Le = lookup_environment(dir);
        const float mis_weight = n_paths > 0 ? power_heuristic(f_p, pdf_environment(dir)) : 1.f;
        const vec3 Li = throughput * mis_weight * Le;
        L -= Li;
    }

    return abs(L);
}

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, resolution))) return;

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * resolution.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir(pixel, resolution, rng2(seed));

    // forward path tracing
    const uint forward_seed = seed;
    const vec3 L = trace_path(pos, dir, seed);

    // compute gradient of l2 loss between Lo (1spp) and reference (Nspp)
    const vec3 L_ref = imageLoad(color_reference, pixel).rgb;
    const vec3 grad = 2 * (L - L_ref);
    
#if 0
    // radiative backprop
    const vec3 Lr = radiative_backprop(pos, dir, seed, grad);
#else
    // path replay backprop
    seed = forward_seed;
    const vec3 Lr = path_replay_backprop(pos, dir, seed, L, grad);
#endif

    // store results
    imageStore(color_prediction, pixel, vec4(mix(imageLoad(color_prediction, pixel).rgb, sanitize(L), 1.f / current_sample), 1));
    imageStore(color_backprop, pixel, vec4(mix(imageLoad(color_backprop, pixel).rgb, sanitize(Lr), 1.f / current_sample), 1));
}
