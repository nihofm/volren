#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f)   uniform image2D color_prediction;
layout (binding = 1, rgba32f)   uniform image2D color_reference;
layout (binding = 2, r32f)      uniform image3D gradients;
layout (binding = 3, rgba32f)   uniform image2D color_backprop;

uniform int current_sample;
uniform int sppx;
uniform int bounces;
uniform int seed;
uniform int show_environment;
uniform ivec2 resolution;

#include "common.glsl"

// ---------------------------------------------------
// helper funcs

float sum(const vec3 x) { return (x.x + x.y + x.z); }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }

struct PathSegment {
    vec3 pos;
    float t_min;
    vec3 dir;
    float t_max;
    vec3 weight;
    float tau;

    void store(vec3 _pos, vec3 _dir, float _t_min, float _t_max, vec3 _weight, float _tau) {
        pos = _pos;
        t_min = _t_min;
        dir = _dir;
        t_max = _t_max;
        weight = _weight;
        tau = _tau;
    }
};

bool sample_volume_backprop(const vec3 wpos, const vec3 wdir, out float t, out float tau, out float tr_pdf, inout vec3 throughput, inout uint seed) {
    // default out variables
    t = 0.f, tau = 0.f, tr_pdf = 1.f;
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // transform to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // compute step size, sample target tau and jitter starting point
    const int steps = 32;
    const float dt = (near_far.y - near_far.x) / float(steps);
    const float sampled_tau = -log(1.f - rng(seed));
    t = near_far.x + rng(seed) * dt;
    // raymarch
    for (int i = 0; i < steps; ++i) {
        const ivec3 curr_p = ivec3(ipos + min(t, near_far.y) * idir);
        const float curr_d = lookup_density(curr_p, seed);
        tau += curr_d * dt;
        t += dt;
        if (tau >= sampled_tau) {
            // solve for exact collision
            const float f = (tau - sampled_tau) / curr_d;
            t -= f * dt;
            tau = sampled_tau;
            tr_pdf = curr_d * exp(-sampled_tau);
            throughput *= vol_albedo;
            return true;
        }
    }
    t = near_far.y; // TODO this necessary?
    tr_pdf = exp(-tau);
    return false;
}

// ---------------------------------------------------
// forward path tracing

vec3 trace_path(vec3 pos, vec3 dir, inout uint seed) {
    // trace path
    vec3 radiance = vec3(0), throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    while (sample_volume(pos, dir, t, throughput, seed)) {
        // advance ray
        pos += t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(seed), w_i);
        if (Li_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = power_heuristic(Li_pdf.w, f_p);
            const float Tr = transmittance(pos, w_i, seed);
            const vec3 Li = throughput * mis_weight * f_p * Tr * Li_pdf.rgb / Li_pdf.w;
            radiance += Li;
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
        const float mis_weight = n_paths > 0 ? power_heuristic(f_p, pdf_environment(dir)) : 1.f;
        radiance += throughput * mis_weight * Le;
    }

    return radiance;
}

// ---------------------------------------------------
// adjoint funcs

float transmittance_adjoint(const vec3 wpos, const vec3 wdir, inout uint seed, const vec3 dy, const float t_max = FLT_MAX) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 0.f;
    near_far.y = min(near_far.y, t_max);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    const float tau = integrate_density(ipos, idir, near_far, seed);
    // store gradients
    const int steps = 32;
    const float dt = (near_far.y - near_far.x) / float(steps);
    const float dx = -exp(-tau) * sum(dy) * dt;
    float t0 = near_far.x + rng(seed) * dt;
    for (int i = 0; i < steps; ++i)
        imageAtomicAdd(gradients, ivec3(ipos + min(t0 + i * dt, near_far.y) * idir), dx);
    return dx;
}

// ---------------------------------------------------
// radiative backprop

vec3 radiative_backprop(vec3 pos, vec3 dir, inout uint seed, const vec3 dy) {
    vec3 throughput = vec3(1), result = vec3(0);
    for (int i = 0; i < bounces; ++i) {
        // sample volume and compute pdf
        float t, tr_pdf;
        const bool escaped = !sample_volume_raymarch_pdf(pos, dir, t, tr_pdf, throughput, seed);
        if (tr_pdf <= 0.f) break;

        // Term: Q2 * Le
        if (escaped && show_environment > 0) {
            const vec3 Le = lookup_environment(dir);
            const vec3 weight = throughput * Le * dy / tr_pdf;
            const vec3 Lr = vec3(transmittance_adjoint(pos, dir, seed, sanitize(weight), t));
            result += Lr;
        }
        if (escaped) break;

        // Term: Q2 * K * Li
        // Approximate Li to avoid recursion (biased)
        vec3 Li = vec3(1);
        // emitter sampling
        vec3 w_i;
        const vec4 env = sample_environment(rng2(seed), w_i);
        if (env.w > 0) {
            const float f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float Tr = transmittance(pos + t * dir, w_i, seed);
            Li += throughput * f_p * Tr * env.rgb / env.w;
        }
        // backprop transmittance
        const vec3 weight = throughput * Li * dy / tr_pdf;
        const vec3 Lr = vec3(transmittance_adjoint(pos, dir, seed, sanitize(weight), t));
        result += Lr;

        // russian roulette
        const float rr_val = luma(throughput);
        if (rr_val < .1f) {
            const float prob = 1 - rr_val;
            if (rng(seed) < prob) break;
            throughput /= 1 - prob;
        }

        // advance and scatter ray
        pos += t * dir;
        dir = sample_phase_henyey_greenstein(dir, vol_phase_g, rng2(seed));
    }
    return result;
}

// ---------------------------------------------------
// combined forward + backprop

vec3 trace_path_backprop(vec3 pos, vec3 dir, inout uint seed) {
    // storage for saved path segments
    const uint N_PATH_SEGMENTS = 3;
    PathSegment segments[N_PATH_SEGMENTS];
    // trace path
    vec3 radiance = vec3(0), throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, tau, tr_pdf, f_p;
    while (sample_volume_backprop(pos, dir, t, tau, tr_pdf, throughput, seed)) {
        // sample light source (environment)
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(seed), w_i);
        if (Li_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = power_heuristic(Li_pdf.w, f_p);
            const float Tr = transmittance(pos + t * dir, w_i, seed);
            const vec3 Li = throughput * mis_weight * f_p * Tr * Li_pdf.rgb / Li_pdf.w;
            radiance += Li;
            if (n_paths < N_PATH_SEGMENTS) segments[n_paths].store(pos, dir, 0.f, t, Li / tr_pdf, tau); // TODO t_min
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

        // advance and scatter ray
        pos += t * dir;
        const vec3 scatter_dir = sample_phase_henyey_greenstein(dir, vol_phase_g, rng2(seed));
        f_p = phase_henyey_greenstein(dot(-dir, scatter_dir), vol_phase_g);
        dir = scatter_dir;
    }

    // free path? -> add envmap contribution
    if (free_path && show_environment > 0) {
        const vec3 Le = lookup_environment(dir);
        const float mis_weight = n_paths > 0 ? power_heuristic(f_p, pdf_environment(dir)) : 1.f;
        const vec3 Li = throughput * mis_weight * Le;
        radiance += Li;
        if (n_paths < N_PATH_SEGMENTS) segments[n_paths].store(pos - t * dir, dir, 0.f, t, Li / tr_pdf, tau);
    }

    // at this point, we can compute the loss and backprop using the saved path segments
    // TODO
    for (int i = 0; i < min(N_PATH_SEGMENTS, n_paths); ++i) {
        
    }

    return radiance;
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
    const vec3 Lo = trace_path(pos, dir, seed);

    // compute gradient of l2 loss between Lo and reference
    const vec3 reference = imageLoad(color_reference, pixel).rgb;
    const vec3 dy = 2 * (Lo - reference);
    
    // radiative backprop TODO re-use paths
    seed = tea(seed * (pixel.y * resolution.x + pixel.x), current_sample, 32);
    const vec3 Lr = radiative_backprop(pos, dir, seed, dy / float(sppx));

    // store results
    imageStore(color_prediction, pixel, vec4(mix(imageLoad(color_prediction, pixel).rgb, sanitize(Lo), 1.f / current_sample), 1));
    imageStore(color_backprop, pixel, vec4(mix(imageLoad(color_backprop, pixel).rgb, sanitize(Lr), 1.f / current_sample), 1));
}
