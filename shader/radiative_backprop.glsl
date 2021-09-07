#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color_adjoint;
layout (binding = 1, rgba32f) uniform image2D color_reference;
layout (binding = 2, r32f) uniform image3D gradients;
layout (binding = 3, rgba32f) uniform image2D color_debug;

uniform int current_sample;
uniform int sppx;
uniform int bounces;
uniform int seed;
uniform int show_environment;

#include "common.glsl"

// ---------------------------------------------------
// helper funcs

float sum(const vec3 x) { return (x.x + x.y + x.z); }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }

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
    const float dx = -exp(-tau) * sum(dy) * vol_step_size;
    if (abs(dx) <= 1e-4f) return dx;
    const int steps = 1 + int(ceil((near_far.y - near_far.x) / vol_step_size));
    float t0 = near_far.x + rng(seed) * vol_step_size;
    for (int i = 0; i < steps; ++i)
        imageAtomicAdd(gradients, ivec3(ipos + min(t0 + i * vol_step_size, near_far.y) * idir), dx);
    return dx;
}

// ---------------------------------------------------
// radiative backprop TODO merge with forward render pass and re-use paths

vec3 radiative_backprop(vec3 pos, vec3 dir, inout uint seed, const vec3 dy) {
    vec3 throughput = vec3(1), result = vec3(0);
    for (int i = 0; i < bounces; ++i) {
        // sample volume and compute pdf
        float t, tr_pdf;
        const bool escaped = !sample_volume_raymarch_pdf(pos, dir, t, tr_pdf, throughput, seed);

        // Term: Q2 * Le
        if (escaped && show_environment > 0) {
            const vec3 Le = lookup_environment(dir);
            const vec3 weight = throughput * Le * dy / tr_pdf;
            result += vec3(transmittance_adjoint(pos, dir, seed, sanitize(weight), t));
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
        result += vec3(transmittance_adjoint(pos, dir, seed, sanitize(weight), t));

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
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 size = imageSize(color_adjoint);
    if (any(greaterThanEqual(pixel, size))) return;
 
    // compute gradient of l2 loss between input and reference
    const vec3 col_adj = imageLoad(color_adjoint, pixel).rgb;
    const vec3 col_ref = imageLoad(color_reference, pixel).rgb;
    // derivative of objective
    const vec3 dy = 2 * (col_adj - col_ref);

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * size.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir(pixel, size, rng2(seed));
    
    const vec3 L = radiative_backprop(pos, dir, seed, dy / float(sppx));
    imageStore(color_debug, pixel, vec4(mix(imageLoad(color_debug, pixel).rgb, sanitize(L), 1.f / current_sample), 1));
}
