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

void transmittance_raymarch_adjoint(const vec3 wpos, const vec3 wdir, inout uint seed, const vec3 dy, const float t_max = FLT_MAX) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return;
    near_far.y = min(near_far.y, t_max);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    const float tau = integrate_density(ipos, idir, near_far, seed);
    // store gradients
    const float dx = -exp(-tau) * sum(dy) * vol_step_size;
    const int steps = 1 + int(ceil((near_far.y - near_far.x) / vol_step_size));
    float t0 = near_far.x + rng(seed) * vol_step_size;
    for (int i = 1; i < steps; ++i) {
        imageAtomicAdd(gradients, ivec3(ipos + min(t0 + (i - 1) * vol_step_size, near_far.y) * idir), .5f * dx);
        imageAtomicAdd(gradients, ivec3(ipos + min(t0 + (i + 0) * vol_step_size, near_far.y) * idir), .5f * dx);
    }
}

bool sample_volume_raymarch(const vec3 wpos, const vec3 wdir, out float t, out float tr_pdf, inout vec3 throughput, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // compute step size and jitter starting point
    const int steps = int(ceil((near_far.y - near_far.x) / vol_step_size));
    const float sampled_tau = -log(1.f - rng(seed));
    t = near_far.x + rng(seed) * vol_step_size, tr_pdf = 0.f;
    // raymarch
    float tau = 0.f;
    for (int i = 0; i < steps; ++i) {
        const ivec3 curr_p = ivec3(ipos + min(t, near_far.y) * idir);
        const float curr_d = lookup_density(curr_p, seed); // TODO extinction coef?
        tau += curr_d * vol_step_size;
        t += vol_step_size;
        if (tau >= sampled_tau) {
            const float f = (tau - sampled_tau) / curr_d;
            t -= f * vol_step_size;
            tr_pdf = curr_d * exp(-sampled_tau);
            throughput *= vol_albedo;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------
// adjoint funcs


// ---------------------------------------------------
// TODO radiative backprop

vec3 radiative_backprop(vec3 pos, vec3 dir, inout uint seed, const vec3 dy) {
     // TODO multiple bounces
    vec3 throughput = vec3(1);
    float t, tr_pdf;
    const bool collision = sample_volume_raymarch(pos, dir, t, tr_pdf, throughput, seed);
    if (false && collision) {
        // Term: G * Q1 * Li + K * Q2 * Li
        const vec3 sampled_pos = pos + t * dir;
        // approximate (biased) Li
        vec3 Li = vec3(1);
        // emitter sampling
        vec3 w_i;
        const vec4 env = sample_environment(rng2(seed), w_i);
        if (env.w > 0) {
            const float f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = 1.f;//power_heuristic(env.w, f_p);
            const float Tr = transmittance(sampled_pos, w_i, seed);
            Li += throughput * mis_weight * f_p * Tr * env.rgb / env.w;
        }
        // backprop transmittance
        const vec3 weight = vol_albedo * throughput * Li * dy / max(0.001f, tr_pdf);
        transmittance_raymarch_adjoint(pos, dir, seed, sanitize(weight), t);
        return weight;
    } else {
        const vec3 Le = show_environment < 1 ? lookup_environment(dir) : vec3(-1); // TODO ???
        const vec3 weight = Le * dy;
        transmittance_raymarch_adjoint(pos, dir, seed, sanitize(weight));
        return weight;
    }
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
    
    // transmittance_raymarch_adjoint(pos, dir, seed, weight);
    const vec3 L = radiative_backprop(pos, dir, seed, dy);
    imageStore(color_debug, pixel, vec4(mix(imageLoad(color_debug, pixel).rgb, sanitize(L), 1.f / current_sample), 1));
}
