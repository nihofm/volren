#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color_adjoint;
layout (binding = 1, rgba32f) uniform image2D color_reference;
layout (binding = 2, r32f) uniform image3D gradients;
layout (binding = 3, rgba32f) uniform image2D color_debug;

uniform int current_sample;
uniform int bounces;
uniform int seed;
uniform int show_environment;

#include "common.glsl"

// ---------------------------------------------------
// helper funcs

float sum(const vec3 x) { return (x.x + x.y + x.z); }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }
vec3 visualize_grad(const float grad) { return abs(grad) * (sign(grad) > 0.f ? vec3(1, 0, 0) : vec3(0, 0, 1)); }

vec3 trace_bounce(vec3 pos, vec3 dir, inout uint seed) {
    vec3 L = vec3(0), throughput = vec3(1);
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    const bool collision = sample_volume(pos, dir, t, throughput, seed);
    if (collision) {
        // advance ray
        pos += t * dir;
        // sample light source (environment)
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(seed), w_i);
        if (Li_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float weight = power_heuristic(Li_pdf.w, f_p);
            const float Tr = transmittance(pos, w_i, seed);
            L += throughput * weight * f_p * Tr * Li_pdf.rgb / Li_pdf.w;
        }
    } else if (show_environment == 0) // TODO logic
        L += throughput * lookup_environment(dir);
    return L;
}

// ---------------------------------------------------
// adjoint funcs

float transmittance_raymarch_adjoint(const vec3 wpos, const vec3 wdir, inout uint seed, const vec3 dy) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    const int steps = 64;
    const float dt = (near_far.y - near_far.x) / float(steps);
    // integrate density
    float t0 = near_far.x + rng(seed) * dt, tau = 0.f;
    for (int i = 0; i < steps; ++i) {
        const ivec3 curr = ivec3(ipos + (t0 + i * dt) * idir);
        tau += lookup_density(curr, seed) * dt;
    }
    // store gradients
    const float y = exp(-tau);
    const float dx = -y * sum(dy) * dt;
    for (int i = 0; i < steps; ++i)
        imageAtomicAdd(gradients, ivec3(ipos + (t0 + i * dt) * idir), dx);
    return y;
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
    // TODO full pathtracing
    const float Tr = transmittance_raymarch_adjoint(pos, dir, seed, dy);

    // DEBUG
    vec3 out_col = dy;
    imageStore(color_debug, pixel, vec4(sanitize(out_col), 1));
}
