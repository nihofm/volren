#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color_adjoint;
layout (binding = 1, rgba32f) uniform image2D color_reference;
layout (binding = 2, r32f) uniform image3D gradients;
layout (binding = 3, rgba32f) uniform image2D color_output;

uniform int current_sample;
uniform int bounces;
uniform int seed;
uniform int show_environment;

#include "common.h"

// ---------------------------------------------------
// helper funcs

float mean(const vec3 x) { return (x.x + x.y + x.z) * (1.f / 3.f); }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }
vec3 visualize_grad(const float grad) { return abs(grad) * (sign(grad) > 0.f ? vec3(1, 0, 0) : vec3(0, 0, 1)); }

// ---------------------------------------------------
// adjoint funcs

vec3 view_dir_adjoint(const ivec2 xy, const ivec2 wh, const vec2 pixel_sample, inout float dy) {
    // TODO forward propagate dy
    const vec2 pixel = (xy + pixel_sample - wh * .5f) / vec2(wh.y);
    const float z = -.5f / tan(.5f * M_PI * cam_fov / 180.f);
    return normalize(cam_transform * normalize(vec3(pixel.x, pixel.y, z)));
}

bool intersect_box_adjoint(const vec3 pos, const vec3 dir, const vec3 bb_min, const vec3 bb_max, out vec2 near_far, inout float dy) {
    // TODO forward propagate dy
    const vec3 inv_dir = 1.f / dir;
    const vec3 lo = (bb_min - pos) * inv_dir;
    const vec3 hi = (bb_max - pos) * inv_dir;
    const vec3 tmin = min(lo, hi), tmax = max(lo, hi);
    near_far.x = max(0.f, max(tmin.x, max(tmin.y, tmin.z)));
    near_far.y = min(tmax.x, min(tmax.y, tmax.z));
    return near_far.x <= near_far.y;
}

float transmittance_adjoint(const vec3 wpos, const vec3 wdir, inout uint seed, inout float dy) {
    // clip volume
    vec2 near_far;
    if (!intersect_box_adjoint(wpos, wdir, vol_bb_min, vol_bb_max, near_far, dy)) return 1.f;
    // TODO forward propagate dy
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // ratio tracking
    float t = near_far.x, Tr = 1.f;
    while (t < near_far.y) {
        t -= log(1 - rng(seed)) * vol_inv_majorant;
        const ivec3 curr = ivec3(ipos + t * idir);
        const float d = lookup_density(curr, seed);
        Tr *= max(0.f, 1 - d * vol_inv_majorant);

        // gather gradients
        imageAtomicAdd(gradients, curr, dy);

        // russian roulette
        if (Tr < .1f) {
            const float prob = 1 - Tr;
            if (rng(seed) < prob) return 0.f;
            Tr /= 1 - prob;
        }
    }
    return Tr;
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
    const vec3 l2_grad = 2 * (col_adj - col_ref);
    // adjoint 
    float dy = mean(l2_grad);

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * size.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir_adjoint(pixel, size, rng2(seed), dy);
    if (dy > 0.f) {
        const float Tr = transmittance_adjoint(pos, dir, seed, dy); 
        // const vec3 radiance = trace_path(pos, dir, seed);
    }

    vec3 out_col = visualize_grad(dy);
    // out_col = abs(l2_grad);
    // out_col = abs(col_adj);

    // write output
    imageStore(color_output, pixel, vec4(sanitize(out_col), 1));
}
