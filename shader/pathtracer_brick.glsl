#version 450 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color;

// ---------------------------------------------------
// settings

#define USE_DDA
#include "common.glsl"

// ---------------------------------------------------
// uniforms

uniform int current_sample;
uniform int seed;
uniform ivec2 resolution;

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, resolution))) return;

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * resolution.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir(pixel, resolution, rng2(seed));

    // trace ray
    const vec4 L = trace_path(pos, dir, seed);

    // write result
    imageStore(color, pixel, mix(imageLoad(color, pixel), sanitize(L), 1.f / current_sample));
}
