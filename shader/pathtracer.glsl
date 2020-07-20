#version 450 core

layout (local_size_x = 32, local_size_y = 32) in;

layout (binding = 0, rgba32f) uniform image2D color;
layout (binding = 1, rgba32f) uniform image2D even;
//layout (binding = 2, rgba32f) uniform image2D f_pos;
//layout (binding = 3, rgba32f) uniform image2D f_norm;
//layout (binding = 4, rgba32f) uniform image2D f_alb;
//layout (binding = 5, rgba32f) uniform image2D f_vol;

#include "common.h"

uniform int current_sample;

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 size = imageSize(color);
	if (any(greaterThanEqual(pixel, size))) return;

    // setup random seed and camera ray (in model space!)
    uint seed = tea(pixel.y * size.x + pixel.x, current_sample, 8);
    vec3 pos = world_to_vol(vec4(cam_pos, 1));
    vec3 dir = world_to_vol(view_dir(pixel, size, rng2(seed)));

    // trace ray
    const vec3 radiance = trace_path(pos, dir, seed);

    // write output
    if (any(isnan(radiance)) || any(isinf(radiance))) return;
    imageStore(color, pixel, vec4(mix(imageLoad(color, pixel).rgb, radiance, 1.f / current_sample), 1));
    if (current_sample % 2 == 1)
        imageStore(even, pixel, vec4(mix(imageLoad(even, pixel).rgb, radiance, 1.f / ((current_sample+ 1) / 2)), 1));
}
