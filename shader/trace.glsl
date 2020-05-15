#version 450 core

#include "random.glsl"

layout (local_size_x = 32, local_size_y = 32) in;

layout (binding = 0, rgba32f) uniform image2D color;
layout (binding = 1, rgba32f) uniform image2D f_pos;
layout (binding = 2, rgba32f) uniform image2D f_norm;
layout (binding = 3, rgba32f) uniform image2D f_alb;
layout (binding = 4, rgba32f) uniform image2D f_vol;

uniform int current_sample;
uniform mat4 model;
uniform mat4 inv_model;
uniform sampler3D volume;
uniform vec3 cam_pos;
uniform float cam_fov;
uniform mat3 cam_transform;
uniform sampler2D env_tex;

#define M_PI float(3.14159265358979323846)

vec3 view_dir(const ivec2 xy, const ivec2 wh, const vec2 pixel_sample) {
    const vec2 pixel = (xy + pixel_sample - wh * .5f) / vec2(wh.y);
    const float z = -.5f / tan(.5f * M_PI * cam_fov / 180.f);
    return normalize(cam_transform * normalize(vec3(pixel.x, pixel.y, z)));
}

vec4 environment(const vec3 dir) {
    const float u = atan(dir.z, dir.x) / (2 * M_PI);
    const float v = -acos(dir.y) / M_PI;
    return texture(env_tex, vec2(u, v));
}

bool intersect_box(const vec3 pos, const vec3 dir, const vec3 bb_min, const vec3 bb_max, out vec2 near_far) {
    // TODO fix inside
    const vec3 inv_dir = 1.f / dir;
    const vec3 lo = (bb_min - pos) * inv_dir;
    const vec3 hi = (bb_max - pos) * inv_dir;
    const vec3 tmin = min(lo, hi), tmax = max(lo, hi);
    near_far.x = max(tmin.x, max(tmin.y, tmin.z));
    near_far.y = min(tmax.x, min(tmax.y, tmax.z));
    return max(0.f, near_far.x) <= near_far.y;
}

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 size = imageSize(color);
	if (any(greaterThanEqual(pixel, size))) return;

    // setup view ray (in model space!)
    uint seed = tea(pixel.y * size.x + pixel.x, current_sample, 8);
    const vec3 pos = vec3(inv_model * vec4(cam_pos, 1));
    const vec3 dir = normalize(mat3(inv_model) * view_dir(pixel, size, rng2(seed)));

    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vec3(0.f), vec3(1.f), near_far)) {
        imageStore(color, pixel, environment(normalize(mat3(model) * dir)));
        return;
    }

    // TODO ray march
    const ivec3 voxels = textureSize(volume, 0);
    const float stepsize = 1.f / min(voxels.x, min(voxels.y, voxels.z));

    float t = near_far.x + rng(seed) * stepsize;

    vec3 col = vec3(0);

    while (t < near_far.y) {
        col += texture(volume, pos + t * dir).r * stepsize;
        t += stepsize;
    }

    // write output
    vec3 old = imageLoad(color, pixel).rgb;
    if (any(isnan(col)) || any(isinf(col)))
        col = vec3(0);
    imageStore(color, pixel, vec4(mix(old, col, 1.f / current_sample), 1));
}
