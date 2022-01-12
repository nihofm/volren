#pragma once

#include <cuda.h>
#include <glm/glm.hpp>

#include "../buffer.cuh"
#include "../ray.cuh"
#include "../aabb.cuh"
#include "../camera.cuh"

inline __device__ glm::vec2 cast(const float2& v) { return glm::vec2(v.x, v.y); }
inline __device__ glm::vec3 cast(const float3& v) { return glm::vec3(v.x, v.y, v.z); }
inline __device__ glm::vec4 cast(const float4& v) { return glm::vec4(v.x, v.y, v.z, v.w); }

inline __device__ glm::vec3 align(const glm::vec3& axis, const glm::vec3& v) {
    const float s = copysignf(1.f, axis.z);
    const glm::vec3 w = glm::vec3(v.x, v.y, v.z * s);
    const glm::vec3 h = glm::vec3(axis.x, axis.y, axis.z + s);
    const float k = glm::dot(w, h) / (1.f + fabsf(axis.z));
    return k * h - w;
}

inline __device__ bool intersect_sphere(const glm::vec3& pos, const glm::vec3& dir, const glm::vec3& center, float radius, glm::vec2& near_far) {
    const glm::vec3 pdir = pos - center;
    const float b = glm::dot(dir, pdir);
    const float c = glm::dot(pdir, pdir) - radius * radius;
    const float h = sqrtf(b * b - c);
    near_far.x = fmaxf(0.f, -b - h);
    near_far.y = fmaxf(0.f, -b + h);
    return near_far.x < near_far.y;
}

inline __device__ bool intersect_box(const glm::vec3& pos, const glm::vec3& dir, const glm::vec3& bb_min, const glm::vec3& bb_max, glm::vec2& near_far) {
    const glm::vec3 inv_dir = 1.f / dir;
    const glm::vec3 lo = (bb_min - pos) * inv_dir;
    const glm::vec3 hi = (bb_max - pos) * inv_dir;
    const glm::vec3 tmin = glm::min(lo, hi), tmax = glm::max(lo, hi);
    near_far.x = fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z)));
    near_far.y = fminf(tmax.x, fminf(tmax.y, tmax.z));
    return near_far.x < near_far.y;
}

inline __device__ glm::vec3 view_dir(const glm::vec3& cam_dir, float cam_fov, const glm::vec2& pixel, const glm::vec2& wh, const glm::vec2& pixel_sample = glm::vec2(.5f)) {
    // compute eye-space direction
    const glm::vec2 xy = (pixel + pixel_sample - wh * .5f) / wh.y;
    const float z = .5f / tanf(.5f * M_PI * cam_fov / 180.f);
    // transform to view space
    const glm::vec3 ndir = glm::normalize(cam_dir);
    const glm::vec3 left = glm::normalize(glm::cross(ndir, glm::vec3(0, 1, 0) + 1e-6f));
    const glm::vec3 up = glm::cross(left, ndir);
    return glm::normalize(xy.x * left + xy.y * up + z * cam_dir);
}