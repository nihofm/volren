#pragma once

// ------------------------------------------
// vector math

#include "helper_math.cuh"

inline __device__ float2 operator-(const float2& v) { return make_float2(-v.x, -v.y); }
inline __device__ float3 operator-(const float3& v) { return make_float3(-v.x, -v.y, -v.z); }
inline __device__ float4 operator-(const float4& v) { return make_float4(-v.x, -v.y, -v.z, -v.w); }

// ------------------------------------------
// helper funcs

inline __device__ float3 align(const float3& axis, const float3& v) {
    const float s = copysignf(1.f, axis.z);
    const float3 w = make_float3(v.x, v.y, v.z * s);
    const float3 h = make_float3(axis.x, axis.y, axis.z + s);
    const float k = dot(w, h) / (1.f + fabsf(axis.z));
    return k * h - w;
}

inline __device__ bool intersect_sphere(const float3& pos, const float3& dir, const float3& center, float radius, float2& near_far) {
    const float3 pdir = pos - center;
    const float b = dot(dir, pdir);
    const float c = dot(pdir, pdir) - radius * radius;
    const float h = sqrtf(b * b - c);
    near_far.x = fmaxf(0.f, -b - h);
    near_far.y = fmaxf(0.f, -b + h);
    return near_far.x < near_far.y;
}

inline __device__ bool intersect_box(const float3& pos, const float3& dir, const float3& bb_min, const float3& bb_max, float2& near_far) {
    const float3 inv_dir = 1.f / dir;
    const float3 lo = (bb_min - pos) * inv_dir;
    const float3 hi = (bb_max - pos) * inv_dir;
    const float3 tmin = fminf(lo, hi), tmax = fmaxf(lo, hi);
    near_far.x = fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z)));
    near_far.y = fmaxf(0.f, fminf(tmax.x, fminf(tmax.y, tmax.z)));
    return near_far.x < near_far.y;
}

inline __device__ float3 view_dir(const float3& cam_dir, float cam_fov, int2 xy, int2 wh, float2 sample = make_float2(.5f)) {
    // compute eye-space direction
    const float2 pixel = (make_float2(xy) + sample - make_float2(wh) * .5f) / make_float2(wh.y);
    const float z = -.5f / tanf(.5f * M_PI * cam_fov / 180.f);
    const float3 eye_dir = make_float3(pixel.x, pixel.y, z);
    // transform to view space
    const float3 ndir = normalize(cam_dir);
    const float3 left = normalize(cross(ndir, make_float3(0, 1, 0)));
    const float3 up = cross(left, ndir);
    return normalize(eye_dir.x * left + eye_dir.y * up + eye_dir.z * -cam_dir);
}