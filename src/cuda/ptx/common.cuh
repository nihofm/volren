#pragma once

#include <optix.h>
#include "device_helpers.cuh"

struct Params
{
    // fbo data
    float4* image;
    float2 resolution;
    
    // camera data
    float3 cam_pos;
    float3 cam_dir;
    float cam_fov;
    // TODO lens/DoF params

    // volume data
    float3 vol_bb_min;
    float3 vol_bb_max;
};

struct RayGenData
{
    float r, g, b;
};

struct CoordinateFrame {
    inline __device__ CoordinateFrame() {}

    inline __device__ CoordinateFrame(const float3& v) {
        normal = normalize(v);
        bitangent = fabs(normal.x) > fabs(normal.y) ?
            make_float3(-normal.y, normal.x, 0) :
            make_float3(0, -normal.z, normal.y);
        bitangent = normalize(bitangent);
        tangent = cross(bitangent, normal);
    }

    inline __device__ float3 transform(const float3& v) {
        return make_float3(dot(v, tangent), dot(v, bitangent), dot(v, normal)); // TODO check
    }

    inline __device__ float3 inv_transform(const float3& v) {
        return v.x * tangent + v.y * bitangent + v.z * normal; // TODO check
    }

    float3 normal;
    float3 tangent;
    float3 bitangent;
};