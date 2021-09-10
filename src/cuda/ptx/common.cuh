#pragma once

#include <optix.h>

struct Params
{
    // fbo data
    float4* image;
    uint2 resolution;
    // camera data
    float3 cam_pos;
    float3 cam_dir;
    float cam_fov;
};

struct RayGenData
{
    float r,g,b;
};