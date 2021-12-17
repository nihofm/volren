#pragma once

#include <cuda_runtime.h>

extern "C" void call_trace_kernel(float4* fbo, uint2 resolution, float3 cam_pos, float3 cam_dir, float cam_fovy);