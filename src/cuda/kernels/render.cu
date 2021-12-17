#include "device_helpers.cuh"
#include "../buffer.cuh"

extern "C" __global__
void trace_kernel(float4* fbo, uint2 resolution, float3 cam_pos, float3 cam_dir, float cam_fovy)
{
    const dim3 gid = blockIdx * blockDim + threadIdx;
    if (gid.x >= resolution.x || gid.y >= resolution.y) return;
    const size_t idx = gid.y * resolution.x + gid.x;

    const float3 pos = cam_pos;
    const float3 dir = view_dir(cam_dir, cam_fovy, {gid.x, gid.y}, resolution);

    fbo[idx] = make_float4(dir.x, dir.y, dir.z, 1);
}

extern "C" void call_trace_kernel(float4* fbo, uint2 resolution, float3 cam_pos, float3 cam_dir, float cam_fovy) {
    const dim3 threads = { 32, 32 };
    const dim3 blocks = { (resolution.x + threads.x - 1) / threads.x, (resolution.y + threads.y - 1) / threads.y };
    trace_kernel<<<blocks, threads>>>(fbo, resolution, cam_pos, cam_dir, cam_fovy);
}