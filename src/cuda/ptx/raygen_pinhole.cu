#include "common.cuh"
#include "device_helpers.cuh"

extern "C" {
__constant__ Params params;
}

inline __device__ float3 pinhole_camera(const float3& cam_dir, float cam_fov, float2 xy, float2 wh, float2 sample = make_float2(.5f)) {
    // compute eye-space direction
    const float2 pixel = (xy + sample - wh * .5f) / make_float2(wh.y);
    const float z = -.5f / tanf(.5f * CUDART_PI_F * cam_fov / 180.f);
    const float3 eye_dir = make_float3(pixel.x, pixel.y, z);
    // transform to view space
    const float3 ndir = normalize(cam_dir);
    const float3 left = normalize(cross(ndir, make_float3(0, 1, 0)));
    const float3 up = cross(left, ndir);
    return normalize(eye_dir.x * left + eye_dir.y * up + eye_dir.z * -cam_dir);
}

extern "C" __global__ void __raygen__pinhole() {
    const uint3 launch_index = optixGetLaunchIndex();
    const float2 pixel = make_float2(launch_index.x, launch_index.y);
    const float3 pos = params.cam_pos;
    const float3 dir = pinhole_camera(params.cam_dir, params.cam_fov, pixel, params.resolution);

    float2 near_far;
    const bool hit = intersect_box(pos, dir, params.vol_bb_min, params.vol_bb_max, near_far);

    params.image[launch_index.y * int(params.resolution.x) + launch_index.x] = 0.001 * make_float4(near_far.x, near_far.y, 0, 1);
}