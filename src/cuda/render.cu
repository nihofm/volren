#include "common.cuh"

__global__ void render_cuda_kernel(float4* fbo, size_t w, size_t h, CameraCUDA* cam, VolumeCUDA* vol) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    RayCUDA ray = cam->view_ray(make_int2(x, y), make_int2(w, h));
    float2 near_far;
    const bool hit = intersect_box(ray.pos, ray.dir, vol->bb_min, vol->bb_max, near_far);
    // to index-space
    const float3 ipos = vol->to_index(make_float4(ray.pos.x, ray.pos.y, ray.pos.z, 1));
    const float3 idir = vol->to_index(make_float4(ray.pos.x, ray.pos.y, ray.pos.z, 0));
    if (hit) {
        fbo[y * w + x] = make_float4(near_far.x / length(vol->bb_max - vol->bb_min), near_far.y / length(vol->bb_max - vol->bb_min), 0, 1);
        fbo[y * w + x] = make_float4(ipos.x, ipos.y, ipos.z, 1); // TODO validate
    } else {
        fbo[y * w + x] = make_float4(ray.dir.x, ray.dir.y, ray.dir.z, 1);
    }
}

void render_cuda(float4* fbo, size_t w, size_t h, const BufferCUDA<CameraCUDA>& cam, const BufferCUDA<VolumeCUDA>& vol) {
    dim3 block(32, 32, 1);
    dim3 grid(w / block.x + 1, h / block.y + 1, 1);
    render_cuda_kernel<<<grid, block>>>(fbo, w, h, cam, vol);
}
