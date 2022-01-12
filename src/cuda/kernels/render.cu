#include "render.h"
#include "device_helpers.cuh"
#include "../rng.cuh"

__global__ void trace_kernel(BufferCUDA<glm::vec4> fbo, CameraCUDA cam, DenseGridCUDA grid, uint32_t sample) {
    // const uint3 gid = blockIdx * blockDim + threadIdx;
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= fbo.size.x || y >= fbo.size.y) return;

    uint32_t seed = tea(y * fbo.size.x + x, sample, 32);

    Ray ray = cam.view_ray_pinhole(glm::vec2(x, y), glm::vec2(fbo.size.x, fbo.size.y));
    // Ray ray = Ray(cam.pos, view_dir(cam.dir, cam.fovy, glm::vec2(x, y), glm::vec2(fbo.size.x, fbo.size.y)));

    glm::vec4 color = glm::vec4(grid.transmittance(ray, seed));
    fbo(x, y) = glm::mix(fbo(x, y), color, 1.f / sample);
}

extern "C" void call_trace_kernel(BufferCUDA<glm::vec4> fbo, CameraCUDA cam, DenseGridCUDA grid, uint32_t sample) {
    const dim3 threads = { 32, 32 };
    const dim3 blocks = { (fbo.size.y + threads.x - 1) / threads.x, (fbo.size.y + threads.y - 1) / threads.y };
    trace_kernel<<<blocks, threads>>>(fbo, cam, grid, sample);
}