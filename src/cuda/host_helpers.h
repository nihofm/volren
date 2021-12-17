#pragma once

#include <string>
#include <exception>
#include <glm/glm.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <optix.h>

// ------------------------------------------
// error checking helpers

#define cuCheckError(code) { _cuCheckError(code, __FILE__, __LINE__); }
inline void _cuCheckError(CUresult code, const std::string& file, int line, bool abort=true) {
    if (code != CUDA_SUCCESS) {
        const char* name; cuGetErrorName(code, &name);
        const char* msg; cuGetErrorString(code, &msg);
        throw std::runtime_error(file + "(" + std::to_string(line) + "): CU error: " + msg + " (" + name + ")");
    }
}

#define cudaCheckError(code) { _cudaCheckError(code, __FILE__, __LINE__); }
inline void _cudaCheckError(cudaError_t code, const std::string& file, int line, bool abort=true) {
    if (code != cudaSuccess)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): CUDA error: " + cudaGetErrorString(code));
}

#define nvrtcCheckError(code) { _nvrtcCheckError(code, __FILE__, __LINE__); }
inline void _nvrtcCheckError(nvrtcResult code, const std::string& file, int line, bool abort = true) {           
    if (code != NVRTC_SUCCESS)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): NVRTC error: " + nvrtcGetErrorString(code));
}

#define optixCheckError(code) { _optixCheckError(code, __FILE__, __LINE__); }
inline void _optixCheckError(OptixResult code, const std::string& file, int line, bool abort=true) {
    if (code != OPTIX_SUCCESS)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): OPTIX error: " + std::to_string(uint32_t(code)));
}

// ------------------------------------------
// cuda init and default context

inline void cuda_init() {
    cudaCheckError(cudaFree(0));
}

inline std::pair<CUdevice, CUcontext> cuda_default_context() {
    cuCheckError(cuInit(0));
    CUdevice device;
    cuCheckError(cuDeviceGet(&device, 0));
    CUcontext context;
    cuCheckError(cuCtxCreate(&context, 0, device));
    return { device, context };
}

// ------------------------------------------
// glm vector casts

inline float3 cast(const glm::vec3& v) { return make_float3(v.x, v.y, v.z); }
inline float4 cast(const glm::vec4& v) { return make_float4(v.x, v.y, v.z, v.w); }
inline uint3 cast(const glm::uvec3& v) { return make_uint3(v.x, v.y, v.z); }
inline dim3 cast_dim(const glm::uvec3& v) { return dim3(v.x, v.y, v.z); }

/*

// ------------------------------------------
// Ray

struct RayCUDA {
    __host__ __device__ RayCUDA() {}
    __host__ __device__ RayCUDA(const float3& origin, const float3& direction, float near = 0.f, float far = 1e+38f) : pos(origin), near(near), dir(direction), far(far) {}

    __device__ float3 operator()(float t) const {
        return pos + t * dir;
    }

    __device__ bool intersect(const float3& bb_min, const float3& bb_max, float2& near_far) const {
        return intersect_box(pos, dir, bb_min, bb_max, near_far);
    }

    __device__ bool intersect(const float3& center, float radius, float2& near_far) const {
        return intersect_sphere(pos, dir, center, radius, near_far);
    }

    // data
    float3 pos;
    float near;
    float3 dir;
    float far;
};

// ------------------------------------------
// Camera

struct CameraCUDA {
    __host__ __device__ CameraCUDA() {}
    __host__ __device__ CameraCUDA(const float3& pos, const float3& dir, float fov) : pos(pos), dir(dir), fov(fov), flags(0) {}

    __device__ float3 to_world(const float3& v) const {
        const float3 ndir = normalize(dir);
        const float3 left = normalize(cross(ndir, make_float3(0, 1, 0)));
        const float3 up = cross(left, ndir);
        return normalize(v.x * left + v.y * up + v.z * -dir);
    }

    __device__ float3 to_tangent(const float3& v) const {
        const float3 ndir = normalize(dir);
        const float3 left = cross(ndir, make_float3(0, 1, 0));
        const float3 up = cross(left, ndir);
        return make_float3(dot(v, left), dot(v, up), dot(v, -dir));
    }

    __device__ RayCUDA view_ray(int2 xy, int2 wh, float2 sample = make_float2(.5f)) const {
        const float2 pixel = (make_float2(xy) + sample - make_float2(wh) * .5f) / make_float2(wh.y);
        const float z = -.5f / tanf(.5f * M_PI * fov / 180.f);
        return RayCUDA(pos, to_world(make_float3(pixel.x, pixel.y, z)));
    }

    // data
    float3 pos;     // camera position
    float3 dir;     // view direction
    float fov;      // fov in degrees
    uint32_t flags; // flags/padding
};


// ------------------------------------------
// Volume TODO

struct VolumeCUDA {
    inline float3 cast(const glm::vec3& v) { return make_float3(v.x, v.y, v.z); }
    inline uint3 cast(const glm::uvec3& v) { return make_uint3(v.x, v.y, v.z); }

    __host__ void commit(const std::shared_ptr<voldata::Volume>& volume) {
        bb_min = cast(volume->AABB().first);
        bb_max = cast(volume->AABB().second);
        albedo = cast(volume->albedo);
        phase = volume->phase;
        density_scale = volume->density_scale;
        majorant = volume->minorant_majorant().second;
        grid = std::make_shared<BufferCUDA<float>(cast(volume->current_grid()->index_extent()));
    }

    __device__ float3 to_index(const float4& v) const {
        return make_float3(dot(transform[0], v), dot(transform[1], v), dot(transform[2], v));
    }

    // transform
    float4 transform[4]; // row-major 4x4 matrix
    // parameters
    float3 bb_min;
    float3 bb_max;
    float3 albedo;
    float phase;
    float density_scale;
    float majorant;
    BufferCUDA<float> grid;
};
*/