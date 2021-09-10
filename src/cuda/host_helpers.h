#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <nvrtc.h>
#include <optix.h>

#include <glm/glm.hpp>

// ------------------------------------------
// error checking helpers

#define cudaCheckError(code) { _cudaCheckError(code, __FILE__, __LINE__); }
inline void _cudaCheckError(cudaError_t code, const std::string& file, int line, bool abort=true) {
    if (code != cudaSuccess)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): CUDA error: " + cudaGetErrorString(code));
}

#define optixCheckError(code) { _optixCheckError(code, __FILE__, __LINE__); }
inline void _optixCheckError(OptixResult code, const std::string& file, int line, bool abort=true) {
    if (code != OPTIX_SUCCESS)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): OPTIX error: " + std::to_string(uint32_t(code)));
}

#define nvrtcCheckError(code) { _nvrtcCheckError(code, __FILE__, __LINE__); }
inline void _nvrtcCheckError(nvrtcResult code, const std::string& file, int line, bool abort = true) {           
    if (code != NVRTC_SUCCESS)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): NVRTC error: " + nvrtcGetErrorString(code));
}

// ------------------------------------------
// glm vector casts

inline float3 cast(const glm::vec3& v) { return make_float3(v.x, v.y, v.z); }
inline float4 cast(const glm::vec4& v) { return make_float4(v.x, v.y, v.z, v.w); }
inline uint3 cast(const glm::uvec3& v) { return make_uint3(v.x, v.y, v.z); }
inline dim3 cast_dim(const glm::uvec3& v) { return dim3(v.x, v.y, v.z); }

// ------------------------------------------
// Buffer

template <typename T> class BufferCUDA {
public:
    BufferCUDA(dim3 size = {1, 1, 1}) : data(0) {
        resize(size);
    }

    BufferCUDA(const BufferCUDA&) = delete;
    BufferCUDA& operator=(const BufferCUDA&) = delete;
    BufferCUDA& operator=(const BufferCUDA&&) = delete;

    __host__ ~BufferCUDA() {
        if (data) cudaCheckError(cudaFree(data));
    }

    __host__ void resize(const dim3& size) {
        this->size = size;
        if (data) cudaCheckError(cudaFree(data));
        cudaCheckError(cudaMallocManaged(&data, size.x * size.y * size.z * sizeof(T)));
    }

    __host__ __device__ inline operator T*() { return data; }
    __host__ __device__ inline operator T*() const { return data; }

    __host__ __device__ inline T* operator ->() { return data; }
    __host__ __device__ inline T* operator ->() const { return data; }

    __host__ __device__ inline T& operator[](const uint32_t& at) { return data[at]; }
    __host__ __device__ inline const T& operator[](const uint32_t& at) const { return data[at]; }

    __host__ __device__ inline T& operator[](const uint3& at) { return data[linear_index(at)]; }
    __host__ __device__ inline const T& operator[](const uint3& at) const { return data[linear_index(at)]; }

    __host__ __device__ inline size_t linear_index(const uint3& v) const { return v.z * size.x * size.y + v.y * size.x + v.x; }
    __host__ __device__ inline uint3 linear_coord(size_t idx) const { return make_uint3(idx % size.x, (idx / size.x) % size.y, idx / (size.x * size.y)); }

    // data
    dim3 size;
    T* data;
};

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