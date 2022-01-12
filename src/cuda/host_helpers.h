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
#ifndef __CUDACC__
    if (code != CUDA_SUCCESS) {
        const char* name; cuGetErrorName(code, &name);
        const char* msg; cuGetErrorString(code, &msg);
        throw std::runtime_error(file + "(" + std::to_string(line) + "): CU error: " + msg + " (" + name + ")");
    }
#endif
}

#define cudaCheckError(code) { _cudaCheckError(code, __FILE__, __LINE__); }
inline void _cudaCheckError(cudaError_t code, const std::string& file, int line, bool abort=true) {
#ifndef __CUDACC__
    if (code != cudaSuccess)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): CUDA error: " + cudaGetErrorString(code));
#endif
}

#define nvrtcCheckError(code) { _nvrtcCheckError(code, __FILE__, __LINE__); }
inline void _nvrtcCheckError(nvrtcResult code, const std::string& file, int line, bool abort = true) {
#ifndef __CUDACC__
    if (code != NVRTC_SUCCESS)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): NVRTC error: " + nvrtcGetErrorString(code));
#endif
}

#define optixCheckError(code) { _optixCheckError(code, __FILE__, __LINE__); }
inline void _optixCheckError(OptixResult code, const std::string& file, int line, bool abort=true) {
#ifndef __CUDACC__
    if (code != OPTIX_SUCCESS)
        throw std::runtime_error(file + "(" + std::to_string(line) + "): OPTIX error: " + std::to_string(uint32_t(code)));
#endif
}

// ------------------------------------------
// cuda init and default context

inline void cuda_init() {
    cudaCheckError(cudaFree(0));
}

inline std::pair<CUdevice, CUcontext> cu_default_context() {
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
