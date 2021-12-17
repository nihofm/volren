#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// ------------------------------------------
// Generic buffer using managed memory

template <typename T> class BufferCUDA {
public:
    BufferCUDA() : size(make_cudaExtent(0, 0, 0)), ptr(0) {}
    BufferCUDA(cudaExtent size) : BufferCUDA() {
        resize(size);
    }
    BufferCUDA(size_t width) : BufferCUDA(make_cudaExtent(width, 1, 1)) {}
    BufferCUDA(size_t width, size_t height) : BufferCUDA(make_cudaExtent(width, height, 1)) {}
    BufferCUDA(size_t width, size_t height, size_t depth) : BufferCUDA(make_cudaExtent(width, height, depth)) {}

    // explicit memory management: malloc
    __host__ void alloc(size_t bytes) {
        cudaCheckError(cudaMallocManaged(&ptr, bytes));
    }

    // explicit memory management: free
    __host__ void free() {
        cudaCheckError(cudaFree(ptr));
        ptr = 0;
    }

    __host__ void resize(cudaExtent size) {
        this->size = size;
        if (ptr) free();
        const size_t bytes = size.width * size.height * size.depth * sizeof(T);
        if (bytes > 0) alloc(bytes);
    }

    __host__ __device__ inline operator T*() { return ptr; }
    __host__ __device__ inline operator T*() const { return ptr; }

    __host__ __device__ inline T* operator ->() { return ptr; }
    __host__ __device__ inline T* operator ->() const { return ptr; }

    __host__ __device__ inline T& operator[](const size_t& at) { return ptr[at]; }
    __host__ __device__ inline const T& operator[](const size_t& at) const { return ptr[at]; }

    __host__ __device__ inline T& operator[](const uint3& at) { return ptr[coord_to_index(at)]; }
    __host__ __device__ inline const T& operator[](const uint3& at) const { return ptr[coord_to_index(at)]; }

    __host__ __device__ inline size_t coord_to_index(const uint3& v) const {
        return v.z * size.width * size.height + v.y * size.width + v.x;
    }

    __host__ __device__ inline uint3 index_to_coord(size_t idx) const {
        return make_uint3(idx % size.width, (idx / size.width) % size.height, idx / (size.width * size.height));
    }

    // data
    cudaExtent size;
    T* ptr;
};