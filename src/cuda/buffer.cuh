#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "host_helpers.h"

// ------------------------------------------
// Generic CUDA buffer for host and device

template <typename T> class BufferCUDA {
public:
    __host__ __device__ BufferCUDA() : size(0), ptr(0) {}
    __host__ __device__ BufferCUDA(const glm::uvec3& size, T* ptr) : size(size), ptr(ptr) {}

    __host__ __device__ inline operator T*() { return ptr; }
    __host__ __device__ inline operator T*() const { return ptr; }

    __host__ __device__ inline T* operator ->() { return ptr; }
    __host__ __device__ inline T* operator ->() const { return ptr; }

    __host__ __device__ inline T& operator[](size_t x) { return ptr[x]; }
    __host__ __device__ inline const T& operator[](size_t x) const { return ptr[x]; }

    __host__ __device__ inline T& operator[](const glm::uvec2& xy) { return ptr[size_t(xy.y) * size.x + xy.x]; }
    __host__ __device__ inline const T& operator[](const glm::uvec2& xy) const { return ptr[size_t(xy.y) * size.x + xy.x]; }

    __host__ __device__ inline T& operator[](const glm::uvec3& xyz) { return ptr[size_t(xyz.z) * size.x * size.y + xyz.y * size.x + xyz.x]; }
    __host__ __device__ inline const T& operator[](const glm::uvec3& xyz) const { return ptr[size_t(xyz.z) * size.x * size.y + xyz.y * size.y + xyz.x]; }

    __host__ __device__ inline T& operator()(uint32_t x) { return ptr[x]; }
    __host__ __device__ inline const T& operator()(uint32_t x) const { return ptr[x]; }

    __host__ __device__ inline T& operator()(uint32_t x, uint32_t y) { return ptr[size_t(y) * size.x + x]; }
    __host__ __device__ inline const T& operator()(uint32_t x, uint32_t y) const { return ptr[size_t(y) * size.x + x]; }

    __host__ __device__ inline T& operator()(uint32_t x, uint32_t y, uint32_t z) { return ptr[size_t(z) * size.x * size.y + y * size.x + x]; }
    __host__ __device__ inline const T& operator()(uint32_t x, uint32_t y, uint32_t z) const { return ptr[size_t(z) * size.x * size.y + y * size.x + x]; }

    __host__ __device__ inline size_t coord_to_index(const glm::uvec3& xyz) const {
        return size_t(xyz.z) * size.x * size.y + xyz.y * size.x + xyz.x;
    }

    __host__ __device__ inline glm::uvec3 index_to_coord(size_t idx) const {
        return glm::uvec3(idx % size.x, (idx / size.x) % size.y, idx / (size.x * size.y));
    }

    __host__ __device__ inline size_t n_elems() const { return size_t(size.x) * size.y * size.z; }
    __host__ __device__ inline size_t n_bytes() const { return sizeof(T) * n_elems(); }

    __host__ __device__ inline uint32_t width() const { return size.x; }
    __host__ __device__ inline uint32_t height() const { return size.y; }
    __host__ __device__ inline uint32_t depth() const { return size.z; }

    // data
    glm::uvec3 size;
    T* ptr;
};

// ------------------------------------------
// Generic CUDA buffer using managed memory

template <typename T> class ManagedBufferCUDA : public BufferCUDA<T> {
public:
    ManagedBufferCUDA() : BufferCUDA<T>() {}
    ManagedBufferCUDA(size_t width) : ManagedBufferCUDA(width, 1, 1) {}
    ManagedBufferCUDA(size_t width, size_t height) : ManagedBufferCUDA(width, height, 1) {}
    ManagedBufferCUDA(size_t width, size_t height, size_t depth) : ManagedBufferCUDA() {
        resize(width, height, depth);
    }
    ManagedBufferCUDA(const glm::uvec3& size) : ManagedBufferCUDA(size.x, size.y, size.z) {}
    ManagedBufferCUDA(cudaExtent size) : ManagedBufferCUDA(size.width, size.height, size.depth) {}

    ~ManagedBufferCUDA() {
        if (this->ptr) cudaCheckError(cudaFree(this->ptr));
    }

    ManagedBufferCUDA(const ManagedBufferCUDA& other) = delete;
    void operator=(const ManagedBufferCUDA& other) = delete;

    inline operator BufferCUDA<T>() const { return BufferCUDA<T>(this->size, this->ptr); }

    inline void resize(uint32_t width, uint32_t height = 1, uint32_t depth = 1) {
        this->size = glm::uvec3(width, height, depth);
        if (this->ptr) cudaCheckError(cudaFree(this->ptr));
        cudaCheckError(cudaMallocManaged(&this->ptr, this->n_bytes()));
    }
};

// ------------------------------------------
// GL_SHADER_STORAGE_BUFFER backed CUDA buffer

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

template <typename T> class BufferCUDAGL : public BufferCUDA<T> {
public:
    BufferCUDAGL() : BufferCUDA<T>(), gl_buf(0), resource(0) {
        glGenBuffers(1, &gl_buf);
    }
    BufferCUDAGL(uint32_t width) : BufferCUDAGL(width, 1, 1) {}
    BufferCUDAGL(uint32_t width, uint32_t height) : BufferCUDAGL(width, height, 1) {}
    BufferCUDAGL(uint32_t width, uint32_t height, uint32_t depth) : BufferCUDAGL() {
        resize(width, height, depth);
    }
    BufferCUDAGL(const glm::uvec3& size) : BufferCUDAGL(size.x, size.y, size.z) {}
    BufferCUDAGL(cudaExtent size) : BufferCUDAGL(size.width, size.height, size.depth) {}

    ~BufferCUDAGL() {
        if (this->ptr) cudaCheckError(cudaGraphicsUnmapResources(1, &resource));
        if (resource) cudaCheckError(cudaGraphicsUnregisterResource(resource));
        glDeleteBuffers(1, &gl_buf);
    }

    BufferCUDAGL(const BufferCUDAGL& other) = delete;
    void operator=(const BufferCUDAGL& other) = delete;

    inline operator BufferCUDA<T>() const { return BufferCUDA<T>(this->size, this->ptr); }

    inline void resize(uint32_t width, uint32_t height = 1, uint32_t depth = 1) {
        this->size = glm::uvec3(width, height, depth);
        if (this->ptr) cudaCheckError(cudaGraphicsUnmapResources(1, &resource));
        if (resource) cudaCheckError(cudaGraphicsUnregisterResource(resource));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gl_buf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->n_bytes(), 0, GL_DYNAMIC_DRAW);
        cudaCheckError(cudaGraphicsGLRegisterBuffer(&resource, gl_buf, cudaGraphicsMapFlagsWriteDiscard));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        size_t n_bytes;
        cudaCheckError(cudaGraphicsMapResources(1, &resource));
        cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&this->ptr, &n_bytes, resource));
    }

    // data
    GLuint gl_buf;
    struct cudaGraphicsResource* resource;
};

// ------------------------------------------
// TODO: GL_TEXTURE_2D backed CUDA buffer
// TODO: cudaArray to T*?
// TODO: always mapped
// TODO: CUDA texture/surface objects

class ImageCUDAGL : public BufferCUDA<glm::vec4> {
public:
    ImageCUDAGL() : BufferCUDA<glm::vec4>(), gl_img(0), resource(0) { // TODO: warning
        glGenTextures(1, &gl_img);
        glBindTexture(GL_TEXTURE_2D, gl_img);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    ImageCUDAGL(uint32_t width) : ImageCUDAGL(width, 1) {}
    ImageCUDAGL(uint32_t width, uint32_t height) : ImageCUDAGL() {
        resize(width, height);
    }
    ImageCUDAGL(const glm::uvec2& size) : ImageCUDAGL(size.x, size.y) {}
    ImageCUDAGL(cudaExtent size) : ImageCUDAGL(size.width, size.height) {}

    ~ImageCUDAGL() {
        if (resource) cudaCheckError(cudaGraphicsUnregisterResource(resource));
        glDeleteTextures(1, &gl_img);
    }

    ImageCUDAGL(const ImageCUDAGL&) = delete;
    ImageCUDAGL& operator=(const ImageCUDAGL&) = delete;

    void resize(uint32_t width, uint32_t height) {
        this->size = glm::uvec3(width, height, 1);
        if (array) cudaCheckError(cudaGraphicsUnmapResources(1, &resource));
        if (resource) cudaCheckError(cudaGraphicsUnregisterResource(resource));
        glBindTexture(GL_TEXTURE_2D, gl_img);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
        cudaCheckError(cudaGraphicsGLRegisterImage(&resource, gl_img, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
        glBindTexture(GL_TEXTURE_2D, 0);
        cudaCheckError(cudaGraphicsMapResources(1, &resource));
        cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));//index, miplevel));
    }

    cudaArray_t map_cuda(uint32_t index = 0, uint32_t miplevel = 0) {
        // cudaArray_t array;
        cudaCheckError(cudaGraphicsMapResources(1, &resource));
        cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&array, resource, index, miplevel));
        return array;
    }

    void unmap_cuda() {
        cudaCheckError(cudaGraphicsUnmapResources(1, &resource));
    }

    // data
    GLuint gl_img;
    struct cudaGraphicsResource* resource;
    cudaArray_t array;
};