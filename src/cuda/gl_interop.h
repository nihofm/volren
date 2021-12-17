#pragma once

#include "host_helpers.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

template <typename T> class BufferCUDAGL {
public:
    BufferCUDAGL(dim3 size = {1, 1, 1}) : gl_buf(0), cuda_resource(0) {
        glGenBuffers(1, &gl_buf);
        resize(size);
    }

    BufferCUDAGL(const BufferCUDAGL&) = delete;
    BufferCUDAGL& operator=(const BufferCUDAGL&) = delete;
    BufferCUDAGL& operator=(const BufferCUDAGL&&) = delete;

    ~BufferCUDAGL() {
        if (cuda_resource) cudaCheckError(cudaGraphicsUnregisterResource(cuda_resource));
        glDeleteBuffers(1, &gl_buf);
    }

    void resize(const dim3& size) {
        this->size = size;
        if (cuda_resource) cudaCheckError(cudaGraphicsUnregisterResource(cuda_resource));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gl_buf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * size.x * size.y * size.z, 0, GL_DYNAMIC_DRAW);
        cudaCheckError(cudaGraphicsGLRegisterBuffer(&cuda_resource, gl_buf, cudaGraphicsMapFlagsWriteDiscard));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    T* map_cuda() {
        T* ptr;
        size_t num_bytes; 
        cudaCheckError(cudaGraphicsMapResources(1, &cuda_resource, 0));
        cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, cuda_resource));
        return ptr;
    }

    void unmap_cuda() {
        cudaCheckError(cudaGraphicsUnmapResources(1, &cuda_resource, 0));
    }

    T* map_cpu() {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gl_buf);
        return (T*) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
    }

    void unmap_cpu() {
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // data
    dim3 size;
    GLuint gl_buf;
    struct cudaGraphicsResource* cuda_resource;
};

class ImageCUDAGL {
public:
    ImageCUDAGL(uint2 size = {1, 1}) : gl_img(0), cuda_resource(0) {
        glGenTextures(1, &gl_img);
        glBindTexture(GL_TEXTURE_2D, gl_img);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        resize(size);
    }

    ImageCUDAGL(const ImageCUDAGL&) = delete;
    ImageCUDAGL& operator=(const ImageCUDAGL&) = delete;
    ImageCUDAGL& operator=(const ImageCUDAGL&&) = delete;

    ~ImageCUDAGL() {
        if (cuda_resource) cudaCheckError(cudaGraphicsUnregisterResource(cuda_resource));
        glDeleteTextures(1, &gl_img);
    }

    void resize(const uint2& size) {
        this->size = size;
        if (cuda_resource) cudaCheckError(cudaGraphicsUnregisterResource(cuda_resource));
        glBindTexture(GL_TEXTURE_2D, gl_img);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, 0);
        cudaCheckError(cudaGraphicsGLRegisterImage(&cuda_resource, gl_img, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // TODO: CUDA texture/surface objects
    cudaArray_t map_cuda(uint32_t index = 0, uint32_t miplevel = 0) {
        cudaArray_t array;
        cudaCheckError(cudaGraphicsMapResources(1, &cuda_resource, 0));
        cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, index, miplevel));
        return array;
    }

    void unmap_cuda() {
        cudaCheckError(cudaGraphicsUnmapResources(1, &cuda_resource, 0));
    }

    // data
    uint2 size;
    GLuint gl_img;
    struct cudaGraphicsResource* cuda_resource;
};