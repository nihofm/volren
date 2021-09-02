#include "common.cuh"
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

    void draw(float exposure = 1.f, float gamma = 2.2f) {
        static Shader tonemap_shader = Shader("tonemap", "shader/quad.vs", "shader/tonemap_buf.fs");
        tonemap_shader->bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gl_buf);
        tonemap_shader->uniform("size", glm::ivec2(size.x, size.y));
        tonemap_shader->uniform("exposure", exposure);
        tonemap_shader->uniform("gamma", gamma);
        Quad::draw();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
        tonemap_shader->unbind();
    }

    // data
    dim3 size;
    GLuint gl_buf;
    struct cudaGraphicsResource* cuda_resource;
};