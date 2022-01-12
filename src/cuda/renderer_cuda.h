#pragma once
#include "../renderer.h"

#include <cuda.h>
#include "buffer.cuh"
#include "grid.cuh"

struct RendererCUDA : public Renderer {
    RendererCUDA();
    ~RendererCUDA();

    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace();
    void draw();

    // CUDA/GL data
    // ImageCUDAGL fbo;
    BufferCUDAGL<glm::vec4> fbo;

    // CUDA data
    ManagedBufferCUDA<float> grid_data;
    DenseGridCUDA grid;
};
