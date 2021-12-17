#pragma once
#include "../renderer.h"

#include <cuda.h>
#include "gl_interop.h"
#include "ptx_cache.h"

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
    BufferCUDAGL<float4> fbo;

    // CUDA data
    PtxCache ptx_cache;
};
