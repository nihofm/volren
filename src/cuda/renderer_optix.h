#pragma once
#include "../renderer.h"

#include <optix.h>
#include "cuda/gl_interop.h"

#include "ptx/common.cuh"

// TODO CUDA
struct RendererOptix : public Renderer {
    RendererOptix();
    ~RendererOptix();

    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace();
    void draw();

    // Optix data
    OptixDeviceContext context;
    OptixModule module;
    OptixProgramGroup raygen_group;
    OptixProgramGroup miss_group;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;

    // CUDA/GL data
    BufferCUDAGL<float4> fbo;
    Params* params;
};
