#pragma once

#include "voldata.h"

// OpenGL includes
#include "cppgl.h"
#include "environment.h"
#include "transferfunc.h"

// CUDA includes
#include "cuda/common.cuh"
#include "cuda/gl.cuh"
#include <optix.h>

struct Renderer {
    // Renderer interface
    virtual void init() = 0;                            // initialize renderer (call once upon initialization)
    virtual void resize(uint32_t w, uint32_t h) = 0;    // resize internal buffers
    virtual void commit() = 0;                          // commit and upload internal data structures (call after changing the scene)
    virtual void trace(uint32_t spp = 1) = 0;           // trace single sample per pixel (wavefront)
    virtual void draw() = 0;                            // draw result on screen

    // Camera data TODO actually use this
    /*
    glm::vec3 cam_pos = glm::vec3(0, 0, 0);
    glm::vec3 cam_dir = glm::vec3(1, 0, 0);
    glm::vec3 cam_up = glm::vec3(0, 1, 0);
    float cam_fov = 70.f;
    */

    // Volume data
    std::shared_ptr<voldata::Volume> volume;
    inline void set_volume(const std::shared_ptr<voldata::Volume>& vol) { volume = vol; }

    // Volume clip planes
    glm::vec3 vol_clip_min = glm::vec3(0.f);
    glm::vec3 vol_clip_max = glm::vec3(1.f);

    // Settings
    int sample = 0;
    int sppx = 1024;
    int seed = 42;
    int bounces = 3;
    float tonemap_exposure = 10.f;
    float tonemap_gamma = 2.2f;
    bool tonemapping = true;
    bool show_environment = true;
    float raymarch_step_size = 0.1f;
};


struct RendererOpenGL : public Renderer {
    static void initOpenGL(uint32_t w = 1920, uint32_t h = 1080, bool vsync = false, bool pinned = false, bool visible = true);

    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace(uint32_t spp = 1);
    void draw();

    // Scene data
    std::shared_ptr<Environment> environment;
    std::shared_ptr<TransferFunction> transferfunc;

    // OpenGL data
    Shader trace_shader;
    Texture2D color;
    Texture3D vol_indirection, vol_range, vol_atlas;
};

struct BackpropRendererOpenGL : public RendererOpenGL {

    void init() override;
    void resize(uint32_t w, uint32_t h) override;
    void commit() override;
    void trace(uint32_t spp = 1) override;
    void draw() override;
    void draw_adjoint();

    void trace_prediction(uint32_t spp = 1);
    void radiative_backprop();
    void apply_gradients();

    // OpenGL data
    Texture2D prediction, debug_color, radiative_debug;
    Shader pred_trace_shader, backprop_shader, gradient_apply_shader, debug_shader;

    // Optimization target and gradients:
    Texture3D vol_dense, vol_grad;
    float learning_rate = 0.001f;
};

// TODO CUDA
struct RendererOptix : public Renderer {
    static void initOptix();

    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace(uint32_t spp = 1);
    void draw();

    // Optix data
    OptixDeviceContext context;
    OptixModule module;

    // CUDA data
    BufferCUDAGL<float4> fbo;
    BufferCUDA<CameraCUDA> cam;
    BufferCUDA<VolumeCUDA> vol;
    // TODO environment + transferfunc
};
