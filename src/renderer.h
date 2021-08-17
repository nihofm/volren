#pragma once

#include <voldata.h>

// OpenGL includes
#include <cppgl.h>
#include "environment.h"
#include "transferfunc.h"

// CUDA includes
#include "cuda/gl.cuh"
#include "cuda/common.cuh"

struct Renderer {
    // Renderer interface
    virtual void init() = 0;                            // initialize renderer (call once upon initialization)
    virtual void resize(uint32_t w, uint32_t h) = 0;    // resize internal buffers
    virtual void commit() = 0;                          // commit and upload internal data structures (call after changing the scene)
    virtual void trace(uint32_t spp = 1) = 0;           // trace single sample per pixel (wavefront)
    virtual void draw() = 0;                            // draw result on screen

    // Camera data TODO actually use this
    glm::vec3 cam_pos = glm::vec3(0, 0, 0);
    glm::vec3 cam_dir = glm::vec3(1, 0, 0);
    glm::vec3 cam_up = glm::vec3(0, 1, 0);
    float cam_fov = 70.f;

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
    int draw_idx = 0;
    std::vector<Texture2D> textures;
    Texture3D vol_indirection, vol_range, vol_atlas;
};

struct BackpropRendererOpenGL : public RendererOpenGL {

    void init() override;
    void resize(uint32_t w, uint32_t h) override;
    void commit() override;
    void trace(uint32_t spp = 1) override;
    void draw() override;

    void trace_prediction(uint32_t spp = 1);
    void backprop();
    void apply_gradients();

    // OpenGL data
    Texture2D prediction, grad_debug;
    Shader pred_trace_shader, backprop_shader, apply_shader;

    // Settings
    bool draw_debug = false;

    // Optimization target and gradients:
    Texture3D vol_dense, vol_grad;
};

// TODO
struct RendererCUDA : public Renderer {
    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace(uint32_t spp = 1);
    void draw();

    // CUDA data
    BufferCUDAGL<float4> fbo;
    BufferCUDA<CameraCUDA> cam;
    BufferCUDA<VolumeCUDA> vol;
    // TODO environment + transferfunc
};
