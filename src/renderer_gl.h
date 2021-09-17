#pragma once

#include "renderer.h"

// OpenGL includes
#include "cppgl.h"
#include "environment.h"
#include "transferfunc.h"

struct RendererOpenGL : public Renderer {
    static void initOpenGL(uint32_t w = 1920, uint32_t h = 1080, bool vsync = false, bool pinned = false, bool visible = true);

    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace();
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
    void trace() override;
    void draw() override;
    void draw_adjoint();

    void radiative_backprop();
    void apply_gradients();

    // OpenGL data
    Texture2D prediction, last_sample, radiative_debug;
    Shader backprop_shader, gradient_apply_shader, draw_shader;

    // Optimization target and gradients:
    Texture3D vol_dense, vol_grad, adam_params;
    float learning_rate = 0.01f;
    int backprop_sample = 0;
    int backprop_sppx = 8;
};