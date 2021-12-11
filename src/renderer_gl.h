#pragma once

#include "renderer.h"

// OpenGL includes
#include "cppgl.h"
#include "environment.h"
#include "transferfunc.h"

struct RendererOpenGL : public Renderer {
    // Renderer interface
    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace();
    void draw();

    // helper to convert brick grid to OpenGL 3D textures
    std::tuple<Texture3D, Texture3D, Texture3D> brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& grid);

    // Scene data
    std::shared_ptr<Environment> environment;
    std::shared_ptr<TransferFunction> transferfunc;

    // OpenGL data
    Shader trace_shader;
    Texture2D color;
    std::vector<std::tuple<Texture3D, Texture3D, Texture3D>> density_grids;
    std::vector<std::tuple<Texture3D, Texture3D, Texture3D>> emission_grids;
    SSBO irradiance_cache;
};

struct BackpropRendererOpenGL : public RendererOpenGL {
    void init() override;
    void resize(uint32_t w, uint32_t h) override;
    void commit() override;
    void trace() override;
    void draw() override;

    void trace_adjoint();
    void backprop();
    void gradient_step();
    void draw_adjoint();

    // TODO finite differences loss
    float compute_loss();

    // OpenGL data
    Texture2D color_prediction, color_backprop;
    Shader backprop_shader, adam_shader, draw_shader, loss_shader;
    SSBO loss_buffer;

    // Optimization target
    uint32_t n_parameters;      // parameter count
    SSBO parameter_buffer;      // parameters
    SSBO gradient_buffer;       // gradients per parameter
    SSBO m1_buffer;             // first moments per parameter
    SSBO m2_buffer;             // second moments per parameter

    // Optimization parameters
    float learning_rate = 0.1f;
    int backprop_sample = 0;
    bool reset_optimization = false;
    bool solve_optimization = false;
};