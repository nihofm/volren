#pragma once

#include "renderer.h"

// OpenGL includes
#include "cppgl.h"
#include "environment.h"
#include "transferfunc.h"

struct RendererOpenGL : public Renderer {
    static void init_opengl(uint32_t w = 1920, uint32_t h = 1080, bool vsync = false, bool pinned = false, bool visible = true);
    std::tuple<Texture3D, Texture3D, Texture3D> brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& grid);

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
    // Texture3D density_indirection, density_range, density_atlas;
    // Texture3D vol_indirection_emission, vol_range_emission, vol_atlas_emission;
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
    void draw_adjoint();

    void backprop();
    void step();

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
    int backprop_sppx = 8;
    bool reset_optimization = false;
    bool solve_optimization = false;
};