#pragma once

#include "renderer.h"
#include "environment.h"
#include "transferfunc.h"

#include <cppgl.h>

struct RendererOpenGL : public Renderer {
    // Renderer interface
    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace();
    void draw();

    // helper to convert brick grid to OpenGL 3D textures
    std::tuple<cppgl::Texture3D, cppgl::Texture3D, cppgl::Texture3D> brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& grid);

    // Scene data
    std::shared_ptr<Environment> environment;
    std::shared_ptr<TransferFunction> transferfunc;

    // OpenGL data
    cppgl::Shader trace_shader;
    cppgl::Texture2D color;
    cppgl::SSBO irradiance_cache;
    std::vector<std::tuple<cppgl::Texture3D, cppgl::Texture3D, cppgl::Texture3D>> density_grids;
    std::vector<std::tuple<cppgl::Texture3D, cppgl::Texture3D, cppgl::Texture3D>> emission_grids;
};

struct BackpropRendererOpenGL : public RendererOpenGL {
    void init() override;
    void resize(uint32_t w, uint32_t h) override;
    void commit() override;
    void trace() override;
    void draw() override;

    void reset() override;

    void trace_adjoint();
    void backprop();
    void gradient_step();
    void draw_adjoint();

    // OpenGL data
    cppgl::Texture2D color_prediction, color_backprop;
    cppgl::Shader backprop_shader, adam_shader, draw_shader;
    cppgl::SSBO irradiance_cache_adjoint;

    // Optimization target (voxel densities)
    glm::uvec3 grid_size;           // density grid size
    uint32_t n_parameters;          // parameter count
    cppgl::SSBO parameter_buffer;   // TF lut parameters
    cppgl::SSBO gradient_buffer;    // gradients per parameter
    cppgl::SSBO m1_buffer;          // first moments per parameter
    cppgl::SSBO m2_buffer;          // second moments per parameter

    // Optimization parameters
    float learning_rate = 0.1f;
    int batch_size = 1;
    int backprop_sample = 0;
    int batch_sample = 0;
    bool reset_optimization = false;
    bool solve_optimization = false;
};
