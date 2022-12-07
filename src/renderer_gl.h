#pragma once

#include "renderer.h"
#include "environment.h"
#include "transferfunc.h"

#include <cppgl.h>

struct BrickGridGL {
    cppgl::Texture3D indirection;
    cppgl::Texture3D range;
    cppgl::Texture3D atlas;
    glm::mat4 transform;
};

struct RendererOpenGL : public Renderer {
    // Renderer interface
    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace();
    void draw();

    // helper to convert brick grid to OpenGL 3D textures
    BrickGridGL brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& grid);

    // Scene data
    std::shared_ptr<Environment> environment;
    std::shared_ptr<TransferFunction> transferfunc;

    // OpenGL data
    cppgl::Shader trace_shader;
    cppgl::Texture2D color;
    std::vector<BrickGridGL> density_grids;
    std::vector<BrickGridGL> emission_grids;
    cppgl::SSBO irradiance_cache;
};
