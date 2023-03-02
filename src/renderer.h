#pragma once

#include <cppgl.h>
#include <voldata.h>

#include "environment.h"
#include "transferfunc.h"

struct BrickGridGL {
    cppgl::Texture3D indirection;
    cppgl::Texture3D range;
    cppgl::Texture3D atlas;
    glm::mat4 transform;
};

struct RendererOpenGL {
    // Renderer interface
    void init();
    void resize(uint32_t w, uint32_t h);
    void commit();
    void trace();
    void draw();
    void reset();

    // helper to convert brick grid to OpenGL 3D textures
    BrickGridGL brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& grid);
    // scale and move volume to fit into [-0.5, 0.5] unit cube
    void scale_and_move_to_unit_cube();

    // General settings
    int sample = 0;
    int sppx = 1024;
    int seed = 42;
    int bounces = 100;
    float tonemap_exposure = 5.f;
    float tonemap_gamma = 2.2f;
    bool tonemapping = true;
    bool show_environment = true;

    // Volume settings
    glm::vec3 albedo = glm::vec3(0.9);  // volume albedo
    float phase = 0.f;                  // volume phase (henyey-greenstein g parameter)
    float density_scale = 1.f;          // volume density scaling factor
    float emission_scale = 100.f;       // volume emission scaling factor

    // OpenGL data
    cppgl::Shader trace_shader, trace_shader_tf, tonemap_shader;
    cppgl::Texture2D color;
    std::vector<BrickGridGL> density_grids;
    std::vector<BrickGridGL> emission_grids;
    float majorant_emission = 0.f;

    // Volume data
    std::shared_ptr<voldata::Volume> volume;

    // Volume clip planes
    glm::vec3 vol_clip_min = glm::vec3(0.f);
    glm::vec3 vol_clip_max = glm::vec3(1.f);

    // Scene data
    std::shared_ptr<Environment> environment;
    std::shared_ptr<TransferFunction> transferfunc;
};
