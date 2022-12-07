#include "renderer_gl.h"

using namespace cppgl;

// -----------------------------------------------------------
// helper funcs

void blit(const Texture2D& tex) {
    static Shader blit_shader = Shader("blit", "shader/quad.vs", "shader/blit.fs");
    blit_shader->bind();
    blit_shader->uniform("tex", tex, 0);
    Quad::draw();
    blit_shader->unbind();
}

void tonemap(const Texture2D& tex, float exposure, float gamma) {
    static Shader tonemap_shader = Shader("tonemap", "shader/quad.vs", "shader/tonemap.fs");
    tonemap_shader->bind();
    tonemap_shader->uniform("tex", tex, 0);
    tonemap_shader->uniform("exposure", exposure);
    tonemap_shader->uniform("gamma", gamma);
    Quad::draw();
    tonemap_shader->unbind();
}

// -----------------------------------------------------------
// OpenGL renderer

BrickGridGL RendererOpenGL::brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& bricks) {
    // create indirection texture
    Texture3D indirection = Texture3D("brick indirection",
            bricks->indirection.stride.x,
            bricks->indirection.stride.y,
            bricks->indirection.stride.z,
            GL_RGB10_A2UI,
            GL_RGBA_INTEGER,
            GL_UNSIGNED_INT_10_10_10_2,
            bricks->indirection.data.data());
    indirection->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    indirection->unbind();
    // create range texture
    Texture3D range = Texture3D("brick range",
            bricks->range.stride.x,
            bricks->range.stride.y,
            bricks->range.stride.z,
            GL_RG16F,
            GL_RG,
            GL_HALF_FLOAT,
            bricks->range.data.data());
    range->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // create min/max mipmaps
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, bricks->range_mipmaps.size());
    for (uint32_t i = 0; i < bricks->range_mipmaps.size(); ++i) {
        glTexImage3D(GL_TEXTURE_3D,
                i + 1,
                GL_RG16F,
                bricks->range_mipmaps[i].stride.x,
                bricks->range_mipmaps[i].stride.y,
                bricks->range_mipmaps[i].stride.z,
                0,
                GL_RG,
                GL_HALF_FLOAT,
                bricks->range_mipmaps[i].data.data());
    }
    range->unbind();
    // create atlas texture
    Texture3D atlas = Texture3D("brick atlas",
            bricks->atlas.stride.x,
            bricks->atlas.stride.y,
            bricks->atlas.stride.z,
            GL_COMPRESSED_RED,
            GL_RED,
            GL_UNSIGNED_BYTE,
            bricks->atlas.data.data());
    atlas->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    atlas->unbind();
    // return BrickGridGL
    return BrickGridGL{ indirection, range, atlas, bricks->transform };
}

void RendererOpenGL::init() {
    // load default volume
    if (!volume)
        volume = std::make_shared<voldata::Volume>();

    // load default environment map
    if (!environment) {
        glm::vec3 color(1.f);
        environment = std::make_shared<Environment>(Texture2D("background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
    }

    // load default transfer function
    if (!transferfunc)
        transferfunc = std::make_shared<TransferFunction>(std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1) }));

    // compile trace shader
    if (!trace_shader)
        trace_shader = Shader("trace brick", "shader/pathtracer_brick.glsl");

    // setup color texture
    if (!color) {
        const glm::ivec2 res = Context::resolution();
        color = Texture2D("color", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    }
}

void RendererOpenGL::resize(uint32_t w, uint32_t h) {
    if (color) color->resize(w, h);
}

void RendererOpenGL::commit() {
    density_grids.clear();
    emission_grids.clear();
    std::cout << "Preparing brick grids for OpenGL..." << std::endl;
    for (const auto& frame : volume->grids) {
        density_grids.push_back(brick_grid_to_textures(voldata::Volume::to_brick_grid(frame.at("density"))));
        if (frame.find("flame") != frame.end())
            emission_grids.push_back(brick_grid_to_textures(voldata::Volume::to_brick_grid(frame.at("flame"))));
        else if (frame.find("temperature") != frame.end())
            emission_grids.push_back(brick_grid_to_textures(voldata::Volume::to_brick_grid(frame.at("temperature"))));
    }
    // create irradiance cache texture in same resolution as largest indirection grid
    glm::uvec3 n_probes = glm::uvec3(0);
    for (const BrickGridGL& grid : density_grids) {
        n_probes = glm::max(n_probes, glm::uvec3(grid.indirection->w, grid.indirection->h, grid.indirection->d));
    }
    irradiance_cache = SSBO("irradiance cache", sizeof(glm::vec4) * n_probes.x * n_probes.y * n_probes.z);
    irradiance_cache->clear();
}

void RendererOpenGL::trace() {
    // bind
    trace_shader->bind();
    color->bind_image(0, GL_READ_WRITE, GL_RGBA32F);
    irradiance_cache->bind_base(5);

    // uniforms
    uint32_t tex_unit = 0;
    trace_shader->uniform("bounces", bounces);
    trace_shader->uniform("seed", seed);
    trace_shader->uniform("show_environment", show_environment ? 1 : 0);
    trace_shader->uniform("optimization", 0);
    // camera
    trace_shader->uniform("cam_pos", current_camera()->pos);
    trace_shader->uniform("cam_fov", current_camera()->fov_degree);
    trace_shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));
    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->minorant_majorant();
    trace_shader->uniform("vol_bb_min", bb_min + vol_clip_min * (bb_max - bb_min));
    trace_shader->uniform("vol_bb_max", bb_min + vol_clip_max * (bb_max - bb_min));
    trace_shader->uniform("vol_minorant", min * volume->density_scale);
    trace_shader->uniform("vol_majorant", maj * volume->density_scale);
    trace_shader->uniform("vol_inv_majorant", 1.f / (maj * volume->density_scale));
    trace_shader->uniform("vol_albedo", volume->albedo);
    trace_shader->uniform("vol_phase_g", volume->phase);
    trace_shader->uniform("vol_density_scale", volume->density_scale);
    trace_shader->uniform("vol_emission_scale", volume->emission_scale);
    // density brick grid data
    const BrickGridGL density = density_grids[volume->grid_frame_counter];
    trace_shader->uniform("vol_density_transform", density.transform);
    trace_shader->uniform("vol_density_inv_transform", glm::inverse(density.transform));
    trace_shader->uniform("vol_density_indirection", density.indirection, tex_unit++);
    trace_shader->uniform("vol_density_range", density.range, tex_unit++);
    trace_shader->uniform("vol_density_atlas", density.atlas, tex_unit++);
    // emission brick grid data
    if (volume->grid_frame_counter < emission_grids.size()) {
        const BrickGridGL emission = emission_grids[volume->grid_frame_counter];
        trace_shader->uniform("vol_emission_transform", emission.transform);
        trace_shader->uniform("vol_emission_inv_transform", glm::inverse(emission.transform));
        trace_shader->uniform("vol_emission_indirection", emission.indirection, tex_unit++);
        trace_shader->uniform("vol_emission_range", emission.range, tex_unit++);
        trace_shader->uniform("vol_emission_atlas", emission.atlas, tex_unit++);
    }
    // irradiance cache
    trace_shader->uniform("irradiance_size", glm::uvec3(density.indirection->w, density.indirection->h, density.indirection->d));
    // transfer function
    transferfunc->set_uniforms(trace_shader, tex_unit, 4);
    // environment
    trace_shader->uniform("env_transform", environment->transform);
    trace_shader->uniform("env_inv_transform", glm::inverse(environment->transform));
    trace_shader->uniform("env_strength", environment->strength);
    trace_shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    trace_shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    trace_shader->uniform("env_envmap", environment->envmap, tex_unit++);
    trace_shader->uniform("env_impmap", environment->impmap, tex_unit++);

    // trace
    const glm::ivec2 resolution = Context::resolution();
    trace_shader->uniform("current_sample", sample);
    trace_shader->uniform("resolution", resolution);
    trace_shader->dispatch_compute(resolution.x, resolution.y);

    // unbind
    irradiance_cache->unbind_base(5);
    color->unbind_image(0);
    trace_shader->unbind();
}

void RendererOpenGL::draw() {
    if (!color) return;
    if (tonemapping)
        tonemap(color, tonemap_exposure, tonemap_gamma);
    else
        blit(color);
}
