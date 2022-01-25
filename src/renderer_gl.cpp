#include "renderer_gl.h"

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

std::tuple<Texture3D, Texture3D, Texture3D> RendererOpenGL::brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& bricks) {
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
    // return as tuple
    return { indirection, range, atlas };
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
    // TODO XXX: properly add emission to brick grid (ensure matching transform) and use for irradiance cache?
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
    for (const auto& [indirection, range, atlas] : density_grids) {
        n_probes = glm::max(n_probes, glm::uvec3(indirection->w, indirection->h, indirection->d));        
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
    trace_shader->uniform("vol_model", volume->get_transform());
    trace_shader->uniform("vol_inv_model", glm::inverse(volume->get_transform()));
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
    const auto [density_indirection, density_range, density_atlas] = density_grids[volume->grid_frame_counter];
    trace_shader->uniform("vol_density_indirection", density_indirection, tex_unit++);
    trace_shader->uniform("vol_density_range", density_range, tex_unit++);
    trace_shader->uniform("vol_density_atlas", density_atlas, tex_unit++);
    // emission brick grid data TODO: finalize layout
    if (volume->grid_frame_counter < emission_grids.size()) {
        const auto [emission_indirection, emission_range, emission_atlas] = emission_grids[volume->grid_frame_counter];
        trace_shader->uniform("vol_emission_indirection", emission_indirection, tex_unit++);
        trace_shader->uniform("vol_emission_range", emission_range, tex_unit++);
        trace_shader->uniform("vol_emission_atlas", emission_atlas, tex_unit++);
    }
    // irradiance cache
    trace_shader->uniform("irradiance_size", glm::uvec3(density_indirection->w, density_indirection->h, density_indirection->d));
    // transfer function
    transferfunc->set_uniforms(trace_shader, tex_unit, 4);
    // environment
    trace_shader->uniform("env_model", environment->model);
    trace_shader->uniform("env_inv_model", glm::inverse(environment->model));
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

// -----------------------------------------------------------
// OpenGL backpropagation

void BackpropRendererOpenGL::init() {
    RendererOpenGL::init();

    // setup textures
    const glm::ivec2 res = Context::resolution();
    if (!color_prediction)
        color_prediction = Texture2D("prediction color", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    if (!color_backprop)
        color_backprop = Texture2D("debug color", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);

    // compile shaders
    if (!backprop_shader)
        backprop_shader = Shader("backprop", "shader/pathtracer_backprop.glsl");
    if (!adam_shader)
        adam_shader = Shader("adam optimizer", "shader/step_adam.glsl");
    if (!draw_shader)
        draw_shader = Shader("draw adjoint", "shader/quad.vs", "shader/adjoint.fs");
}

void BackpropRendererOpenGL::resize(uint32_t w, uint32_t h) {
    RendererOpenGL::resize(w, h);
    if (color_prediction) color_prediction->resize(w, h);
    if (color_backprop) color_backprop->resize(w, h);
}

void BackpropRendererOpenGL::commit() {
    RendererOpenGL::commit();
    // init parameter, gradient and moments buffers (optimize density)
    grid_size = volume->current_grid()->index_extent();
    n_parameters = grid_size.x * grid_size.y * grid_size.z;
    {
        parameter_buffer = SSBO("parameter buffer");
        auto param_data = std::vector<float>(n_parameters, 0.1f);
        parameter_buffer->upload_data(param_data.data(), param_data.size() * sizeof(float));
    }
    {
        gradient_buffer = SSBO("gradients buffer");
        auto grad_data = std::vector<float>(n_parameters, 0.f);
        gradient_buffer->upload_data(grad_data.data(), grad_data.size() * sizeof(float));
    }
    {
        m1_buffer = SSBO("first moments buffer");
        auto m1_data = std::vector<float>(n_parameters, 0.f);
        m1_buffer->upload_data(m1_data.data(), m1_data.size() * sizeof(float));
    }
    {
        m2_buffer = SSBO("second moments buffer");
        auto m2_data = std::vector<float>(n_parameters, 1.f);
        m2_buffer->upload_data(m2_data.data(), m2_data.size() * sizeof(float));
    }
}

void BackpropRendererOpenGL::trace() {
    // trace reference sample
    RendererOpenGL::trace();
}

void BackpropRendererOpenGL::reset() {
    sample = 0;
    backprop_sample = 0;
    batch_sample = 0;
}

void BackpropRendererOpenGL::trace_adjoint() {
    // bind
    trace_shader->bind();
    parameter_buffer->bind_base(0);
    color_prediction->bind_image(0, GL_READ_WRITE, GL_RGBA32F);
    irradiance_cache->bind_base(5);

    // uniforms
    uint32_t tex_unit = 0;
    trace_shader->uniform("bounces", bounces);
    trace_shader->uniform("seed", seed + 42); // magic number
    trace_shader->uniform("show_environment", show_environment ? 1 : 0);
    trace_shader->uniform("grid_size", grid_size);
    trace_shader->uniform("n_parameters", n_parameters);
    trace_shader->uniform("optimization", 1);
    // camera
    trace_shader->uniform("cam_pos", current_camera()->pos);
    trace_shader->uniform("cam_fov", current_camera()->fov_degree);
    trace_shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));
    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->minorant_majorant();
    trace_shader->uniform("vol_model", volume->get_transform());
    trace_shader->uniform("vol_inv_model", glm::inverse(volume->get_transform()));
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
    const auto [density_indirection, density_range, density_atlas] = density_grids[volume->grid_frame_counter];
    trace_shader->uniform("vol_density_indirection", density_indirection, tex_unit++);
    trace_shader->uniform("vol_density_range", density_range, tex_unit++);
    trace_shader->uniform("vol_density_atlas", density_atlas, tex_unit++);
    // emission brick grid data TODO: finalize layout
    if (volume->grid_frame_counter < emission_grids.size()) {
        const auto [emission_indirection, emission_range, emission_atlas] = emission_grids[volume->grid_frame_counter];
        trace_shader->uniform("vol_emission_indirection", emission_indirection, tex_unit++);
        trace_shader->uniform("vol_emission_range", emission_range, tex_unit++);
        trace_shader->uniform("vol_emission_atlas", emission_atlas, tex_unit++);
    }
    // irradiance cache
    trace_shader->uniform("irradiance_size", glm::uvec3(density_indirection->w, density_indirection->h, density_indirection->d));
    // transfer function
    transferfunc->set_uniforms(trace_shader, tex_unit, 4);
    // environment
    trace_shader->uniform("env_model", environment->model);
    trace_shader->uniform("env_inv_model", glm::inverse(environment->model));
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
    color_prediction->unbind_image(0);
    parameter_buffer->unbind_base(0);
    trace_shader->unbind();
}

void BackpropRendererOpenGL::backprop() {
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    // bind
    backprop_shader->bind();
    parameter_buffer->bind_base(0);
    gradient_buffer->bind_base(1);
    color_prediction->bind_image(0, GL_READ_WRITE, GL_RGBA32F);
    color->bind_image(1, GL_READ_ONLY, GL_RGBA32F);
    color_backprop->bind_image(2, GL_WRITE_ONLY, GL_RGBA32F);
    irradiance_cache->bind_base(5);

    // uniforms
    uint32_t tex_unit = 0;
    backprop_shader->uniform("sppx", sppx);
    backprop_shader->uniform("bounces", bounces);
    backprop_shader->uniform("seed", seed + 42); // magic number
    backprop_shader->uniform("show_environment", show_environment ? 1 : 0);
    backprop_shader->uniform("grid_size", grid_size);
    backprop_shader->uniform("n_parameters", n_parameters);
    backprop_shader->uniform("optimization", 1);
    // camera
    backprop_shader->uniform("cam_pos", current_camera()->pos);
    backprop_shader->uniform("cam_fov", current_camera()->fov_degree);
    backprop_shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));
    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->minorant_majorant();
    backprop_shader->uniform("vol_model", volume->get_transform());
    backprop_shader->uniform("vol_inv_model", glm::inverse(volume->get_transform()));
    backprop_shader->uniform("vol_bb_min", bb_min + vol_clip_min * (bb_max - bb_min));
    backprop_shader->uniform("vol_bb_max", bb_min + vol_clip_max * (bb_max - bb_min));
    backprop_shader->uniform("vol_minorant", min * volume->density_scale);
    backprop_shader->uniform("vol_majorant", maj * volume->density_scale);
    backprop_shader->uniform("vol_inv_majorant", 1.f / (maj * volume->density_scale));
    backprop_shader->uniform("vol_albedo", volume->albedo);
    backprop_shader->uniform("vol_phase_g", volume->phase);
    backprop_shader->uniform("vol_density_scale", volume->density_scale);
    backprop_shader->uniform("vol_emission_scale", volume->emission_scale);
    // brick grid data
    const auto [density_indirection, density_range, density_atlas] = density_grids[volume->grid_frame_counter];
    backprop_shader->uniform("vol_density_indirection", density_indirection, tex_unit++);
    backprop_shader->uniform("vol_density_range", density_range, tex_unit++);
    backprop_shader->uniform("vol_density_atlas", density_atlas, tex_unit++);
    // emission brick grid data TODO: finalize layout
    if (volume->grid_frame_counter < emission_grids.size()) {
        const auto [emission_indirection, emission_range, emission_atlas] = emission_grids[volume->grid_frame_counter];
        backprop_shader->uniform("vol_emission_indirection", emission_indirection, tex_unit++);
        backprop_shader->uniform("vol_emission_range", emission_range, tex_unit++);
        backprop_shader->uniform("vol_emission_atlas", emission_atlas, tex_unit++);
    }
    // irradiance cache
    backprop_shader->uniform("irradiance_size", glm::uvec3(density_indirection->w, density_indirection->h, density_indirection->d));
    // transfer function
    transferfunc->set_uniforms(backprop_shader, tex_unit, 4);
    // environment
    backprop_shader->uniform("env_model", environment->model);
    backprop_shader->uniform("env_inv_model", glm::inverse(environment->model));
    backprop_shader->uniform("env_strength", environment->strength);
    backprop_shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    backprop_shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    backprop_shader->uniform("env_envmap", environment->envmap, tex_unit++);
    backprop_shader->uniform("env_impmap", environment->impmap, tex_unit++);

    // backprop
    const glm::ivec2 resolution = Context::resolution();
    backprop_shader->uniform("current_sample", backprop_sample);
    backprop_shader->uniform("resolution", resolution);
    backprop_shader->dispatch_compute(resolution.x, resolution.y);

    // unbind
    irradiance_cache->unbind_base(5);
    color_backprop->unbind_image(2);
    color->unbind_image(1);
    color_prediction->unbind_image(0);
    gradient_buffer->unbind_base(1);
    parameter_buffer->unbind_base(0);
    backprop_shader->unbind();
}

void BackpropRendererOpenGL::gradient_step() {
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    adam_shader->bind();
    parameter_buffer->bind_base(0);
    gradient_buffer->bind_base(1);
    m1_buffer->bind_base(2);
    m2_buffer->bind_base(3);

    adam_shader->uniform("grid_size", grid_size);
    adam_shader->uniform("n_parameters", n_parameters);
    adam_shader->uniform("learning_rate", learning_rate);
    adam_shader->uniform("gradient_normalization", 1.f / float(batch_size * sppx));
    const auto [min, maj] = volume->minorant_majorant();
    adam_shader->uniform("param_min", min);
    adam_shader->uniform("param_max", maj);
    // debug: reset optimization
    adam_shader->uniform("reset", reset_optimization ? 1 : 0);
    // debug: solve optimization
    adam_shader->uniform("solve", solve_optimization ? 1 : 0);
    // brick grid data
    uint32_t tex_unit = 0;
    const auto [density_indirection, density_range, density_atlas] = density_grids[volume->grid_frame_counter];
    adam_shader->uniform("vol_density_indirection", density_indirection, tex_unit++);
    adam_shader->uniform("vol_density_range", density_range, tex_unit++);
    adam_shader->uniform("vol_density_atlas", density_atlas, tex_unit++);
    //transferfunc->set_uniforms(adam_shader, tex_unit, 4);
    
    adam_shader->dispatch_compute(n_parameters);

    m2_buffer->unbind_base(3);
    m1_buffer->unbind_base(2);
    gradient_buffer->unbind_base(1);
    parameter_buffer->unbind_base(0);
    adam_shader->unbind();

    solve_optimization = false;
    reset_optimization = false;
}

void BackpropRendererOpenGL::draw_adjoint() {
    draw_shader->bind();
    parameter_buffer->bind_base(0);
    gradient_buffer->bind_base(1);
    draw_shader->uniform("n_parameters", n_parameters);
    // textures
    uint32_t tex_unit = 0;
    draw_shader->uniform("color_prediction", color_prediction, tex_unit++);
    draw_shader->uniform("color_reference", color, tex_unit++);
    draw_shader->uniform("color_backprop", color_backprop, tex_unit++);
    // transferfunc
    transferfunc->set_uniforms(draw_shader, tex_unit, 4);

    Quad::draw();

    gradient_buffer->unbind_base(1);
    parameter_buffer->unbind_base(0);
    draw_shader->unbind();
}

void BackpropRendererOpenGL::draw() {
    RendererOpenGL::draw();
}
