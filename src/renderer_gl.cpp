#include "renderer.h"

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

void RendererOpenGL::initOpenGL(uint32_t w, uint32_t h, bool vsync, bool pinned, bool visible) {
    static bool is_init = false;
    if (is_init) return;

    ContextParameters params;
    params.width = w;
    params.height = h;
    params.title = "VolumeRenderer";
    params.floating = pinned ? GLFW_TRUE : GLFW_FALSE;
    params.resizable = pinned ? GLFW_FALSE : GLFW_TRUE;
    params.swap_interval = vsync ? 1 : 0;
    params.visible = visible ? GLFW_TRUE : GLFW_FALSE;
    try  {
        Context::init(params);
    } catch (std::runtime_error& e) {
        std::cerr << "Failed to create context: " << e.what() << std::endl;
        std::cerr << "Retrying for offline rendering..." << std::endl;
        params.visible = GLFW_FALSE;
        Context::init(params);
    }

    is_init = true;    
}

void RendererOpenGL::init() {
    initOpenGL();

    // load default volume
    if (!volume)
        volume = std::make_shared<voldata::Volume>();

    // load default environment map
    if (!environment) {
        glm::vec3 color(.5f);
        environment = std::make_shared<Environment>(Texture2D("background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
    }

    // load default transfer function
    if (!transferfunc)
        transferfunc = std::make_shared<TransferFunction>(std::vector<glm::vec4>({ glm::vec4(1) }));

    // compile trace shader
    if (!trace_shader)
        trace_shader = Shader("trace brick", "shader/pathtracer_brick.glsl");

    // setup textures
    if (textures.empty()) {
        const glm::ivec2 res = Context::resolution();
        textures.push_back(Texture2D("color", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
        textures.push_back(Texture2D("features1", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
        textures.push_back(Texture2D("features2", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
        textures.push_back(Texture2D("features3", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
        textures.push_back(Texture2D("features4", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    }
}

void RendererOpenGL::resize(uint32_t w, uint32_t h) {
    for (auto& tex : textures) 
        tex->resize(w, h);
}

void RendererOpenGL::commit() {
    // convert volume to brick grid
    const auto bricks = volume->current_grid_brick();
    // upload indirection texture
    vol_indirection = Texture3D("brick indirection",
            bricks->indirection.stride.x,
            bricks->indirection.stride.y,
            bricks->indirection.stride.z,
            GL_RGB10_A2UI,
            GL_RGBA_INTEGER,
            GL_UNSIGNED_INT_2_10_10_10_REV,
            bricks->indirection.data.data());
    vol_indirection->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    vol_indirection->unbind();
    // upload range texture
    vol_range = Texture3D("brick range",
            bricks->range.stride.x,
            bricks->range.stride.y,
            bricks->range.stride.z,
            GL_RG16F,
            GL_RG,
            GL_HALF_FLOAT,
            bricks->range.data.data());
    vol_range->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // upload min/max mipmaps
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
    vol_range->unbind();
    // upload atlas texture
    vol_atlas = Texture3D("brick atlas",
            bricks->atlas.stride.x,
            bricks->atlas.stride.y,
            bricks->atlas.stride.z,
            GL_COMPRESSED_RED,
            GL_RED,
            GL_UNSIGNED_BYTE,
            bricks->atlas.data.data());
    vol_atlas->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    vol_atlas->unbind();
}

void RendererOpenGL::trace(uint32_t spp) {
    // bind
    trace_shader->bind();
    for (uint32_t i = 0; i < textures.size(); ++i)
        textures[i]->bind_image(i, GL_READ_WRITE, GL_RGBA32F);

    // uniforms
    uint32_t tex_unit = 0;
    trace_shader->uniform("bounces", bounces);
    trace_shader->uniform("seed", seed);
    trace_shader->uniform("show_environment", show_environment ? 0 : 1);
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
    trace_shader->uniform("vol_inv_minorant", 1.f / (min * volume->density_scale));
    trace_shader->uniform("vol_majorant", maj * volume->density_scale);
    trace_shader->uniform("vol_inv_majorant", 1.f / (maj * volume->density_scale));
    trace_shader->uniform("vol_albedo", volume->albedo);
    trace_shader->uniform("vol_phase_g", volume->phase);
    trace_shader->uniform("vol_density_scale", volume->density_scale);
    // brick grid data
    trace_shader->uniform("vol_grid_type", 0);
    if (vol_indirection) trace_shader->uniform("vol_indirection", vol_indirection, tex_unit++);
    if (vol_range) trace_shader->uniform("vol_range", vol_range, tex_unit++);
    if (vol_atlas) trace_shader->uniform("vol_atlas", vol_atlas, tex_unit++);
    // transfer function
    trace_shader->uniform("tf_window_left", transferfunc->window_left);
    trace_shader->uniform("tf_window_width", transferfunc->window_width);
    trace_shader->uniform("tf_texture", transferfunc->texture, tex_unit++);
    // environment
    trace_shader->uniform("env_model", environment->model);
    trace_shader->uniform("env_inv_model", glm::inverse(environment->model));
    trace_shader->uniform("env_strength", environment->strength);
    trace_shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    trace_shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    trace_shader->uniform("env_envmap", environment->envmap, tex_unit++);
    trace_shader->uniform("env_impmap", environment->impmap, tex_unit++);

    // trace
    const glm::ivec2 size = Context::resolution();
    for (uint32_t i = 0; i < spp; ++i) {
        trace_shader->uniform("current_sample", ++sample);
        trace_shader->dispatch_compute(size.x, size.y);
    }

    // unbind
    for (uint32_t i = 0; i < textures.size(); ++i)
        textures[i]->unbind_image(i);
    trace_shader->unbind();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT); // TODO this required?
}

void RendererOpenGL::draw() {
    if (textures.empty()) return;
    const auto tex = textures[draw_idx % textures.size()];
    if (tonemapping)
        tonemap(tex, tonemap_exposure, tonemap_gamma);
    else
        blit(tex);
}

// -----------------------------------------------------------
// OpenGL backpropagation

void BackpropRendererOpenGL::init() {
    RendererOpenGL::init();

    // setup textures
    const glm::ivec2 res = Context::resolution();
    if (!prediction)
        prediction = Texture2D("prediction", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    if (!grad_debug)
        grad_debug = Texture2D("backprop grad", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);

    // compile shaders
    if (!pred_trace_shader)
        pred_trace_shader = Shader("trace dense", "shader/pathtracer_dense.glsl");
    if (!backprop_shader)
        backprop_shader = Shader("backprop", "shader/backprop.glsl");
    if (!apply_shader)
        apply_shader = Shader("apply gradients", "shader/apply_gradients.glsl");
}

void BackpropRendererOpenGL::resize(uint32_t w, uint32_t h) {
    RendererOpenGL::resize(w, h);
    if (prediction) prediction->resize(w, h);
    if (grad_debug) grad_debug->resize(w, h);
}

void BackpropRendererOpenGL::commit() {
    RendererOpenGL::commit();
    // initialize dense grid and gradients for optimization
    const auto n_voxels = volume->current_grid()->index_extent();
    const auto data = std::vector<float>(n_voxels.x * n_voxels.y * n_voxels.z, 0.1f);
    // upload dense texture
    vol_dense = Texture3D("dense grid data",
            n_voxels.x,
            n_voxels.y,
            n_voxels.z,
            GL_R32F,
            GL_RED,
            GL_FLOAT,
            data.data());
    vol_dense->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    vol_dense->unbind();
    // upload dense texture
    const auto grad = std::vector<float>(n_voxels.x * n_voxels.y * n_voxels.z, 0.f);
    vol_grad = Texture3D("dense grid gradients",
            n_voxels.x,
            n_voxels.y,
            n_voxels.z,
            GL_R32F,
            GL_RED,
            GL_FLOAT,
            grad.data());
    vol_grad->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    vol_grad->unbind();
}

void BackpropRendererOpenGL::trace(uint32_t spp) {
    // trace reference sample
    RendererOpenGL::trace(spp);
}

void BackpropRendererOpenGL::trace_prediction(uint32_t spp) {
    // trace prediction sample
    uint32_t tex_unit = 0;
    pred_trace_shader->bind();
    prediction->bind_image(tex_unit++, GL_READ_WRITE, GL_RGBA32F);

    // uniforms
    pred_trace_shader->uniform("bounces", bounces);
    pred_trace_shader->uniform("seed", seed);
    pred_trace_shader->uniform("show_environment", show_environment ? 0 : 1);
    // camera
    pred_trace_shader->uniform("cam_pos", current_camera()->pos);
    pred_trace_shader->uniform("cam_fov", current_camera()->fov_degree);
    pred_trace_shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));
    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->minorant_majorant();
    pred_trace_shader->uniform("vol_model", volume->get_transform());
    pred_trace_shader->uniform("vol_inv_model", glm::inverse(volume->get_transform()));
    pred_trace_shader->uniform("vol_bb_min", bb_min + vol_clip_min * (bb_max - bb_min));
    pred_trace_shader->uniform("vol_bb_max", bb_min + vol_clip_max * (bb_max - bb_min));
    pred_trace_shader->uniform("vol_minorant", min * volume->density_scale);
    pred_trace_shader->uniform("vol_inv_minorant", 1.f / (min * volume->density_scale));
    pred_trace_shader->uniform("vol_majorant", maj * volume->density_scale);
    pred_trace_shader->uniform("vol_inv_majorant", 1.f / (maj * volume->density_scale));
    pred_trace_shader->uniform("vol_albedo", volume->albedo);
    pred_trace_shader->uniform("vol_phase_g", volume->phase);
    pred_trace_shader->uniform("vol_density_scale", volume->density_scale);
    // brick grid data
    pred_trace_shader->uniform("vol_grid_type", 1);
    if (vol_dense) pred_trace_shader->uniform("vol_dense", vol_dense, tex_unit++);
    // transfer function
    pred_trace_shader->uniform("tf_window_left", transferfunc->window_left);
    pred_trace_shader->uniform("tf_window_width", transferfunc->window_width);
    pred_trace_shader->uniform("tf_texture", transferfunc->texture, tex_unit++);
    // environment
    pred_trace_shader->uniform("env_model", environment->model);
    pred_trace_shader->uniform("env_inv_model", glm::inverse(environment->model));
    pred_trace_shader->uniform("env_strength", environment->strength);
    pred_trace_shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    pred_trace_shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    pred_trace_shader->uniform("env_envmap", environment->envmap, tex_unit++);
    pred_trace_shader->uniform("env_impmap", environment->impmap, tex_unit++);

    // trace
    const glm::ivec2 res = Context::resolution();
    for (uint32_t i = 0; i < spp; ++i) {
        pred_trace_shader->uniform("current_sample", int(sample - spp + i + 1));
        pred_trace_shader->dispatch_compute(res.x, res.y);
    }

    // unbind
    prediction->unbind_image(0);
    pred_trace_shader->unbind();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT); // TODO this required?
}

void BackpropRendererOpenGL::backprop() {
    // bind
    backprop_shader->bind();
    prediction->bind_image(0, GL_READ_ONLY, GL_RGBA32F);
    textures[0]->bind_image(1, GL_READ_ONLY, GL_RGBA32F);
    vol_grad->bind_image(2, GL_READ_WRITE, GL_R32F);
    grad_debug->bind_image(3, GL_WRITE_ONLY, GL_RGBA32F);

    // uniforms
    uint32_t tex_unit = 0;
    backprop_shader->uniform("bounces", 1);
    backprop_shader->uniform("seed", seed);
    backprop_shader->uniform("show_environment", show_environment ? 0 : 1);
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
    backprop_shader->uniform("vol_inv_minorant", 1.f / (min * volume->density_scale));
    backprop_shader->uniform("vol_majorant", maj * volume->density_scale);
    backprop_shader->uniform("vol_inv_majorant", 1.f / (maj * volume->density_scale));
    backprop_shader->uniform("vol_albedo", volume->albedo);
    backprop_shader->uniform("vol_phase_g", volume->phase);
    backprop_shader->uniform("vol_density_scale", volume->density_scale);
    // brick grid data
    backprop_shader->uniform("vol_grid_type", 1);
    if (vol_dense) backprop_shader->uniform("vol_dense", vol_dense, tex_unit++);
    // transfer function
    backprop_shader->uniform("tf_window_left", transferfunc->window_left);
    backprop_shader->uniform("tf_window_width", transferfunc->window_width);
    backprop_shader->uniform("tf_texture", transferfunc->texture, tex_unit++);
    // environment
    backprop_shader->uniform("env_model", environment->model);
    backprop_shader->uniform("env_inv_model", glm::inverse(environment->model));
    backprop_shader->uniform("env_strength", environment->strength);
    backprop_shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    backprop_shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    backprop_shader->uniform("env_envmap", environment->envmap, tex_unit++);
    backprop_shader->uniform("env_impmap", environment->impmap, tex_unit++);

    // backprop
    const glm::ivec2 res = Context::resolution();
    backprop_shader->uniform("current_sample", sample);
    backprop_shader->dispatch_compute(res.x, res.y);

    // unbind
    grad_debug->unbind_image(3);
    vol_grad->unbind_image(2);
    textures[0]->unbind_image(1);
    prediction->unbind_image(0);
    backprop_shader->unbind();
}

void BackpropRendererOpenGL::apply_gradients() {
    // TODO optimization step
    apply_shader->bind();
    vol_dense->bind_image(0, GL_READ_WRITE, GL_R32F);
    vol_grad->bind_image(1, GL_READ_WRITE, GL_R32F);

    apply_shader->uniform("learning_rate", 0.01f); // TODO
    const auto [min, maj] = volume->minorant_majorant();
    apply_shader->uniform("vol_majorant", maj);
    apply_shader->uniform("size", glm::ivec3(vol_dense->w, vol_dense->h, vol_dense->d));
    apply_shader->dispatch_compute(vol_dense->w, vol_dense->h, vol_dense->d);

    vol_grad->unbind_image(1);
    vol_dense->unbind_image(0);
    apply_shader->unbind();
}

void BackpropRendererOpenGL::draw() {
    if (draw_debug) {
        if (tonemapping)
            tonemap(grad_debug, tonemap_exposure, tonemap_gamma);
        else
            blit(grad_debug);
    } else
        RendererOpenGL::draw();
}
