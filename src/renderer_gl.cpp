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
        glm::vec3 color(1.f);
        environment = std::make_shared<Environment>(Texture2D("background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
    }

    // load default transfer function
    if (!transferfunc)
        transferfunc = std::make_shared<TransferFunction>(std::vector<glm::vec4>({ glm::vec4(1) }));

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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    vol_atlas->unbind();
}

void RendererOpenGL::trace() {
    // bind
    trace_shader->bind();
    color->bind_image(0, GL_READ_WRITE, GL_RGBA32F);

    // uniforms
    uint32_t tex_unit = 0;
    trace_shader->uniform("bounces", bounces);
    trace_shader->uniform("seed", seed);
    trace_shader->uniform("show_environment", show_environment ? 1 : 0);
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
    const glm::ivec2 resolution = Context::resolution();
    trace_shader->uniform("current_sample", sample);
    trace_shader->uniform("resolution", resolution);
    trace_shader->dispatch_compute(resolution.x, resolution.y);

    // unbind
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
    if (!prediction)
        prediction = Texture2D("prediction", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    if (!last_sample)
        last_sample = Texture2D("grad last sample", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    if (!radiative_debug)
        radiative_debug = Texture2D("grad debug2", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);

    // compile shaders
    if (!backprop_shader)
        backprop_shader = Shader("backprop", "shader/radiative_backprop.glsl");
    if (!gradient_apply_shader)
        gradient_apply_shader = Shader("apply gradients", "shader/apply_gradients.glsl");
    if (!draw_shader)
        draw_shader = Shader("draw adjoint", "shader/quad.vs", "shader/adjoint.fs");
}

void BackpropRendererOpenGL::resize(uint32_t w, uint32_t h) {
    RendererOpenGL::resize(w, h);
    if (prediction) prediction->resize(w, h);
    if (last_sample) last_sample->resize(w, h);
    if (radiative_debug) radiative_debug->resize(w, h);
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
    // upload dense texture gradients
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
    // upload adam parameters
    const auto adam_data = std::vector<float>(n_voxels.x * n_voxels.y * n_voxels.z * 2, 0.f);
    adam_params = Texture3D("dense grid gradients",
            n_voxels.x,
            n_voxels.y,
            n_voxels.z,
            GL_RG32F,
            GL_RG,
            GL_FLOAT,
            adam_data.data());
    adam_params->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    adam_params->unbind();
}

void BackpropRendererOpenGL::trace() {
    // trace reference sample
    RendererOpenGL::trace();
}

void BackpropRendererOpenGL::radiative_backprop() {
    // bind
    backprop_shader->bind();
    prediction->bind_image(0, GL_READ_WRITE, GL_RGBA32F);
    color->bind_image(1, GL_READ_ONLY, GL_RGBA32F);
    vol_grad->bind_image(2, GL_READ_WRITE, GL_R32F);
    radiative_debug->bind_image(3, GL_WRITE_ONLY, GL_RGBA32F);

    // uniforms
    uint32_t tex_unit = 0;
    backprop_shader->uniform("sppx", backprop_sppx);
    backprop_shader->uniform("bounces", bounces);
    backprop_shader->uniform("seed", seed + 42); // use different seed than forward
    backprop_shader->uniform("show_environment", show_environment ? 1 : 0);
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
    // dense grid data
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
    const glm::ivec2 resolution = Context::resolution();
    backprop_shader->uniform("current_sample", backprop_sample);
    backprop_shader->uniform("resolution", resolution);
    backprop_shader->dispatch_compute(resolution.x, resolution.y);

    // unbind
    radiative_debug->unbind_image(3);
    vol_grad->unbind_image(2);
    color->unbind_image(1);
    prediction->unbind_image(0);
    backprop_shader->unbind();
}

void BackpropRendererOpenGL::apply_gradients() {
    // TODO adam optimizer
    gradient_apply_shader->bind();
    vol_dense->bind_image(0, GL_READ_WRITE, GL_R32F);
    vol_grad->bind_image(1, GL_READ_WRITE, GL_R32F);
    adam_params->bind_image(2, GL_READ_WRITE, GL_RG32F);

    gradient_apply_shader->uniform("learning_rate", learning_rate);
    const auto [min, maj] = volume->minorant_majorant();
    gradient_apply_shader->uniform("vol_majorant", maj);
    gradient_apply_shader->uniform("size", glm::ivec3(vol_dense->w, vol_dense->h, vol_dense->d));
    gradient_apply_shader->dispatch_compute(vol_dense->w, vol_dense->h, vol_dense->d);

    adam_params->unbind_image(2);
    vol_grad->unbind_image(1);
    vol_dense->unbind_image(0);
    gradient_apply_shader->unbind();
}

void BackpropRendererOpenGL::draw_adjoint() {
    draw_shader->bind();
    draw_shader->uniform("color_prediction", prediction, 0);
    draw_shader->uniform("color_reference", color, 1);
    draw_shader->uniform("color_debug", radiative_debug, 2);
    Quad::draw();
    draw_shader->unbind();
}

void BackpropRendererOpenGL::draw() {
    RendererOpenGL::draw();
}
