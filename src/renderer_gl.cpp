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
    trace_shader->uniform("vol_majorant", maj * volume->density_scale);
    trace_shader->uniform("vol_inv_majorant", 1.f / (maj * volume->density_scale));
    trace_shader->uniform("vol_albedo", volume->albedo);
    trace_shader->uniform("vol_phase_g", volume->phase);
    trace_shader->uniform("vol_density_scale", volume->density_scale);
    // brick grid data
    trace_shader->uniform("vol_indirection", vol_indirection, tex_unit++);
    trace_shader->uniform("vol_range", vol_range, tex_unit++);
    trace_shader->uniform("vol_atlas", vol_atlas, tex_unit++);
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
    if (!loss_shader)
        loss_shader = Shader("finite differences", "shader/loss.glsl");
}

void BackpropRendererOpenGL::resize(uint32_t w, uint32_t h) {
    RendererOpenGL::resize(w, h);
    if (color_prediction) color_prediction->resize(w, h);
    if (color_backprop) color_backprop->resize(w, h);
}

void BackpropRendererOpenGL::commit() {
    RendererOpenGL::commit();
    // init parameter, gradient and moments buffers
    n_parameters = transferfunc->lut.size();
    {
        parameter_buffer = SSBO("parameter buffer");
        std::vector<glm::vec4> param_data = TransferFunction::compute_lut_cdf(std::vector<glm::vec4>(n_parameters, glm::vec4(1)));
        parameter_buffer->upload_data(param_data.data(), param_data.size() * sizeof(glm::vec4));
    }
    {
        gradient_buffer = SSBO("gradients buffer");
        std::vector<glm::vec4> grad_data = std::vector<glm::vec4>(n_parameters, glm::vec4(0));
        gradient_buffer->upload_data(grad_data.data(), grad_data.size() * sizeof(glm::vec4));
    }
    {
        m1_buffer = SSBO("first moments buffer");
        std::vector<glm::vec4> m1_data = std::vector<glm::vec4>(n_parameters, glm::vec4(0));
        m1_buffer->upload_data(m1_data.data(), m1_data.size() * sizeof(glm::vec4));
    }
    {
        m2_buffer = SSBO("second moments buffer");
        std::vector<glm::vec4> m2_data = std::vector<glm::vec4>(n_parameters, glm::vec4(1));
        m2_buffer->upload_data(m2_data.data(), m2_data.size() * sizeof(glm::vec4));
    }
    // init loss buffer
    loss_buffer = SSBO("finite differences loss buffer", sizeof(float));
}

void BackpropRendererOpenGL::trace() {
    // trace reference sample
    RendererOpenGL::trace();
}

void BackpropRendererOpenGL::backprop() {
    // bind
    backprop_shader->bind();
    parameter_buffer->bind_base(0);
    gradient_buffer->bind_base(1);
    color_prediction->bind_image(0, GL_READ_WRITE, GL_RGBA32F);
    color->bind_image(1, GL_READ_ONLY, GL_RGBA32F);
    color_backprop->bind_image(2, GL_WRITE_ONLY, GL_RGBA32F);

    // uniforms
    uint32_t tex_unit = 0;
    backprop_shader->uniform("sppx", backprop_sppx);
    backprop_shader->uniform("bounces", bounces);
    backprop_shader->uniform("seed", seed + 42); // use different seed than forward
    backprop_shader->uniform("show_environment", show_environment ? 1 : 0);
    backprop_shader->uniform("n_parameters", n_parameters);
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
    // brick grid data
    backprop_shader->uniform("vol_indirection", vol_indirection, tex_unit++);
    backprop_shader->uniform("vol_range", vol_range, tex_unit++);
    backprop_shader->uniform("vol_atlas", vol_atlas, tex_unit++);
    // transfer function
    transferfunc->set_uniforms(backprop_shader, tex_unit, 4);
    backprop_shader->uniform("tf_optimization", 1);
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
    color_backprop->unbind_image(2);
    color->unbind_image(1);
    color_prediction->unbind_image(0);
    gradient_buffer->unbind_base(1);
    parameter_buffer->unbind_base(0);
    backprop_shader->unbind();
}

void BackpropRendererOpenGL::step() {
    adam_shader->bind();
    parameter_buffer->bind_base(0);
    gradient_buffer->bind_base(1);
    m1_buffer->bind_base(2);
    m2_buffer->bind_base(3);

    adam_shader->uniform("n_parameters", n_parameters);
    adam_shader->uniform("learning_rate", learning_rate);
    // debug: reset optimization
    adam_shader->uniform("reset", reset_optimization ? 1 : 0);
    // debug: solve optimization
    adam_shader->uniform("solve", solve_optimization ? 1 : 0);
    uint32_t tex_unit = 0;
    transferfunc->set_uniforms(adam_shader, tex_unit, 4);
    
    adam_shader->dispatch_compute(n_parameters);

    m2_buffer->unbind_base(3);
    m1_buffer->unbind_base(2);
    gradient_buffer->unbind_base(1);
    parameter_buffer->unbind_base(0);
    adam_shader->unbind();

    solve_optimization = false;
    reset_optimization = false;
}

float BackpropRendererOpenGL::compute_loss() {
    // TODO update this to new optimization target
    loss_buffer->clear();
    loss_shader->bind();
    parameter_buffer->bind_base(0);
    loss_buffer->bind_base(1);
    loss_shader->uniform("n_parameters", n_parameters);
    uint32_t tex_unit = 0;
    transferfunc->set_uniforms(loss_shader, tex_unit, 4);
    loss_shader->dispatch_compute(n_parameters);
    loss_buffer->unbind_base(1);
    parameter_buffer->bind_base(0);
    loss_shader->unbind();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    const float* data = (float*)loss_buffer->map(GL_READ_ONLY);
    const float loss = data[0];
    loss_buffer->unmap();
    return loss / float(n_parameters);
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
