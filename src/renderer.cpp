#include "renderer.h"

// Settings
int Renderer::sample = 0;
int Renderer::sppx = 1000;
int Renderer::bounces = 100;
float Renderer::tonemap_exposure = 10.f;
float Renderer::tonemap_gamma = 2.2f;
bool Renderer::tonemapping = true;
bool Renderer::show_convergence = false;
bool Renderer::show_environment = true;

// Scene data
Environment Renderer::environment;
TransferFunction Renderer::transferfunc;
std::shared_ptr<voldata::Volume> Renderer::volume;
glm::vec3 Renderer::vol_crop_min = glm::vec3(0);
glm::vec3 Renderer::vol_crop_max = glm::vec3(1);

// OpenGL data
Framebuffer Renderer::fbo;
Shader Renderer::trace_shader;
Texture3D Renderer::vol_dense;
Texture3D Renderer::vol_indirection, Renderer::vol_range, Renderer::vol_atlas;

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

void convergence(const Texture2D& color, const Texture2D& even) {
    static Shader conv_shader = Shader("convergence", "shader/quad.vs", "shader/convergence.fs");
    conv_shader->bind();
    conv_shader->uniform("color", color, 0);
    conv_shader->uniform("even", even, 1);
    Quad::draw();
    conv_shader->unbind();
}

// -----------------------------------------------------------
// init

void Renderer::init(uint32_t w, uint32_t h, bool vsync, bool pinned, bool visible) {
    // init GL
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

    // setup fbo
    const glm::ivec2 res = Context::resolution();
    fbo = Framebuffer("fbo", res.x, res.y);
    fbo->attach_depthbuffer();
    fbo->attach_colorbuffer(Texture2D("fbo/col", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(Texture2D("fbo/even", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->check();

    // compile trace shader
    trace_shader = Shader("trace", "shader/pathtracer.glsl");

    // load default envmap
    glm::vec3 color(1);
    environment = Environment("white_background", Texture2D("white_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));

    // load default transfer function
    transferfunc = TransferFunction("tf", std::vector<glm::vec4>({ glm::vec4(1) }));

    // load default volume 
    // TODO interface for openvdb generated shapes?
    volume = std::make_shared<voldata::Volume>();
}


// -----------------------------------------------------------
// upload volume grid data to OpenGL textures

void Renderer::commit() {
    std::cout << "Volume: " << std::endl << volume->to_string() << std::endl;
    // load dense volume texture
    std::shared_ptr<voldata::DenseGrid> dense = std::dynamic_pointer_cast<voldata::DenseGrid>(volume->current_grid()); // check type
    if (!dense) dense = std::make_shared<voldata::DenseGrid>(volume->current_grid()); // type not matching, convert grid
    //std::cout << "dense grid:" << std::endl << dense->to_string() << std::endl;
    vol_dense = Texture3D("vol dense",
            dense->n_voxels.x,
            dense->n_voxels.y,
            dense->n_voxels.z,
            GL_RED,
            GL_RED,
            GL_UNSIGNED_BYTE,
            dense->voxel_data.data());
    // load brick volume textures
    std::shared_ptr<voldata::BrickGrid> bricks = std::dynamic_pointer_cast<voldata::BrickGrid>(volume->current_grid()); // check type
    if (!bricks) bricks = std::make_shared<voldata::BrickGrid>(volume->current_grid()); // type not matching, convert grid
    //std::cout << "brick grid:" << std::endl << bricks->to_string() << std::endl;
    // indirection texture
    vol_indirection = Texture3D("brick indirection",
            bricks->indirection.stride.x,
            bricks->indirection.stride.y,
            bricks->indirection.stride.z,
            GL_RGBA8UI,
            GL_RGBA_INTEGER,
            GL_UNSIGNED_BYTE,
            // TODO 10/10/10/2 layout
            //GL_RGB10_A2UI,
            //GL_RGBA,
            //GL_BGRA,
            //GL_UNSIGNED_INT_2_10_10_10_REV,
            //GL_UNSIGNED_INT_10_10_10_2,
            bricks->indirection.data.data());
    vol_indirection->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    vol_indirection->unbind();
    // range texture
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
    // atlas texture
    vol_atlas = Texture3D("brick atlas",
            bricks->atlas.stride.x,
            bricks->atlas.stride.y,
            bricks->atlas.stride.z,
            //GL_RED,
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

// ------------------------------------------
// trace one sample per pixel

void Renderer::trace() {
    // bind
    uint32_t tex_unit = 0;
    trace_shader->bind();
    for (auto tex : fbo->color_textures)
        tex->bind_image(tex_unit++, GL_READ_WRITE, GL_RGBA32F);

    // uniforms
    trace_shader->uniform("current_sample", ++sample);
    trace_shader->uniform("bounces", bounces);
    trace_shader->uniform("show_environment", show_environment ? 0 : 1);
    // camera
    trace_shader->uniform("cam_pos", current_camera()->pos);
    trace_shader->uniform("cam_fov", current_camera()->fov_degree);
    trace_shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));
    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->current_grid()->minorant_majorant();
    trace_shader->uniform("vol_model", volume->get_transform());
    trace_shader->uniform("vol_inv_model", glm::inverse(volume->get_transform()));
    trace_shader->uniform("vol_bb_min", bb_min + vol_crop_min * (bb_max - bb_min));
    trace_shader->uniform("vol_bb_max", bb_min + vol_crop_max * (bb_max - bb_min));
    trace_shader->uniform("vol_majorant", maj * volume->get_density_scale());
    trace_shader->uniform("vol_inv_majorant", 1.f / (maj * volume->get_density_scale()));
    trace_shader->uniform("vol_albedo", volume->get_albedo());
    trace_shader->uniform("vol_phase_g", volume->get_phase());
    trace_shader->uniform("vol_density_scale", volume->get_density_scale());
    // dense grid data
    trace_shader->uniform("vol_min_maj", glm::vec2(min, maj));
    trace_shader->uniform("vol_dense", vol_dense, tex_unit++);
    // brick grid data
    trace_shader->uniform("vol_indirection", vol_indirection, tex_unit++);
    trace_shader->uniform("vol_range", vol_range, tex_unit++);
    trace_shader->uniform("vol_atlas", vol_atlas, tex_unit++);
    // transfer function
    transferfunc->set_uniforms(trace_shader, tex_unit);
    // environment
    environment->set_uniforms(trace_shader, tex_unit);

    // trace
    const glm::ivec2 size = Context::resolution();
    trace_shader->dispatch_compute(size.x, size.y);

    // unbind
    for (uint32_t i = 0; i < fbo->color_textures.size(); ++i)
        fbo->color_textures[i]->unbind_image(i);
    trace_shader->unbind();
}

// ------------------------------------------
// draw results on screen

void Renderer::draw() {
    if (show_convergence)
        convergence(fbo->color_textures[0], fbo->color_textures[1]);
    else {
        if (tonemapping)
            tonemap(fbo->color_textures[0], tonemap_exposure, tonemap_gamma);
        else
            blit(fbo->color_textures[0]);
    }
}
