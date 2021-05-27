#include <cppgl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>

#include <voldata/voldata.h>

#include "environment.h"
#include "transferfunc.h"

// ------------------------------------------
// state / variables / settings

static int sample = 0;
static int sppx = 1000;
static int bounces = 3;
static bool tonemapping = true;
static float tonemap_exposure = 10.f;
static float tonemap_gamma = 2.2f;
static bool show_convergence = false;
static bool show_environment = true;
static std::shared_ptr<voldata::Volume> volume;
static Texture3D vol_dense;
static Texture3D vol_indirection, vol_range, vol_atlas;
static glm::vec3 vol_crop_min = glm::vec3(0), vol_crop_max = glm::vec3(1);
static TransferFunction transferfunc;
static Environment environment;
static Shader trace_shader;
static Framebuffer fbo;
static float shader_check_delay_ms = 1000;

// ------------------------------------------
// helper funcs and callbacks

void blit(const Texture2D& tex) {
    static Shader blit_shader = Shader("blit", "shader/quad.vs", "shader/blit.fs");
    blit_shader->bind();
    blit_shader->uniform("tex", tex, 0);
    Quad::draw();
    blit_shader->unbind();
}

void tonemap(const Texture2D& tex) {
    static Shader tonemap_shader = Shader("tonemap", "shader/quad.vs", "shader/tonemap.fs");
    tonemap_shader->bind();
    tonemap_shader->uniform("tex", tex, 0);
    tonemap_shader->uniform("exposure", tonemap_exposure);
    tonemap_shader->uniform("gamma", tonemap_gamma);
    Quad::draw();
    tonemap_shader->unbind();
}

void convergence(const Texture2D& color, const Texture2D& even) {
    static Shader tonemap_shader = Shader("convergence", "shader/quad.vs", "shader/convergence.fs");
    tonemap_shader->bind();
    tonemap_shader->uniform("color", color, 0);
    tonemap_shader->uniform("even", even, 1);
    Quad::draw();
    tonemap_shader->unbind();
}

void resize_callback(int w, int h) {
    fbo->resize(w, h);
    sample = 0; // restart rendering
}

void keyboard_callback(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (/*mods == GLFW_MOD_SHIFT && */key == GLFW_KEY_B && action == GLFW_PRESS) {
        show_environment = !show_environment;
        sample = 0;
    }
    if (/*mods == GLFW_MOD_SHIFT && */key == GLFW_KEY_C && action == GLFW_PRESS)
        show_convergence = !show_convergence;
    if (/*mods == GLFW_MOD_SHIFT && */key == GLFW_KEY_T && action == GLFW_PRESS)
        tonemapping = !tonemapping;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        sample = 0;
    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        Context::screenshot("screenshot.png");
}

void mouse_button_callback(int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
}

void mouse_callback(double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    static double old_xpos = -1, old_ypos = -1;
    if (old_xpos == -1 || old_ypos == -1) {
        old_xpos = xpos;
        old_ypos = ypos;
    }
    if (Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_RIGHT)) {
        const auto [min, maj] = volume->current_grid()->minorant_majorant();
        if (Context::key_pressed(GLFW_KEY_LEFT_SHIFT))
            transferfunc->window_width += (xpos - old_xpos) * (maj - min) * 0.001;
        else
            transferfunc->window_left += (xpos - old_xpos) * (maj - min) * 0.001;
        sample = 0;
    }
    old_xpos = xpos;
    old_ypos = ypos;
}

void gui_callback(void) {
    const glm::ivec2 size = Context::resolution();
    ImGui::SetNextWindowPos(ImVec2(size.x-260, 20));
    ImGui::SetNextWindowSize(ImVec2(250, -1));
    if (ImGui::Begin("Stuff", 0, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoFocusOnAppearing)) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, .9f);
        static float est_ravg = 0.f;
        est_ravg = glm::mix(est_ravg, float(Context::frame_time() * (sppx - sample) / 1000.f), 0.1f);
        ImGui::Text("Sample: %i/%i (est: %um, %us)", sample, sppx, uint32_t(est_ravg) / 60, uint32_t(est_ravg) % 60);
        if (ImGui::InputInt("Sppx", &sppx)) sample = 0;
        if (ImGui::InputInt("Bounces", &bounces)) sample = 0;
        ImGui::Separator();
        if (ImGui::Checkbox("Environment", &show_environment)) sample = 0;
        if (ImGui::DragFloat("Env strength", &environment->strength, 0.1f)) {
            sample = 0;
            environment->strength = fmaxf(0.f, environment->strength);
        }
        ImGui::Checkbox("Tonemapping", &tonemapping);
        if (ImGui::DragFloat("Exposure", &tonemap_exposure, 0.01f))
            tonemap_exposure = fmaxf(0.f, tonemap_exposure);
        ImGui::DragFloat("Gamma", &tonemap_gamma, 0.01f);
        ImGui::Checkbox("Show convergence", &show_convergence);
        ImGui::Separator();
        if (ImGui::DragFloat3("Albedo", &volume->get_albedo().x, 0.01f, 0.f, 1.f)) sample = 0;
        if (ImGui::DragFloat("Density scale", &volume->get_density_scale(), 0.01f, 0.01f, 1000.f)) sample = 0;
        if (ImGui::SliderFloat("Phase g", &volume->get_phase(), -.95f, .95f)) sample = 0;
        ImGui::Separator();
        if (ImGui::DragFloat("Window left", &transferfunc->window_left, 0.01f)) sample = 0;
        if (ImGui::DragFloat("Window width", &transferfunc->window_width, 0.01f)) sample = 0;
        if (ImGui::Button("Neutral TF")) {
            transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(1) });
            transferfunc->upload_gpu();
            sample = 0;
        }
        if (ImGui::Button("Gradient TF")) {
            transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1) });
            transferfunc->upload_gpu();
            sample = 0;
        }
        if (ImGui::Button("Triangle TF")) {
            transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1), glm::vec4(0) });
            transferfunc->upload_gpu();
            sample = 0;
        }
        if (ImGui::Button("White background")) {
            glm::vec3 color(1);
            environment = Environment("white_background", Texture2D("white_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
            sample = 0;
        }
        ImGui::Separator();
        if (ImGui::SliderFloat3("Vol crop min", &vol_crop_min.x, 0.f, 1.f)) sample = 0;
        if (ImGui::SliderFloat3("Vol crop max", &vol_crop_max.x, 0.f, 1.f)) sample = 0;
        ImGui::Separator();
        ImGui::Text("Modelmatrix:");
        glm::mat4 row_maj = glm::transpose(volume->model);
        bool modified = false;
        if (ImGui::InputFloat4("row0", &row_maj[0][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row1", &row_maj[1][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row2", &row_maj[2][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row3", &row_maj[3][0], "%.2f")) modified = true;
        if (modified) {
            volume->model = glm::transpose(row_maj);
            sample = 0;
        }
        ImGui::Separator();
        ImGui::Text("Rotate VOLUME");
        if (ImGui::Button("90° X##V")) {
            volume->model = glm::rotate(volume->model, 1.5f * float(M_PI), glm::vec3(1, 0, 0));
            sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##V")) {
            volume->model = glm::rotate(volume->model, 1.5f * float(M_PI), glm::vec3(0, 1, 0));
            sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##V")) {
            volume->model = glm::rotate(volume->model, 1.5f * float(M_PI), glm::vec3(0, 0, 1));
            sample = 0;
        }
        ImGui::Separator();
        ImGui::Text("Rotate ENVMAP");
        if (ImGui::Button("90° X##E")) {
            environment->model = glm::mat3(glm::rotate(glm::mat4(environment->model), 1.5f * float(M_PI), glm::vec3(1, 0, 0)));
            sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##E")) {
            environment->model = glm::mat3(glm::rotate(glm::mat4(environment->model), 1.5f * float(M_PI), glm::vec3(0, 1, 0)));
            sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##E")) {
            environment->model = glm::mat3(glm::rotate(glm::mat4(environment->model), 1.5f * float(M_PI), glm::vec3(0, 0, 1)));
            sample = 0;
        }
        ImGui::PopStyleVar();
        ImGui::End();
    }
}

// ------------------------------------------
// main

int main(int argc, char** argv) {
    // init GL
    ContextParameters params;
    params.width = 1920;
    params.height = 1080;
    params.title = "VolRen";
    params.floating = GLFW_TRUE;
    params.resizable = GLFW_FALSE;
    params.swap_interval = 0;
    try  {
        Context::init(params);
    } catch (std::runtime_error& e) {
        std::cerr << "Failed to create context: " << e.what() << std::endl;
        std::cerr << "Retrying for offline rendering..." << std::endl;
        params.visible = GLFW_FALSE;
        Context::init(params);
    }
    Context::set_resize_callback(resize_callback);
    Context::set_keyboard_callback(keyboard_callback);
    Context::set_mouse_button_callback(mouse_button_callback);
    Context::set_mouse_callback(mouse_callback);
    gui_add_callback("vol_gui", gui_callback);

    // parse cmd line args
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-w")
            Context::resize(std::stoi(argv[++i]), Context::resolution().y);
        else if (arg == "-h")
            Context::resize(Context::resolution().x, std::stoi(argv[++i]));
        else if (arg == "-sppx")
            sppx = std::stoi(argv[++i]);
        else if (arg == "-d")
            bounces = std::stoi(argv[++i]);
        else if (arg == "-env")
            environment = Environment("environment", Texture2D("environment", argv[++i]));
        else if (arg == "-lut")
            transferfunc = TransferFunction("tf", argv[++i]);
        else if (arg == "-pos") {
            current_camera()->pos.x = std::stof(argv[++i]);
            current_camera()->pos.y = std::stof(argv[++i]);
            current_camera()->pos.z = std::stof(argv[++i]);
        } else if (arg == "-dir") {
            current_camera()->dir.x = std::stof(argv[++i]);
            current_camera()->dir.y = std::stof(argv[++i]);
            current_camera()->dir.z = std::stof(argv[++i]);
        } else if (arg == "-fov")
            current_camera()->fov_degree = std::stof(argv[++i]);
        else if (arg == "-exp")
            tonemap_exposure = std::stof(argv[++i]);
        else if (arg == "-gamma")
            tonemap_gamma = std::stof(argv[++i]);
        else
            volume = std::make_shared<voldata::Volume>(argv[i]);
    }

    // setup fbo
    const glm::ivec2 res = Context::resolution();
    fbo = Framebuffer("fbo", res.x, res.y);
    fbo->attach_depthbuffer();
    fbo->attach_colorbuffer(Texture2D("fbo/col", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(Texture2D("fbo/even", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->check();

    // setup trace shader
    trace_shader = Shader("trace", "shader/pathtracer.glsl");

    // load default envmap?
    if (!environment) {
        glm::vec3 color(1);
        environment = Environment("white_background", Texture2D("white_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
    }

    // load default volume?
    if (!volume)
        volume = std::make_shared<voldata::Volume>("data/head_8bit.dat");
    std::cout << "Volume: " << std::endl << volume->to_string() << std::endl;

    // load dense volume texture
    std::shared_ptr<voldata::DenseGrid> dense = std::dynamic_pointer_cast<voldata::DenseGrid>(volume->current_grid()); // check type
    if (!dense) dense = std::make_shared<voldata::DenseGrid>(volume->current_grid()); // type not matching, convert grid
    std::cout << "dense grid:" << std::endl << dense->to_string() << std::endl;
    vol_dense = Texture3D("vol dense",
            dense->n_voxels.x,
            dense->n_voxels.y,
            dense->n_voxels.z,
            GL_RED,
            GL_RED,
            GL_UNSIGNED_BYTE,
            dense->voxel_data.data());
    ///*
    // load brick volume textures
    std::shared_ptr<voldata::BrickGrid> bricks = std::dynamic_pointer_cast<voldata::BrickGrid>(volume->current_grid()); // check type
    if (!bricks) bricks = std::make_shared<voldata::BrickGrid>(volume->current_grid()); // type not matching, convert grid
    std::cout << "brick grid:" << std::endl << bricks->to_string() << std::endl;
    // indirection texture
    vol_indirection = Texture3D("brick indirection",
            bricks->indirection.stride.x,
            bricks->indirection.stride.y,
            bricks->indirection.stride.z,
            GL_RGBA8UI,
            GL_RGBA_INTEGER,
            GL_UNSIGNED_BYTE,
            // TODO 10/10/10/2 layout
            //GL_RGB10_A2,
            //GL_RGBA,
            //GL_UNSIGNED_INT_2_10_10_10_REV,
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
    for (uint32_t i = 0; i < bricks->range_mipmaps.size(); ++i) {
        glTexImage3D(GL_TEXTURE_3D,
                i + 1,
                vol_range->internal_format,
                bricks->range_mipmaps[i].stride.x,
                bricks->range_mipmaps[i].stride.y,
                bricks->range_mipmaps[i].stride.z,
                0,
                vol_range->format,
                vol_range->type,
                bricks->range_mipmaps[i].data.data());
    }
    vol_range->unbind();
    // atlas texture
    vol_atlas = Texture3D("brick atlas",
            bricks->atlas.stride.x,
            bricks->atlas.stride.y,
            bricks->atlas.stride.z,
            GL_RED,
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
    //*/

    // load default transfer function
    if (!transferfunc)
        transferfunc = TransferFunction("tf", std::vector<glm::vec4>({ glm::vec4(1) }));

    // default setup
    {
        const auto [bb_min, bb_max] = volume->AABB();
        const auto [min, maj] = volume->current_grid()->minorant_majorant();
        current_camera()->pos = bb_min + (bb_max - bb_min) * glm::vec3(-.5f, .5f, 0.f);
        current_camera()->dir = glm::normalize((bb_max + bb_min)*.5f - current_camera()->pos);
        environment->strength = 1.f;
        volume->set_albedo(glm::vec3(0.5f));
        volume->set_density_scale(2.f / maj);
        volume->set_phase(0.5f);
        transferfunc->window_left = min;
        transferfunc->window_width = maj - min;
    }

    // run
    float shader_timer = 0;
    while (Context::running()) {
        // handle input
        glfwPollEvents();
        if (CameraImpl::default_input_handler(Context::frame_time()))
            sample = 0; // restart rendering

        // update
        current_camera()->update();
        // reload shaders?
        shader_timer -= Context::frame_time();
        if (shader_timer <= 0) {
            if (reload_modified_shaders())
                sample = 0;
            shader_timer = shader_check_delay_ms;
        }

        // render
        if (sample < sppx) {
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
        } else
            glfwWaitEventsTimeout(1.f / 10); // 10fps idle

        // draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (show_convergence)
            convergence(fbo->color_textures[0], fbo->color_textures[1]);
        else {
            if (tonemapping)
                tonemap(fbo->color_textures[0]);
            else
                blit(fbo->color_textures[0]);
        }

        // finish frame
        Context::swap_buffers();
    }
}
