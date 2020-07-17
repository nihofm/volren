#include <cppgl/cppgl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>

#include "volume.h"
#include "environment.h"
#include "transferfunc.h"

// ------------------------------------------
// state / variables / settings

static int sample = 0;
static int sppx = 1000;
static int bounces = 100;
static bool tonemapping = true;
static float tonemap_exposure = 10.f;
static float tonemap_gamma = 2.2f;
static bool auto_exposure = false; // TODO
static bool show_convergence = false;
static bool show_environment = false;
static Volume volume;
static glm::vec3 vol_bb_min = glm::vec3(0), vol_bb_max = glm::vec3(1);
static TransferFunction transferfunc;
static Environment environment;
static Shader trace_shader;
static Framebuffer fbo;
static SSBO reservoir;
static Animation animation;
static float animation_frames_per_node = 100;
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
    tonemap_shader->uniform("auto_exposure", int(auto_exposure));
    tonemap_shader->uniform("default_exposure", tonemap_exposure);
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
    reservoir->resize(w * h * sizeof(float)*6);
    sample = 0; // restart rendering
}

void keyboard_callback(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (mods == GLFW_MOD_SHIFT && key == GLFW_KEY_E && action == GLFW_PRESS)
        show_environment = !show_environment;
    if (mods == GLFW_MOD_SHIFT && key == GLFW_KEY_C && action == GLFW_PRESS)
        show_convergence = !show_convergence;
    if (mods == GLFW_MOD_SHIFT && key == GLFW_KEY_T && action == GLFW_PRESS)
        tonemapping = !tonemapping;
    if (mods == GLFW_MOD_SHIFT && key == GLFW_KEY_R && action == GLFW_PRESS)
        reload_modified_shaders();
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        sample = 0;
    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        Context::screenshot("screenshot.png");

    if (key == GLFW_KEY_O && action == GLFW_PRESS) {
        glm::vec3 pos; glm::quat rot;
        current_camera()->store(pos, rot);
        const size_t i = animation->push_node(pos, rot);
        animation->put_data("absorbtion_coefficient", i, volume->absorbtion_coefficient);
        animation->put_data("scattering_coefficient", i, volume->scattering_coefficient);
        animation->put_data("phase_g", i, volume->phase_g);
        animation->put_data("vol_bb_min", i, vol_bb_min);
        animation->put_data("vol_bb_max", i, vol_bb_max);
        animation->put_data("window_center", i, transferfunc->window_center);
        animation->put_data("window_width", i, transferfunc->window_width);
        std::cout << "curr anim length: " << i + 1 << std::endl;
    }
    if (key == GLFW_KEY_L && action == GLFW_PRESS)
        animation->clear();
    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        animation->play();
        sample = sppx;
    }
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
        transferfunc->window_center += (xpos - old_xpos) * 0.001;
        transferfunc->window_width += (old_ypos - ypos) * 0.001;
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
        if (ImGui::DragFloat("Env strength", &environment->strength, 0.1f)) sample = 0;
        ImGui::Checkbox("Tonemapping", &tonemapping);
        ImGui::Checkbox("Auto exposure", &auto_exposure);
        ImGui::DragFloat("Exposure", &tonemap_exposure, 0.01f);
        ImGui::DragFloat("Gamma", &tonemap_gamma, 0.01f);
        ImGui::Checkbox("Show convergence", &show_convergence);
        ImGui::Separator();
        if (ImGui::DragFloat("Absorb", &volume->absorbtion_coefficient, 0.1f)) sample = 0;
        if (ImGui::DragFloat("Scatter", &volume->scattering_coefficient, 0.1f)) sample = 0;
        if (ImGui::SliderFloat("Phase g", &volume->phase_g, -.9f, .9f)) sample = 0;
        ImGui::Separator();
        if (ImGui::DragFloat("Window center", &transferfunc->window_center, 0.01f)) sample = 0;
        if (ImGui::DragFloat("Window width", &transferfunc->window_width, 0.01f)) sample = 0;
        ImGui::Separator();
        if (ImGui::SliderFloat3("Vol bb min", &vol_bb_min.x, 0.f, 1.f)) sample = 0;
        if (ImGui::SliderFloat3("Vol bb max", &vol_bb_max.x, 0.f, 1.f)) sample = 0;
        ImGui::Separator();
        ImGui::Text("Model:");
        glm::mat4 row_maj = glm::transpose(volume->model);
        bool modified = false;
        if (ImGui::InputFloat4("row0", &row_maj[0][0], "%.1f")) modified = true;
        if (ImGui::InputFloat4("row1", &row_maj[1][0], "%.1f")) modified = true;
        if (ImGui::InputFloat4("row2", &row_maj[2][0], "%.1f")) modified = true;
        if (ImGui::InputFloat4("row3", &row_maj[3][0], "%.1f")) modified = true;
        if (modified) {
            volume->model = glm::transpose(row_maj);
            sample = 0;
        }
        ImGui::Separator();
        ImGui::Text("Animation time %.3f / %lu", animation->time, animation->camera_path.size());
        ImGui::Checkbox("Animation running", &animation->running);
        ImGui::InputFloat("Animation frames per node", &animation_frames_per_node);
        ImGui::PopStyleVar();
        ImGui::End();
    }
}

// ------------------------------------------
// main

int main(int argc, char** argv) {
    // init GL
    ContextParameters params;
    params.width = 1024;
    params.height = 1024;
    params.title = "VolGL";
    //params.floating = GLFW_TRUE;
    //params.resizable = GLFW_FALSE;
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
    Context::set_gui_callback(gui_callback);

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
            volume = Volume(fs::path(argv[i]).filename(), argv[i]);
    }

    // setup fbo
    const glm::ivec2 res = Context::resolution();
    fbo = Framebuffer("fbo", res.x, res.y);
    fbo->attach_depthbuffer(Texture2D("fbo/depth", res.x, res.y, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT));
    fbo->attach_colorbuffer(Texture2D("fbo/col", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(Texture2D("fbo/f_pos", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(Texture2D("fbo/f_norm", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(Texture2D("fbo/f_alb", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(Texture2D("fbo/f_vol", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(Texture2D("fbo/even", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->check();

    // reservoir buffer
    reservoir = SSBO("reservoir buffer");
    reservoir->resize(res.x * res.y * sizeof(float)*6);

    // setup trace shader and animation
    trace_shader = Shader("trace", "shader/trace.glsl");
    animation = Animation("animation");

    // load default envmap?
    if (!environment) {
        glm::vec3 color(1);
        environment = Environment("white_background", Texture2D("white_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
    }

    // load default volume?
    if (!volume)
        volume = Volume("cloud", "data/head_8bit.dat");

    // setup transfer function (LUT)
    if (!transferfunc)
        transferfunc = TransferFunction("tf", "data/AbdShaded_c.txt");

    // test default setup
    current_camera()->pos = glm::vec3(.5, .5, .3);
    current_camera()->dir = glm::vec3(-.4, -.2, -.8);
    volume->scattering_coefficient = 50;
    volume->absorbtion_coefficient = 1;
    volume->phase_g = 0.0;

    // run
    float shader_timer = 0;
    while (Context::running()) {
        // handle input
        glfwPollEvents();
        if (CameraImpl::default_input_handler(Context::frame_time()))
            sample = 0; // restart rendering

        // update and reload shaders
        current_camera()->update();
        if (animation->running && sample >= sppx) {
            animation->update(animation->ms_between_nodes / animation_frames_per_node);
            sample = 0;
            volume->scattering_coefficient = animation->eval_float("scattering_coefficient");
            volume->absorbtion_coefficient = animation->eval_float("absorbtion_coefficient");
            volume->phase_g = animation->eval_float("phase_g");
            vol_bb_min = animation->eval_vec3("vol_bb_min");
            vol_bb_max = animation->eval_vec3("vol_bb_max");
            transferfunc->window_center = animation->eval_float("window_center");
            transferfunc->window_width = animation->eval_float("window_width");
        }
        shader_timer -= Context::frame_time();
        if (shader_timer <= 0) {
            reload_modified_shaders();
            shader_timer = shader_check_delay_ms;
        }

        // render
        if (sample < sppx) {
            // bind
            trace_shader->bind();
            for (uint32_t i = 0; i < fbo->color_textures.size(); ++i)
                fbo->color_textures[i]->bind_image(i, GL_READ_WRITE, GL_RGBA32F);
            environment->cdf_U->bind_base(0);
            environment->cdf_V->bind_base(1);

            // uniforms
            trace_shader->uniform("current_sample", ++sample);
            trace_shader->uniform("bounces", bounces);
            trace_shader->uniform("show_environment", show_environment ? 0 : 1);
            // volume
            trace_shader->uniform("vol_model", volume->model);
            trace_shader->uniform("vol_inv_model", glm::inverse(volume->model));
            trace_shader->uniform("vol_tex", volume->texture, 0);
            trace_shader->uniform("vol_absorb", volume->absorbtion_coefficient);
            trace_shader->uniform("vol_scatter", volume->scattering_coefficient);
            trace_shader->uniform("vol_phase_g", volume->phase_g);
            trace_shader->uniform("vol_bb_min", vol_bb_min);
            trace_shader->uniform("vol_bb_max", vol_bb_max);
            // transfer function
            trace_shader->uniform("tf_window_center", transferfunc->window_center);
            trace_shader->uniform("tf_window_width", transferfunc->window_width);
            trace_shader->uniform("tf_lut_texture", transferfunc->texture, 1);
            // camera
            trace_shader->uniform("cam_pos", current_camera()->pos);
            trace_shader->uniform("cam_fov", current_camera()->fov_degree);
            trace_shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));
            // environment
            trace_shader->uniform("env_model", environment->model);
            trace_shader->uniform("env_inv_model", glm::inverse(environment->model));
            trace_shader->uniform("env_strength", environment->strength);
            trace_shader->uniform("env_texture", environment->texture, 2);
            trace_shader->uniform("env_integral", environment->integral);

            // trace
            const glm::ivec2 size = Context::resolution();
            trace_shader->dispatch_compute(size.x, size.y);

            // unbind
            environment->cdf_U->unbind_base(0);
            environment->cdf_V->unbind_base(1);
            for (uint32_t i = 0; i < fbo->color_textures.size(); ++i)
                fbo->color_textures[i]->unbind_image(i);
            trace_shader->unbind();
        } else
            glfwWaitEventsTimeout(1.f / 10); // 10fps idle

        // draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (show_convergence)
            convergence(fbo->color_textures[0], fbo->color_textures[fbo->color_textures.size() - 1]);
        else {
            if (tonemapping) {
                if (false || auto_exposure) {
                    // TODO compute min/max
                }
                tonemap(fbo->color_textures[0]);
            } else
                blit(fbo->color_textures[0]);
        }

        // finish frame
        Context::swap_buffers();
    }
}
