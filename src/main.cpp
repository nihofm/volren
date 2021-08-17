#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cppgl.h>
#include <voldata.h>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
namespace py = pybind11;

#include "renderer.h"

// ------------------------------------------
// CUDA

#include "cuda/common.cuh"
#include "cuda/gl.cuh"

inline float2 cast(const glm::vec2& v) { return make_float2(v.x, v.y); }
inline float3 cast(const glm::vec3& v) { return make_float3(v.x, v.y, v.z); }
inline float4 cast(const glm::vec4& v) { return make_float4(v.x, v.y, v.z, v.w); }

// ------------------------------------------
// settings

static bool adjoint = false;

static int sppx = 1024;
static bool use_vsync = true;
static float shader_check_delay_ms = 1000;

static std::shared_ptr<BackpropRendererOpenGL> renderer;

// ------------------------------------------
// helper funcs

void load_volume(const std::string& path) {
    try {
        renderer->volume = std::make_shared<voldata::Volume>(path);
        const auto [bb_min, bb_max] = renderer->volume->AABB();
        const auto extent = glm::abs(bb_max - bb_min);
        renderer->volume->model = glm::translate(glm::mat4(1), current_camera()->pos - .5f * extent + current_camera()->dir * .5f * glm::length(extent));
        renderer->commit();
        renderer->sample = 0;
    } catch (std::runtime_error& e) {
        std::cerr << "Unable to load volume from " << path << ": " << e.what() << std::endl;
    }
}

void load_envmap(const std::string& path) {
    try {
        renderer->environment = std::make_shared<Environment>(path);
        renderer->sample = 0;
    } catch (std::runtime_error& e) {
        std::cerr << "Unable to load envmap from " << path << ": " << e.what() << std::endl;
    }
}

void load_transferfunc(const std::string& path) {
    try {
        renderer->transferfunc = std::make_shared<TransferFunction>(path);
        renderer->sample = 0;
    } catch (std::runtime_error& e) {
        std::cerr << "Unable to load transferfunc from " << path << ": " << e.what() << std::endl;
    }
}

void run_script(const std::string& path) {
    try {
        py::scoped_interpreter guard{};
        py::eval_file(path);
        renderer->sample = 0;
    } catch (pybind11::error_already_set& e) {
        std::cerr << "Error executing python script " << path << ": " << e.what() << std::endl;
    }
}

void handle_path(const std::string& path) {
    if (fs::path(path).extension() == ".py")
        run_script(path);
    else if (fs::path(path).extension() == ".hdr")
        load_envmap(path);
    else if (fs::path(path).extension() == ".txt")
        load_transferfunc(path);
    else
        load_volume(path);
}

// ------------------------------------------
// callbacks

void resize_callback(int w, int h) {
    // resize buffers
    renderer->resize(w, h);
    // restart rendering
    renderer->sample = 0;
}

void keyboard_callback(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (key == GLFW_KEY_B && action == GLFW_PRESS) {
        renderer->show_environment = !renderer->show_environment;
        renderer->sample = 0;
    }
    if (key == GLFW_KEY_V && action == GLFW_PRESS) {
        use_vsync = !use_vsync;
        Context::set_swap_interval(use_vsync ? 1 : 0);
    }
    if (key == GLFW_KEY_C && action == GLFW_PRESS) {
        adjoint = !adjoint;
        renderer->sample = 0;
    }
    if (key == GLFW_KEY_U && action == GLFW_PRESS) {
        if (adjoint) {
            renderer->apply_gradients();
            renderer->sample = 0;
        }
    }
    if (key == GLFW_KEY_T && action == GLFW_PRESS)
        renderer->tonemapping = !renderer->tonemapping;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        renderer->sample = 0;
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
        const auto [min, maj] = renderer->volume->current_grid()->minorant_majorant();
        if (Context::key_pressed(GLFW_KEY_LEFT_SHIFT))
            renderer->transferfunc->window_width += (xpos - old_xpos) * (maj - min) * 0.001;
        else
            renderer->transferfunc->window_left += (xpos - old_xpos) * (maj - min) * 0.001;
        renderer->sample = 0;
    }
    old_xpos = xpos;
    old_ypos = ypos;
}

void drag_drop_callback(GLFWwindow* window, int path_count, const char* paths[]) {
    for (int i = 0; i < path_count; ++i)
        handle_path(paths[i]);
}

void gui_callback(void) {
    const glm::ivec2 size = Context::resolution();
    ImGui::SetNextWindowPos(ImVec2(size.x-300, 20));
    ImGui::SetNextWindowSize(ImVec2(300, -1));
    if (ImGui::Begin("Stuff", 0, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoFocusOnAppearing)) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, .9f);
        static float est_ravg = 0.f;
        est_ravg = glm::mix(est_ravg, float(Context::frame_time() * (sppx - renderer->sample) / 1000.f), 0.1f);
        ImGui::Text("Sample: %i/%i (est: %um, %us)", renderer->sample, sppx, uint32_t(est_ravg) / 60, uint32_t(est_ravg) % 60);
        if (ImGui::InputInt("Sppx", &sppx)) renderer->sample = 0;
        if (ImGui::InputInt("Bounces", &renderer->bounces)) renderer->sample = 0;
        if (ImGui::Checkbox("Adjoint", &adjoint)) renderer->sample = 0;
        if (ImGui::Checkbox("Vsync", &use_vsync)) Context::set_swap_interval(use_vsync ? 1 : 0);
        ImGui::Separator();
        if (ImGui::Checkbox("Environment", &renderer->show_environment)) renderer->sample = 0;
        if (ImGui::DragFloat("Env strength", &renderer->environment->strength, 0.1f)) {
            renderer->sample = 0;
            renderer->environment->strength = fmaxf(0.f, renderer->environment->strength);
        }
        ImGui::Checkbox("Tonemapping", &renderer->tonemapping);
        if (ImGui::DragFloat("Exposure", &renderer->tonemap_exposure, 0.01f))
            renderer->tonemap_exposure = fmaxf(0.f, renderer->tonemap_exposure);
        ImGui::DragFloat("Gamma", &renderer->tonemap_gamma, 0.01f);
        ImGui::SliderInt("Draw Buffer", &renderer->draw_idx, 0, renderer->textures.size() - 1);
        ImGui::Separator();
        if (ImGui::DragFloat3("Albedo", &renderer->volume->albedo.x, 0.01f, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::DragFloat("Density scale", &renderer->volume->density_scale, 0.01f, 0.01f, 1000.f)) renderer->sample = 0;
        if (ImGui::SliderFloat("Phase g", &renderer->volume->phase, -.95f, .95f)) renderer->sample = 0;
        ImGui::Separator();
        if (ImGui::DragFloat("Window left", &renderer->transferfunc->window_left, 0.01f)) renderer->sample = 0;
        if (ImGui::DragFloat("Window width", &renderer->transferfunc->window_width, 0.01f)) renderer->sample = 0;
        if (ImGui::Button("Neutral TF")) {
            renderer->transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(1) });
            renderer->transferfunc->upload_gpu();
            renderer->sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("Gradient TF")) {
            renderer->transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1) });
            renderer->transferfunc->upload_gpu();
            renderer->sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("Triangle TF")) {
            renderer->transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1), glm::vec4(0) });
            renderer->transferfunc->upload_gpu();
            renderer->sample = 0;
        }
        if (ImGui::Button("Gray background")) {
            glm::vec3 color(.5f);
            renderer->environment = std::make_shared<Environment>(Texture2D("gray_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
            renderer->sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("White background")) {
            glm::vec3 color(1);
            renderer->environment = std::make_shared<Environment>(Texture2D("white_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
            renderer->sample = 0;
        }
        ImGui::Separator();
        if (ImGui::SliderFloat3("Vol crop min", &renderer->vol_clip_min.x, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::SliderFloat3("Vol crop max", &renderer->vol_clip_max.x, 0.f, 1.f)) renderer->sample = 0;
        ImGui::Separator();
        ImGui::Text("Modelmatrix:");
        glm::mat4 row_maj = glm::transpose(renderer->volume->model);
        bool modified = false;
        if (ImGui::InputFloat4("row0", &row_maj[0][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row1", &row_maj[1][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row2", &row_maj[2][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row3", &row_maj[3][0], "%.2f")) modified = true;
        if (modified) {
            renderer->volume->model = glm::transpose(row_maj);
            renderer->sample = 0;
        }
        ImGui::Separator();
        ImGui::Text("Rotate VOLUME");
        if (ImGui::Button("90° X##V")) {
            renderer->volume->model = glm::rotate(renderer->volume->model, 1.5f * float(M_PI), glm::vec3(1, 0, 0));
            renderer->sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##V")) {
            renderer->volume->model = glm::rotate(renderer->volume->model, 1.5f * float(M_PI), glm::vec3(0, 1, 0));
            renderer->sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##V")) {
            renderer->volume->model = glm::rotate(renderer->volume->model, 1.5f * float(M_PI), glm::vec3(0, 0, 1));
            renderer->sample = 0;
        }
        ImGui::Separator();
        ImGui::Text("Rotate ENVMAP");
        if (ImGui::Button("90° X##E")) {
            renderer->environment->model = glm::mat3(glm::rotate(glm::mat4(renderer->environment->model), 1.5f * float(M_PI), glm::vec3(1, 0, 0)));
            renderer->sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##E")) {
            renderer->environment->model = glm::mat3(glm::rotate(glm::mat4(renderer->environment->model), 1.5f * float(M_PI), glm::vec3(0, 1, 0)));
            renderer->sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##E")) {
            renderer->environment->model = glm::mat3(glm::rotate(glm::mat4(renderer->environment->model), 1.5f * float(M_PI), glm::vec3(0, 0, 1)));
            renderer->sample = 0;
        }
        ImGui::PopStyleVar();
        ImGui::End();
    }
}

// ------------------------------------------
// parse command line options

static void parse_cmd(int argc, char** argv) {
    // parse cmd line args
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-w")
            Context::resize(std::stoi(argv[++i]), Context::resolution().y);
        else if (arg == "-h")
            Context::resize(Context::resolution().x, std::stoi(argv[++i]));
        else if (arg == "-spp")
            sppx = std::stoi(argv[++i]);
        else if (arg == "-b")
            renderer->bounces = std::stoi(argv[++i]);
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
            renderer->tonemap_exposure = std::stof(argv[++i]);
        else if (arg == "-gamma")
            renderer->tonemap_gamma = std::stof(argv[++i]);
        else
            handle_path(argv[i]);
    }
}

// ------------------------------------------
// main

int main(int argc, char** argv) {
    // explicitly initialize OpenGL
    RendererOpenGL::initOpenGL(1920, 1080, use_vsync, /*pinned = */false, /*visible = */true);

    // initialize the renderer(s)
    renderer = std::make_shared<BackpropRendererOpenGL>();
    renderer->init();

    // install callbacks for interactive mode
    Context::set_resize_callback(resize_callback);
    Context::set_keyboard_callback(keyboard_callback);
    Context::set_mouse_button_callback(mouse_button_callback);
    Context::set_mouse_callback(mouse_callback);
    gui_add_callback("vol_gui", gui_callback);
    glfwSetDropCallback(Context::instance().glfw_window, drag_drop_callback);
    Context::swap_buffers(); // fetch updates once

    // parse command line arguments
    parse_cmd(argc, argv);

    // set some defaults if volume has been loaded
    if (renderer->volume->grids.size() > 0) {
        const auto [bb_min, bb_max] = renderer->volume->AABB();
        const auto [min, maj] = renderer->volume->current_grid()->minorant_majorant();
        current_camera()->pos = bb_min + (bb_max - bb_min) * glm::vec3(-.5f, .5f, 0.f);
        current_camera()->dir = glm::normalize((bb_max + bb_min) * .5f - current_camera()->pos);
        renderer->transferfunc->window_left = min;
        renderer->transferfunc->window_width = maj - min;
    }

    // run the main loop
    float shader_timer = 0;
    while (Context::running()) {
        // handle input
        if (CameraImpl::default_input_handler(Context::frame_time()))
            renderer->sample = 0; // restart rendering

        // update
        current_camera()->update();
        // reload shaders?
        shader_timer -= Context::frame_time();
        if (shader_timer <= 0) {
            if (reload_modified_shaders())
                renderer->sample = 0;
            shader_timer = shader_check_delay_ms;
        }

        // render sample
        if (renderer->sample < sppx)
            renderer->trace();
        else
            glfwWaitEventsTimeout(1.f / 10); // 10fps idle

        // TODO backprop logic
        if (adjoint) {
            renderer->trace_prediction();
            renderer->backprop();
            //renderer->apply_gradients();
        }

        // draw results
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderer->draw_debug = adjoint;
        renderer->draw();

        // finish frame
        Context::swap_buffers();
    }
}
