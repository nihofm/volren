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
// settings

static bool use_vsync = false;
static float shader_check_delay_ms = 1000;

// ------------------------------------------
// helper funcs

void load_volume(const std::string& path) {
    try {
        Renderer::volume = std::make_shared<voldata::Volume>(path);
        const auto [bb_min, bb_max] = Renderer::volume->AABB();
        const auto extent = glm::abs(bb_max - bb_min);
        Renderer::volume->model = glm::translate(glm::mat4(1), current_camera()->pos - .5f * extent + current_camera()->dir * .5f * glm::length(extent));
        Renderer::commit();
        Renderer::sample = 0;
    } catch (std::runtime_error& e) {
        std::cerr << "Unable to load volume from " << path << ": " << e.what() << std::endl;
    }
}

void load_envmap(const std::string& path) {
    try {
        Renderer::environment = std::make_shared<Environment>(path);
        Renderer::sample = 0;
    } catch (std::runtime_error& e) {
        std::cerr << "Unable to load envmap from " << path << ": " << e.what() << std::endl;
    }
}

void load_transferfunc(const std::string& path) {
    try {
        Renderer::transferfunc = std::make_shared<TransferFunction>(path);
        Renderer::sample = 0;
    } catch (std::runtime_error& e) {
        std::cerr << "Unable to load transferfunc from " << path << ": " << e.what() << std::endl;
    }
}

void run_script(const std::string& path) {
    try {
        // TODO regular vs embedded module (either or)
        py::scoped_interpreter guard{};
        py::eval_file(path);
        Renderer::sample = 0;
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
    Renderer::fbo->resize(w, h);
    Renderer::Renderer::sample = 0; // restart rendering
}

void keyboard_callback(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (key == GLFW_KEY_B && action == GLFW_PRESS) {
        Renderer::show_environment = !Renderer::show_environment;
        Renderer::sample = 0;
    }
    if (key == GLFW_KEY_V && action == GLFW_PRESS) {
        use_vsync = !use_vsync;
        Context::set_swap_interval(use_vsync ? 1 : 0);
    }
    if (key == GLFW_KEY_T && action == GLFW_PRESS)
        Renderer::tonemapping = !Renderer::tonemapping;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        Renderer::sample = 0;
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
        const auto [min, maj] = Renderer::volume->current_grid()->minorant_majorant();
        if (Context::key_pressed(GLFW_KEY_LEFT_SHIFT))
            Renderer::transferfunc->window_width += (xpos - old_xpos) * (maj - min) * 0.001;
        else
            Renderer::transferfunc->window_left += (xpos - old_xpos) * (maj - min) * 0.001;
        Renderer::sample = 0;
    }
    old_xpos = xpos;
    old_ypos = ypos;
}

void drag_drop_callback(GLFWwindow* window, int path_count, const char* paths[]) {
    for (int i = 0; i < path_count; ++i) {
        handle_path(paths[i]);
    }
}

void gui_callback(void) {
    const glm::ivec2 size = Context::resolution();
    ImGui::SetNextWindowPos(ImVec2(size.x-260, 20));
    ImGui::SetNextWindowSize(ImVec2(300, -1));
    if (ImGui::Begin("Stuff", 0, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoFocusOnAppearing)) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, .9f);
        static float est_ravg = 0.f;
        est_ravg = glm::mix(est_ravg, float(Context::frame_time() * (Renderer::sppx - Renderer::sample) / 1000.f), 0.1f);
        ImGui::Text("Sample: %i/%i (est: %um, %us)", Renderer::sample, Renderer::sppx, uint32_t(est_ravg) / 60, uint32_t(est_ravg) % 60);
        if (ImGui::InputInt("Sppx", &Renderer::sppx)) Renderer::sample = 0;
        if (ImGui::InputInt("Bounces", &Renderer::bounces)) Renderer::sample = 0;
        if (ImGui::Checkbox("Vsync", &use_vsync)) Context::set_swap_interval(use_vsync ? 1 : 0);
        ImGui::Separator();
        if (ImGui::Checkbox("Environment", &Renderer::show_environment)) Renderer::sample = 0;
        if (ImGui::DragFloat("Env strength", &Renderer::environment->strength, 0.1f)) {
            Renderer::sample = 0;
            Renderer::environment->strength = fmaxf(0.f, Renderer::environment->strength);
        }
        ImGui::Checkbox("Tonemapping", &Renderer::tonemapping);
        if (ImGui::DragFloat("Exposure", &Renderer::tonemap_exposure, 0.01f))
            Renderer::tonemap_exposure = fmaxf(0.f, Renderer::tonemap_exposure);
        ImGui::DragFloat("Gamma", &Renderer::tonemap_gamma, 0.01f);
        ImGui::Separator();
        if (ImGui::DragFloat3("Albedo", &Renderer::volume->albedo.x, 0.01f, 0.f, 1.f)) Renderer::sample = 0;
        if (ImGui::DragFloat("Density scale", &Renderer::volume->density_scale, 0.01f, 0.01f, 1000.f)) Renderer::sample = 0;
        if (ImGui::SliderFloat("Phase g", &Renderer::volume->phase, -.95f, .95f)) Renderer::sample = 0;
        ImGui::Separator();
        if (ImGui::DragFloat("Window left", &Renderer::transferfunc->window_left, 0.01f)) Renderer::sample = 0;
        if (ImGui::DragFloat("Window width", &Renderer::transferfunc->window_width, 0.01f)) Renderer::sample = 0;
        if (ImGui::Button("Neutral TF")) {
            Renderer::transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(1) });
            Renderer::transferfunc->upload_gpu();
            Renderer::sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("Gradient TF")) {
            Renderer::transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1) });
            Renderer::transferfunc->upload_gpu();
            Renderer::sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("Triangle TF")) {
            Renderer::transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1), glm::vec4(0) });
            Renderer::transferfunc->upload_gpu();
            Renderer::sample = 0;
        }
        if (ImGui::Button("Gray background")) {
            glm::vec3 color(.5f);
            Renderer::environment = std::make_shared<Environment>(Texture2D("gray_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
            Renderer::sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("White background")) {
            glm::vec3 color(1);
            Renderer::environment = std::make_shared<Environment>(Texture2D("white_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
            Renderer::sample = 0;
        }
        ImGui::Separator();
        if (ImGui::SliderFloat3("Vol crop min", &Renderer::vol_crop_min.x, 0.f, 1.f)) Renderer::sample = 0;
        if (ImGui::SliderFloat3("Vol crop max", &Renderer::vol_crop_max.x, 0.f, 1.f)) Renderer::sample = 0;
        ImGui::Separator();
        ImGui::Text("Modelmatrix:");
        glm::mat4 row_maj = glm::transpose(Renderer::volume->model);
        bool modified = false;
        if (ImGui::InputFloat4("row0", &row_maj[0][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row1", &row_maj[1][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row2", &row_maj[2][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row3", &row_maj[3][0], "%.2f")) modified = true;
        if (modified) {
            Renderer::volume->model = glm::transpose(row_maj);
            Renderer::sample = 0;
        }
        ImGui::Separator();
        ImGui::Text("Rotate VOLUME");
        if (ImGui::Button("90° X##V")) {
            Renderer::volume->model = glm::rotate(Renderer::volume->model, 1.5f * float(M_PI), glm::vec3(1, 0, 0));
            Renderer::sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##V")) {
            Renderer::volume->model = glm::rotate(Renderer::volume->model, 1.5f * float(M_PI), glm::vec3(0, 1, 0));
            Renderer::sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##V")) {
            Renderer::volume->model = glm::rotate(Renderer::volume->model, 1.5f * float(M_PI), glm::vec3(0, 0, 1));
            Renderer::sample = 0;
        }
        ImGui::Separator();
        ImGui::Text("Rotate ENVMAP");
        if (ImGui::Button("90° X##E")) {
            Renderer::environment->model = glm::mat3(glm::rotate(glm::mat4(Renderer::environment->model), 1.5f * float(M_PI), glm::vec3(1, 0, 0)));
            Renderer::sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##E")) {
            Renderer::environment->model = glm::mat3(glm::rotate(glm::mat4(Renderer::environment->model), 1.5f * float(M_PI), glm::vec3(0, 1, 0)));
            Renderer::sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##E")) {
            Renderer::environment->model = glm::mat3(glm::rotate(glm::mat4(Renderer::environment->model), 1.5f * float(M_PI), glm::vec3(0, 0, 1)));
            Renderer::sample = 0;
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
            Renderer::sppx = std::stoi(argv[++i]);
        else if (arg == "-b")
            Renderer::bounces = std::stoi(argv[++i]);
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
            Renderer::tonemap_exposure = std::stof(argv[++i]);
        else if (arg == "-gamma")
            Renderer::tonemap_gamma = std::stof(argv[++i]);
        else
            handle_path(argv[i]);
    }
}

// ------------------------------------------
// main

int main(int argc, char** argv) {
    // initialize the renderer
    Renderer::init(1920, 1080, use_vsync, /*pinned = */true, /*visible = */true);

    // install callbacks for interactive mode
    Context::set_resize_callback(resize_callback);
    Context::set_keyboard_callback(keyboard_callback);
    Context::set_mouse_button_callback(mouse_button_callback);
    Context::set_mouse_callback(mouse_callback);
    gui_add_callback("vol_gui", gui_callback);
    glfwSetDropCallback(Context::instance().glfw_window, drag_drop_callback);

    // parse command line arguments
    parse_cmd(argc, argv);

    // set some defaults if volume has been loaded
    if (Renderer::volume->grids.size() > 0) {
        const auto [bb_min, bb_max] = Renderer::volume->AABB();
        const auto [min, maj] = Renderer::volume->current_grid()->minorant_majorant();
        current_camera()->pos = bb_min + (bb_max - bb_min) * glm::vec3(-.5f, .5f, 0.f);
        current_camera()->dir = glm::normalize((bb_max + bb_min)*.5f - current_camera()->pos);
        Renderer::transferfunc->window_left = min;
        Renderer::transferfunc->window_width = maj - min;
    }

    // run the main loop
    float shader_timer = 0;
    while (Context::running()) {
        // handle input
        if (CameraImpl::default_input_handler(Context::frame_time()))
            Renderer::sample = 0; // restart rendering

        // update
        current_camera()->update();
        // reload shaders?
        shader_timer -= Context::frame_time();
        if (shader_timer <= 0) {
            if (reload_modified_shaders())
                Renderer::sample = 0;
            shader_timer = shader_check_delay_ms;
        }

        // render
        if (Renderer::sample < Renderer::sppx)
            Renderer::trace();
        else
            glfwWaitEventsTimeout(1.f / 10); // 10fps idle

        // draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Renderer::draw();

        // finish frame
        Context::swap_buffers();
    }
}
