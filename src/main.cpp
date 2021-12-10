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
#include "renderer_gl.h"
#include "cuda/renderer_optix.h"

// ------------------------------------------
// settings

static bool adjoint = false, randomize = false;

static int sppx = 1024;
static bool use_vsync = true;
static float shader_check_delay_ms = 1000;

static bool animate = false;
static float animation_fps = 30;
static bool render_animation = false; // TODO animation rendering

static std::shared_ptr<BackpropRendererOpenGL> renderer;

// ------------------------------------------
// helper funcs

void load_volume(const std::string& path) {
    try {
        if (fs::is_directory(path)) {
            // load contents of folder
            // TODO FIXME handle empty emission grids
            renderer->volume = voldata::Volume::load_folder(path, { "density", "flame", "temperature" }); // TODO: hardcoded grid names
        } else {
            // load single grid
            renderer->volume = std::make_shared<voldata::Volume>(path);
            const auto [bb_min, bb_max] = renderer->volume->AABB();
            const auto extent = glm::abs(bb_max - bb_min);
            renderer->volume->model = glm::translate(glm::mat4(1), current_camera()->pos - .5f * extent + current_camera()->dir * .5f * glm::length(extent));
            // try to add emission grid
            if (fs::path(path).extension() == ".vdb") {
                try {
                    renderer->volume->update_grid_frame(renderer->volume->grid_frame_counter, voldata::Volume::load_grid(path, "flame"), "flame");
                    renderer->volume->emission_scale = 1.f;
                } catch (std::runtime_error& e) {}
                try {
                    renderer->volume->update_grid_frame(renderer->volume->grid_frame_counter, voldata::Volume::load_grid(path, "temperature"), "temperature");
                    renderer->volume->emission_scale = 1.f;
                } catch (std::runtime_error& e) {}
            }
        }
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
        renderer->commit();
        renderer->sample = 0;
        renderer->reset_optimization = true;
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

inline float randf() { return rand() / (RAND_MAX + 1.f); }

inline float vandercorput(uint32_t i, uint32_t scramble) {
    i = (i << 16) | (i >> 16);
    i = ((i & 0x00ff00ff) << 8) | ((i & 0xff00ff00) >> 8);
    i = ((i & 0x0f0f0f0f) << 4) | ((i & 0xf0f0f0f0) >> 4);
    i = ((i & 0x33333333) << 2) | ((i & 0xcccccccc) >> 2);
    i = ((i & 0x55555555) << 1) | ((i & 0xaaaaaaaa) >> 1);
    i ^= scramble;
    return ((i >> 8) & 0xffffff) / float(1 << 24);
}

inline float sobol2(uint32_t i, uint32_t scramble) {
    for (uint32_t v = 1 << 31; i != 0; i >>= 1, v ^= v >> 1)
        if (i & 0x1)
            scramble ^= v;
    return ((scramble >> 8) & 0xffffff) / float(1 << 24);
}

inline glm::vec2 sample02(uint32_t i) {
    return glm::vec2(vandercorput(i, 0xDEADBEEF), sobol2(i, 0x8BADF00D));
}

glm::vec3 uniform_sample_sphere() {
    static uint32_t i = 0;
    const glm::vec2 sample = sample02(++i);
    const float z = 1.f - 2.f * sample.x;
    const float r = sqrtf(fmaxf(0.f, 1.f - z * z));
    const float phi = 2.f * M_PI * sample.y;
    return glm::vec3(r * cosf(phi), r * sinf(phi), z);
}

void randomize_parameters() {
    // randomize camera
    const auto& [bb_min, bb_max] = renderer->volume->AABB();
    const auto& center = bb_min + (bb_max - bb_min) * .5f;
    const float radius = glm::length(bb_max - bb_min) * .5f;
    current_camera()->pos = center + uniform_sample_sphere() * radius;
    current_camera()->dir = glm::normalize(center + uniform_sample_sphere() * .1f * radius - current_camera()->pos);
    current_camera()->up = glm::vec3(0, 1, 0);
    current_camera()->fov_degree = 40 + randf() * 60;
    // randomize volume
    renderer->volume->density_scale = 0.01 + randf() * 2.f;
    // randomize TF
    renderer->transferfunc->window_left = randf() * 0.5;
    renderer->transferfunc->window_width = 0.5 + randf() * 0.5;
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
    if (key == GLFW_KEY_H && action == GLFW_PRESS) {
        // DEBUG AffineTransform class
        renderer->volume->current_grid()->transform2.from_mat4x4(renderer->volume->get_transform());
        std::cout << "Matrix4x4:" << std::endl << glm::to_string(renderer->volume->get_transform()) << std::endl;
        std::cout << "AffineTransform:" << std::endl << renderer->volume->current_grid()->transform2.to_string() << std::endl;
    }
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
        renderer->backprop_sample = 0;
    }
    if (key == GLFW_KEY_X && action == GLFW_PRESS) {
        randomize = !randomize;
        renderer->sample = 0;
        renderer->backprop_sample = 0;
    }
    if (key == GLFW_KEY_T && action == GLFW_PRESS)
        renderer->tonemapping = !renderer->tonemapping;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        renderer->sample = 0;
        animate = !animate;
    }
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
            renderer->transferfunc->window_width = clamp(renderer->transferfunc->window_width + (xpos - old_xpos) * (maj - min) * 0.001, 0.f, 1.f);
        else
            renderer->transferfunc->window_left = clamp(renderer->transferfunc->window_left + (xpos - old_xpos) * (maj - min) * 0.001, -1.f, 1.f);
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
        if (ImGui::Checkbox("Vsync", &use_vsync)) Context::set_swap_interval(use_vsync ? 1 : 0);
        if (ImGui::Button("Use Brick PT")) {
            renderer->trace_shader = Shader("trace brick", "shader/pathtracer_brick.glsl");
            renderer->sample = 0;
            renderer->backprop_sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("Use Irradiance PT")) {
            renderer->trace_shader = Shader("trace brick irradiance", "shader/pathtracer_irradiance.glsl");
            renderer->sample = 0;
            renderer->backprop_sample = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear Irradiance Cache")) {
            renderer->irradiance_cache->clear();
            renderer->sample = 0;
            renderer->backprop_sample = 0;
        }
        ImGui::Separator();
        ImGui::Text("Backprop:");
        if (ImGui::InputInt("Backprop sppx", &renderer->backprop_sppx)) {
            renderer->sample = 0;
            renderer->backprop_sample = 0;
        }
        ImGui::DragFloat("Learning rate", &renderer->learning_rate, 0.0001f, 0.0001f, 1.f);
        if (ImGui::Checkbox("Adjoint", &adjoint)) { 
            renderer->sample = 0;
            renderer->backprop_sample = 0;
        }
        ImGui::Checkbox("Randomize params", &randomize);
        if (ImGui::Button("Reset"))
            renderer->reset_optimization = true;
        ImGui::SameLine();
        if (ImGui::Button("Solve"))
            renderer->solve_optimization = true;
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
        ImGui::Separator();
        if (ImGui::DragFloat3("Albedo", &renderer->volume->albedo.x, 0.01f, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::DragFloat("Density scale", &renderer->volume->density_scale, 0.01f, 0.01f, 1000.f)) renderer->sample = 0;
        if (ImGui::DragFloat("Emission scale", &renderer->volume->emission_scale, 0.01f, 0.f, 1000.f)) renderer->sample = 0;
        if (ImGui::SliderFloat("Phase g", &renderer->volume->phase, -.95f, .95f)) renderer->sample = 0;
        size_t frame_min = 0, frame_max = renderer->volume->n_grid_frames() - 1;
        if (ImGui::SliderScalar("Grid frame", ImGuiDataType_U64, &renderer->volume->grid_frame_counter, &frame_min, &frame_max)) renderer->sample = 0;
        ImGui::Checkbox("Animate Volume", &animate);
        ImGui::SameLine();
        ImGui::DragFloat("FPS", &animation_fps, 0.01, 1, 60);
        ImGui::Separator();
        if (ImGui::DragFloat("Window left", &renderer->transferfunc->window_left, 0.01f, -1.f, 1.f)) renderer->sample = 0;
        if (ImGui::DragFloat("Window width", &renderer->transferfunc->window_width, 0.01f, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::Button("Neutral TF")) {
            renderer->transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(1) });
            renderer->transferfunc->upload_gpu();
            renderer->sample = 0;
            renderer->commit();
            renderer->reset_optimization = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Gradient TF")) {
            renderer->transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1) });
            renderer->transferfunc->upload_gpu();
            renderer->sample = 0;
            renderer->commit();
            renderer->reset_optimization = true;
        }
        if (ImGui::Button("RGB TF")) {
            renderer->transferfunc->lut = std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1,0,0,0.25), glm::vec4(0,1,0,0.5), glm::vec4(0,0,1,0.75), glm::vec4(1) });
            renderer->transferfunc->upload_gpu();
            renderer->sample = 0;
            renderer->commit();
            renderer->reset_optimization = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("RNG TF")) {
            renderer->transferfunc->lut.clear();
            const int N = 32;
            for (int i = 0; i < N; ++i)
                renderer->transferfunc->lut.push_back(glm::vec4(randf(), randf(), randf(), randf()));
            renderer->transferfunc->upload_gpu();
            renderer->sample = 0;
            renderer->commit();
            renderer->reset_optimization = true;
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
        if (ImGui::SliderFloat("Vol crop min X", &renderer->vol_clip_min.x, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::SliderFloat("Vol crop min Y", &renderer->vol_clip_min.y, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::SliderFloat("Vol crop min Z", &renderer->vol_clip_min.z, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::SliderFloat("Vol crop max X", &renderer->vol_clip_max.x, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::SliderFloat("Vol crop max Y", &renderer->vol_clip_max.y, 0.f, 1.f)) renderer->sample = 0;
        if (ImGui::SliderFloat("Vol crop max Z", &renderer->vol_clip_max.z, 0.f, 1.f)) renderer->sample = 0;
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
// command line options

// parse gl cmd line args
static void init_opengl_from_args(int argc, char** argv) {
    // collect args
    ContextParameters params;
    params.title = "VolRen";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-w")
            params.width = std::stoi(argv[++i]);
        else if (arg == "-h")
            params.height = std::stoi(argv[++i]);
        else if (arg == "--title")
            params.title = argv[++i];
        else if (arg == "--major")
            params.gl_major = std::stoi(argv[++i]);
        else if (arg == "--minor")
            params.gl_major = std::stoi(argv[++i]);
        else if (arg == "--no-resize")
            params.resizable = GLFW_FALSE;
        else if (arg == "--hidden")
            params.visible = GLFW_FALSE;
        else if (arg == "--no-decoration")
            params.decorated = GLFW_FALSE;
        else if (arg == "--floating")
            params.floating = GLFW_TRUE;
        else if (arg == "--maximised")
            params.maximised = GLFW_TRUE;
        else if (arg == "--no-debug")
            params.gl_debug_context = GLFW_FALSE;
        else if (arg == "--swap")
            params.swap_interval = std::stoi(argv[++i]);
        else if (arg == "--font")
            params.font_ttf_filename = argv[++i];
        else if (arg == "--fontsize")
            params.font_size_pixels = std::stoi(argv[++i]);
    }
    // create context
    try  {
        Context::init(params);
    } catch (std::runtime_error& e) {
        std::cerr << "Failed to create context: " << e.what() << std::endl;
        std::cerr << "Retrying for offline rendering..." << std::endl;
        params.visible = GLFW_FALSE;
        Context::init(params);
    }
}

// parse regular cmd line args
static void parse_cmd(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-spp")
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
    // initialize OpenGL
    init_opengl_from_args(argc, argv);

    // initialize Renderer
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
        current_camera()->pos = bb_min + (bb_max - bb_min) * glm::vec3(-.5f, .5f, 0.f);
        current_camera()->dir = glm::normalize((bb_max + bb_min) * .5f - current_camera()->pos);
    } else {
        // load debug box
        const uint32_t size = 4;
        const float scale = 1.f;
        std::vector<float> values = { 1, 2.5, 5, 10 };
        auto box = std::make_shared<voldata::DenseGrid>(1, 1, 4, values.data());
        box->transform = glm::translate(glm::scale(glm::mat4(1), glm::vec3(scale)), 2 * scale * current_camera()->dir + glm::vec3(0, -0.5, -2));
        renderer->volume = std::make_shared<voldata::Volume>(box);
        renderer->commit();
    }

    // setup timers
    auto timer_trace = TimerQueryGL("trace");
    auto timer_backprop = TimerQueryGL("backprop");
    auto timer_update = TimerQueryGL("gradient step");

    // run the main loop
    float shader_timer = 0, animation_timer = 0;
    while (Context::running()) {
        // handle input
        if (CameraImpl::default_input_handler(Context::frame_time())) {
            renderer->sample = 0;
            renderer->backprop_sample = 0;
        }

        // update
        current_camera()->update();
        // reload shaders?
        shader_timer -= Context::frame_time();
        if (shader_timer <= 0) {
            if (reload_modified_shaders()) {
                renderer->sample = 0;
                renderer->backprop_sample = 0;
            }
            shader_timer = shader_check_delay_ms;
        }
        // advance animation?
        if (animate) {
            animation_timer -= Context::frame_time();
            if (animation_timer <= 0) {
                animation_timer = 1000 / animation_fps;
                renderer->volume->grid_frame_counter = (renderer->volume->grid_frame_counter + 1) % renderer->volume->n_grid_frames();
                renderer->sample = 0;
            }
        }

        // trace
        if (renderer->sample < sppx) {
            // forward rendering
            renderer->sample++;
            timer_trace->begin();
            renderer->trace();
            timer_trace->end();
        } else if (adjoint) {
            if (renderer->backprop_sample < renderer->backprop_sppx) {
                // radiative backprop
                renderer->backprop_sample++;
                timer_backprop->begin();
                renderer->backprop();
                timer_backprop->end();
            } else {
                // gradient update step
                timer_update->begin();
                renderer->step();
                timer_update->end();
                if (randomize) randomize_parameters();
                // reset
                renderer->sample = 0;
                renderer->backprop_sample = 0;
                renderer->seed = rand();
            }
        } else
            glfwWaitEventsTimeout(1.f / 10); // 10fps idle

        // draw results
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (adjoint)
            renderer->draw_adjoint();
        else
            renderer->draw();

        // finish frame
        Context::swap_buffers();
    }

    // cleanup
    renderer.reset();
}
