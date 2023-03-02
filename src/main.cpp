#include <iostream>
#include <fstream>
#include <filesystem>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cppgl.h>
#include <voldata/voldata.h>

#include <pybind11/embed.h>
#include <pybind11/eval.h>

#include "renderer.h"
#include "renderer_gl.h"

using namespace cppgl;

// ------------------------------------------
// settings

static bool use_vsync = false;
static float shader_check_delay_ms = 1000;

static bool animate = false;
static float animation_fps = 30;

static bool interactive = true;
static std::string out_filename = "output.png";

static std::shared_ptr<RendererOpenGL> renderer;

// ------------------------------------------
// helper funcs

void load_volume(const std::string& path) {
    try {
        std::cout << "load volume: " << path << std::endl;
        if (fs::is_directory(path)) {
            // load contents of folder
            renderer->volume = voldata::Volume::load_folder(path, { "density", "temperature", "flame", "flames" });
        } else {
            // load single grid
            renderer->volume = std::make_shared<voldata::Volume>(path);
            // try to add emission grid
            if (std::filesystem::path(path).extension() == ".vdb") {
                for (const auto& name : { "flame", "flames", "temperature" }) {
                    try {
                        renderer->volume->update_grid_frame(renderer->volume->grid_frame_counter, voldata::Volume::load_grid(path, name), name);
                        renderer->volume->emission_scale = 1.f;
                    } catch (std::runtime_error& e) {}
                }
            }
        }
        renderer->volume->scale_and_move_to_unit_cube();
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
        renderer->transferfunc->upload_gpu();
        renderer->trace_shader = Shader("trace_tf", "shader/pathtracer_brick_tf.glsl");
        renderer->show_environment = false;
        renderer->sample = 0;
    } catch (std::runtime_error& e) {
        std::cerr << "Unable to load transferfunc from " << path << ": " << e.what() << std::endl;
    }
}

void run_script(const std::string& path) {
    try {
        pybind11::scoped_interpreter guard{};
        pybind11::eval_file(path);
        renderer->sample = 0;
    } catch (pybind11::error_already_set& e) {
        std::cerr << "Error executing python script " << path << ": " << e.what() << std::endl;
    }
}

void handle_path(const std::string& path) {
    if (std::filesystem::path(path).extension() == ".py")
        run_script(path);
    else if (std::filesystem::path(path).extension() == ".hdr")
        load_envmap(path);
    else if (std::filesystem::path(path).extension() == ".txt")
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

// ------------------------------------------
// callbacks

void resize_callback(int w, int h) {
    // resize buffers
    renderer->resize(w, h);
    // restart rendering
    renderer->reset();
}

void keyboard_callback(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    if (key == GLFW_KEY_B && action == GLFW_PRESS) {
        renderer->show_environment = !renderer->show_environment;
        renderer->reset();
    }
    if (key == GLFW_KEY_V && action == GLFW_PRESS) {
        use_vsync = !use_vsync;
        Context::set_swap_interval(use_vsync ? 1 : 0);
    }
    if (key == GLFW_KEY_T && action == GLFW_PRESS)
        renderer->tonemapping = !renderer->tonemapping;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        animate = !animate;
        renderer->reset();
    }
    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        Context::screenshot("screenshot.png");
}

void mouse_button_callback(int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
}

void mouse_callback(double xpos, double ypos) {
    static double old_xpos = -1, old_ypos = -1;
    if (old_xpos == -1 || old_ypos == -1) {
        old_xpos = xpos;
        old_ypos = ypos;
    }
    if (!ImGui::GetIO().WantCaptureMouse && renderer->transferfunc) {
        if (Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_RIGHT)) {
            const auto [min, maj] = renderer->volume->current_grid()->minorant_majorant();
            if (Context::key_pressed(GLFW_KEY_LEFT_SHIFT))
                renderer->transferfunc->window_width = glm::clamp(renderer->transferfunc->window_width + (xpos - old_xpos) * (maj - min) * 0.001, 0.0, 1.0);
            else
                renderer->transferfunc->window_left = glm::clamp(renderer->transferfunc->window_left + (xpos - old_xpos) * (maj - min) * 0.001, -1.0, 1.0);
            renderer->sample = 0;
        }
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
        est_ravg = glm::mix(est_ravg, float(Context::frame_time() * (renderer->sppx - renderer->sample) / 1000.f), 0.1f);
        ImGui::Text("Sample: %i/%i (est: %um, %us)", renderer->sample, renderer->sppx, uint32_t(est_ravg) / 60, uint32_t(est_ravg) % 60);
        if (ImGui::InputInt("Sppx", &renderer->sppx)) renderer->reset();
        if (ImGui::InputInt("Bounces", &renderer->bounces)) renderer->reset();
        if (ImGui::Checkbox("Vsync", &use_vsync)) Context::set_swap_interval(use_vsync ? 1 : 0);
        ImGui::Separator();
        if (ImGui::Checkbox("Environment", &renderer->show_environment)) renderer->reset();
        if (ImGui::DragFloat("Env strength", &renderer->environment->strength, 0.01f, 0.f, 1000.f)) renderer->reset();
        if (ImGui::Button("White background")) {
            glm::vec3 color(1);
            renderer->environment = std::make_shared<Environment>(Texture2D("white_background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
            renderer->reset();
        }
        ImGui::Separator();
        ImGui::Checkbox("Tonemapping", &renderer->tonemapping);
        if (ImGui::DragFloat("Exposure", &renderer->tonemap_exposure, 0.01f, 0.f))
            renderer->tonemap_exposure = fmaxf(0.f, renderer->tonemap_exposure);
        ImGui::DragFloat("Gamma", &renderer->tonemap_gamma, 0.01f, 0.f);
        ImGui::Separator();
        if (ImGui::DragFloat3("Albedo", &renderer->volume->albedo.x, 0.01f, 0.f, 1.f)) renderer->reset();
        if (ImGui::DragFloat("Density scale", &renderer->volume->density_scale, 0.1f, 0.f, 1e6f)) renderer->reset();
        if (ImGui::DragFloat("Emission scale", &renderer->volume->emission_scale, 0.1f, 0.f, 1e6f)) renderer->reset();
        if (ImGui::SliderFloat("Phase g", &renderer->volume->phase, -.95f, .95f)) renderer->reset();
        size_t frame_min = 0, frame_max = renderer->volume->n_grid_frames() - 1;
        if (ImGui::SliderScalar("Grid frame", ImGuiDataType_U64, &renderer->volume->grid_frame_counter, &frame_min, &frame_max)) renderer->reset();
        ImGui::Checkbox("Animate Volume", &animate);
        ImGui::SameLine();
        ImGui::DragFloat("FPS", &animation_fps, 0.01, 1, 60);
        ImGui::Separator();
        if (ImGui::Button("Clear TF")) {
            renderer->transferfunc.reset();
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Gradient TF")) {
            renderer->transferfunc = std::make_shared<TransferFunction>(std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1) }));
            renderer->transferfunc->upload_gpu();
            renderer->reset();
        }
        if (ImGui::Button("RGB TF")) {
            renderer->transferfunc = std::make_shared<TransferFunction>(std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1,0,0,0.25), glm::vec4(0,1,0,0.5), glm::vec4(0,0,1,0.75), glm::vec4(1) }));
            renderer->transferfunc->upload_gpu();
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Random TF")) {
            renderer->transferfunc = std::make_shared<TransferFunction>();
            for (int i = 0; i < 8; ++i)
                renderer->transferfunc->lut.push_back(i == 0 ? glm::vec4(0) : glm::vec4(randf(), randf(), randf(), randf()));
            renderer->transferfunc->upload_gpu();
            renderer->reset();
        }
        if (renderer->transferfunc) {
            if (ImGui::DragFloat("Window left", &renderer->transferfunc->window_left, 0.01f, -1.f, 1.f)) renderer->reset();
            if (ImGui::DragFloat("Window width", &renderer->transferfunc->window_width, 0.01f, 0.f, 1.f)) renderer->reset();
        }
        ImGui::Separator();
        if (ImGui::SliderFloat("Vol crop min X", &renderer->vol_clip_min.x, 0.f, 1.f)) renderer->reset();
        if (ImGui::SliderFloat("Vol crop min Y", &renderer->vol_clip_min.y, 0.f, 1.f)) renderer->reset();
        if (ImGui::SliderFloat("Vol crop min Z", &renderer->vol_clip_min.z, 0.f, 1.f)) renderer->reset();
        if (ImGui::SliderFloat("Vol crop max X", &renderer->vol_clip_max.x, 0.f, 1.f)) renderer->reset();
        if (ImGui::SliderFloat("Vol crop max Y", &renderer->vol_clip_max.y, 0.f, 1.f)) renderer->reset();
        if (ImGui::SliderFloat("Vol crop max Z", &renderer->vol_clip_max.z, 0.f, 1.f)) renderer->reset();
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
            renderer->reset();
        }
        ImGui::Separator();
        ImGui::Text("Rotate VOLUME");
        if (ImGui::Button("90° X##V")) {
            renderer->volume->model = glm::rotate(renderer->volume->model, .5f * float(M_PI), glm::vec3(1, 0, 0));
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##V")) {
            renderer->volume->model = glm::rotate(renderer->volume->model, .5f * float(M_PI), glm::vec3(0, 1, 0));
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##V")) {
            renderer->volume->model = glm::rotate(renderer->volume->model, .5f * float(M_PI), glm::vec3(0, 0, 1));
            renderer->reset();
        }
        ImGui::Separator();
        ImGui::Text("Rotate ENVMAP");
        if (ImGui::Button("90° X##E")) {
            renderer->environment->transform = glm::mat3(glm::rotate(glm::mat4(renderer->environment->transform), .5f * float(M_PI), glm::vec3(1, 0, 0)));
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##E")) {
            renderer->environment->transform = glm::mat3(glm::rotate(glm::mat4(renderer->environment->transform), .5f * float(M_PI), glm::vec3(0, 1, 0)));
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##E")) {
            renderer->environment->transform = glm::mat3(glm::rotate(glm::mat4(renderer->environment->transform), .5f * float(M_PI), glm::vec3(0, 0, 1)));
            renderer->reset();
        }
        ImGui::Separator();
        if (ImGui::Button("Print volume properties"))
            std::cout << "volume: " << std::endl << renderer->volume->to_string("\t") << std::endl;
        // ImGui::Separator();
        // if (ImGui::Button("Use Quilt PT")) {
        //     renderer->trace_shader = Shader("trace_quilt", "shader/pathtracer_quilt.glsl");
        //     renderer->reset();
        // }
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
    params.swap_interval = use_vsync ? 1 : 0;
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
        else if (arg == "--hidden" || arg == "--render")
            params.visible = GLFW_FALSE;
        else if (arg == "--no-decoration")
            params.decorated = GLFW_FALSE;
        else if (arg == "--floating")
            params.floating = GLFW_TRUE;
        else if (arg == "--maximised")
            params.maximised = GLFW_TRUE;
        else if (arg == "--no-debug" || arg == "--render")
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
        if (arg == "--render")
            interactive = false;
        else if (arg == "--samples" || arg == "--spp" || arg == "--sppx")
            renderer->sppx = std::stoi(argv[++i]);
        else if (arg == "--bounces")
            renderer->bounces = std::stoi(argv[++i]);
        else if (arg == "--albedo")
            renderer->volume->albedo = glm::vec3(std::stof(argv[++i]));
        else if (arg == "--density")
            renderer->volume->density_scale = std::stof(argv[++i]);
        else if (arg == "--emission")
            renderer->volume->emission_scale = std::stof(argv[++i]);
        else if (arg == "--phase")
            renderer->volume->phase = std::stof(argv[++i]);
        else if (arg == "--env_strength")
            renderer->environment->strength = std::stof(argv[++i]);
        else if (arg == "--env_rot")
            renderer->environment->transform = glm::mat3(glm::rotate(glm::mat4(renderer->environment->transform), glm::radians(std::stof(argv[++i])), glm::vec3(0, 1, 0)));
        else if (arg == "--env_hide")
            renderer->show_environment = false;
        else if (arg == "--cam_pos") {
            current_camera()->pos.x = std::stof(argv[++i]);
            current_camera()->pos.y = std::stof(argv[++i]);
            current_camera()->pos.z = std::stof(argv[++i]);
        } else if (arg == "--cam_dir") {
            current_camera()->dir.x = std::stof(argv[++i]);
            current_camera()->dir.y = std::stof(argv[++i]);
            current_camera()->dir.z = std::stof(argv[++i]);
        } else if (arg == "--cam_fov")
            current_camera()->fov_degree = std::stof(argv[++i]);
        else if (arg == "--exposure")
            renderer->tonemap_exposure = std::stof(argv[++i]);
        else if (arg == "--gamma")
            renderer->tonemap_gamma = std::stof(argv[++i]);
        // TODO XXX: this is just a hack
        else if (arg == "--quilt")
            renderer->trace_shader = Shader("trace_quilt", "shader/pathtracer_quilt.glsl");
        else if (arg == "--vol_rot_x")
            renderer->volume->model = glm::mat3(glm::rotate(glm::mat4(renderer->volume->model), glm::radians(std::stof(argv[++i])), glm::vec3(1, 0, 0)));
        else if (arg == "--vol_rot_y")
            renderer->volume->model = glm::mat3(glm::rotate(glm::mat4(renderer->volume->model), glm::radians(std::stof(argv[++i])), glm::vec3(0, 1, 0)));
        else if (arg == "--vol_rot_z")
            renderer->volume->model = glm::mat3(glm::rotate(glm::mat4(renderer->volume->model), glm::radians(std::stof(argv[++i])), glm::vec3(0, 0, 1)));
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
    renderer = std::make_shared<RendererOpenGL>();
    renderer->init();

    // install callbacks for interactive mode
    Context::set_resize_callback(resize_callback);
    Context::set_keyboard_callback(keyboard_callback);
    Context::set_mouse_button_callback(mouse_button_callback);
    Context::set_mouse_callback(mouse_callback);
    gui_add_callback("vol_gui", gui_callback);
    glfwSetDropCallback(Context::instance().glfw_window, drag_drop_callback);
    Context::swap_buffers(); // fetch updates once

    // default cam pos
    current_camera()->pos = glm::vec3(1, 0, 1);
    current_camera()->dir = glm::normalize(-current_camera()->pos);

    // parse command line arguments
    parse_cmd(argc, argv);

    // set some defaults if no volume has been loaded
    if (renderer->volume->grids.empty()) {
        // load debug box
        const uint32_t size = 4;
        const float scale = 1.f;
        std::vector<float> values = { 1, 2.5, 5, 10 };
        auto box = std::make_shared<voldata::DenseGrid>(1, 1, 4, values.data());
        box->transform = glm::translate(glm::scale(glm::mat4(1), glm::vec3(scale)), 2 * scale * current_camera()->dir + glm::vec3(0, -0.5, -2));
        renderer->volume = std::make_shared<voldata::Volume>(box);
        renderer->commit();
    }
    renderer->reset();

    if (interactive) {
        // setup timers
        auto timer_trace = TimerQueryGL("trace");
        // run the main loop
        float shader_timer = 0, animation_timer = 0;
        while (Context::running()) {
            // handle input
            if (CameraImpl::default_input_handler(Context::frame_time())) {
                renderer->reset();
            }

            // update
            current_camera()->update();
            // reload shaders?
            shader_timer -= Context::frame_time();
            if (shader_timer <= 0) {
                if (reload_modified_shaders())
                    renderer->reset();
                shader_timer = shader_check_delay_ms;
            }
            // advance animation?
            if (animate) {
                animation_timer -= Context::frame_time();
                if (animation_timer <= 0) {
                    animation_timer = 1000 / animation_fps;
                    renderer->volume->grid_frame_counter = (renderer->volume->grid_frame_counter + 1) % renderer->volume->n_grid_frames();
                    renderer->reset();
                }
            }

            // trace
            if (renderer->sample < renderer->sppx) {
                timer_trace->begin();
                renderer->trace();
                timer_trace->end();
            } else
                glfwWaitEventsTimeout(1.f / 10); // 10fps idle

            // draw results
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderer->draw();

            // finish frame
            Context::swap_buffers();
        }
    } else {
        // prepare rendering
        current_camera()->update();
        reload_modified_shaders();
        // render
        std::cout << "rendering..." << std::endl;
        for (int i = 0; i < renderer->volume->n_grid_frames(); ++i) {
            renderer->reset();
            renderer->volume->grid_frame_counter = i;
            while (renderer->sample < renderer->sppx) {
                renderer->trace();
                Context::swap_buffers(); // sync (please don't ask why this is only required for >= 1024spp)
            }
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            // tonemap
            static Shader tonemap_shader("tonemap", "shader/tonemap.glsl");
            tonemap_shader->bind();
            renderer->color->bind_image(0, GL_READ_WRITE, GL_RGBA32F);
            const glm::ivec2 resolution = Context::resolution();
            tonemap_shader->uniform("resolution", resolution);
            tonemap_shader->uniform("exposure", renderer->tonemap_exposure);
            tonemap_shader->uniform("gamma", renderer->tonemap_gamma);
            tonemap_shader->dispatch_compute(resolution.x, resolution.y);
            renderer->color->unbind_image(0);
            tonemap_shader->unbind();
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            // write result
            const size_t n_zero = 6;
            std::string out_filename = "render_" + std::string(n_zero - std::min(n_zero, std::to_string(i).length()), '0') + std::to_string(i) + ".png";
            renderer->color->save_ldr(out_filename);
            std::cout << out_filename << " written." << std::endl;
            Context::swap_buffers();
        }
    }
}
