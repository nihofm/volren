#include <iostream>
#include <fstream>
#include <filesystem>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cppgl.h>
#include <voldata.h>

#include <pybind11/embed.h>
#include <pybind11/eval.h>


#include "renderer.h"

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

inline float randf() { return rand() / (RAND_MAX + 1.f); }

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
                    } catch (std::runtime_error& e) {}
                }
            }
        }
        renderer->density_scale = 1.f;
        renderer->scale_and_move_to_unit_cube();
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
    // --------------------------------------------------------------
    // LNdW 2023 demo scenes
    // std::filesystem::path base_dir = "/media/ul40ovyj/T7 Touch/";
    std::filesystem::path base_dir = "/media/niko/T7 Touch/";
    // head demo
    if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
        load_volume(base_dir / "data/head_8bit.dat");
        load_envmap(base_dir / "envmaps/chapmans_drive_2k.hdr");
        renderer->show_environment = false;
        renderer->environment->strength = 10.f;
        renderer->albedo = glm::vec3(0.9);
        renderer->density_scale = 500.f;
        renderer->emission_scale = 0.f;
        renderer->phase = 0.f;
        renderer->transferfunc = std::make_shared<TransferFunction>();
        renderer->transferfunc->colormap(tinycolormap::ColormapType::Cubehelix);
        renderer->transferfunc->window_left = 0.235f;
        renderer->transferfunc->window_width = 0.145f;
        renderer->vol_clip_min = glm::vec3(0.38, 0.0, 0.0);
        renderer->vol_clip_max = glm::vec3(1.0, 1.0, 1.0);
        current_camera()->pos = glm::vec3(-0.8, 0.1, -0.32);
        current_camera()->dir = normalize(glm::vec3(0.99, -0.1, -0.05));
        current_camera()->up = glm::vec3(0, 1, 0);
        animate = false;
        renderer->reset();
    }
    // fullbody demo
    if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
        load_volume(base_dir / "data/fullbody.brick");
        renderer->volume->transform = glm::rotate(renderer->volume->transform, 1.f * float(M_PI), glm::vec3(0, 0, 1));
        load_envmap(base_dir / "envmaps/kiara_8_sunset_2k.hdr");
        renderer->show_environment = false;
        renderer->environment->strength = 3.f;
        renderer->albedo = glm::vec3(0.5);
        renderer->density_scale = 750.f;
        renderer->emission_scale = 0.f;
        renderer->phase = 0.f;
        renderer->transferfunc = std::make_shared<TransferFunction>();
        renderer->transferfunc->colormap(tinycolormap::ColormapType::Cubehelix);
        renderer->transferfunc->window_left = 0.242f;
        renderer->transferfunc->window_width = 0.081f;
        // load_transferfunc(base_dir / "data/SplineShaded.txt");
        // renderer->transferfunc->window_left = 0.150f;
        // renderer->transferfunc->window_width = 0.290f;
        renderer->vol_clip_min = glm::vec3(0.0, 0.5, 0.0);
        renderer->vol_clip_max = glm::vec3(1.0, 1.0, 1.0);
        current_camera()->pos = glm::vec3(-0.075, 0.42, -0.12);
        current_camera()->dir = normalize(glm::vec3(0.25, -0.95, -0.22));
        current_camera()->up = glm::vec3(0, 1, 0);
        animate = false;
        renderer->reset();
    }
    //  demo objektiv
    if (key == GLFW_KEY_3 && action == GLFW_PRESS) {
        load_volume(base_dir / "volumes/objektiv.brick");
        load_envmap(base_dir / "envmaps/forest_slope_2k.hdr");
        renderer->show_environment = false;
        renderer->environment->strength = 3.f;
        renderer->albedo = glm::vec3(1.0);
        renderer->density_scale = 200.f;
        renderer->emission_scale = 0.f;
        renderer->phase = 0.f;
        renderer->transferfunc = std::make_shared<TransferFunction>();
        renderer->transferfunc->colormap(tinycolormap::ColormapType::Cividis);
        renderer->transferfunc->window_left = 0.146f;
        renderer->transferfunc->window_width = 0.479f;
        renderer->vol_clip_min = glm::vec3(0.0, 0.0, 0.5);
        renderer->vol_clip_max = glm::vec3(1.0, 1.0, 1.0);
        current_camera()->pos = glm::vec3(0.27, 0.04, -0.58);
        current_camera()->dir = normalize(glm::vec3(-0.30, -0.02, 0.95));
        current_camera()->up = glm::vec3(0, 1, 0);
        animate = false;
        renderer->reset();
    }
    // cloud demo
    if (key == GLFW_KEY_4 && action == GLFW_PRESS) {
        load_volume(base_dir / "volumes/wdas_cloud/wdas_cloud_half.brick");
        load_envmap(base_dir / "envmaps/syferfontein_1d_clear_4k.hdr");
        renderer->show_environment = true;
        renderer->environment->strength = 1.f;
        renderer->albedo = glm::vec3(1.0);
        renderer->density_scale = 200.f;
        renderer->emission_scale = 0.f;
        renderer->phase = 0.f;
        renderer->transferfunc.reset();
        renderer->vol_clip_min = glm::vec3(0.0, 0.0, 0.0);
        renderer->vol_clip_max = glm::vec3(1.0, 1.0, 1.0);
        current_camera()->pos = glm::vec3(0.560, -0.249, -0.271);
        current_camera()->dir = normalize(glm::vec3(-0.81, 0.40, 0.42));
        current_camera()->up = glm::vec3(0, 1, 0);
        animate = false;
        renderer->reset();
    }
    // tornado demo
    if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
        load_volume(base_dir / "volumes/tornado_brick");
        renderer->volume->transform = glm::rotate(renderer->volume->transform, 1.5f * float(M_PI), glm::vec3(1, 0, 0));
        load_envmap(base_dir / "envmaps/lilienstein_2k.hdr");
        renderer->show_environment = true;
        renderer->environment->strength = 1.f;
        renderer->albedo = glm::vec3(0.53, 0.37, 0.16);
        renderer->density_scale = 500.f;
        renderer->emission_scale = 0.f;
        renderer->phase = 0.f;
        renderer->transferfunc.reset();
        renderer->vol_clip_min = glm::vec3(0.0, 0.0, 0.0);
        renderer->vol_clip_max = glm::vec3(1.0, 1.0, 1.0);
        current_camera()->pos = glm::vec3(-0.89, -0.52, 0.37);
        current_camera()->dir = normalize(glm::vec3(0.91, 0.31, -0.26));
        current_camera()->up = glm::vec3(0, 1, 0);
        animate = true;
        renderer->reset();
    }
    // explosion demo
    if (key == GLFW_KEY_6 && action == GLFW_PRESS) {
        load_volume(base_dir / "volumes/air_explosion_brick");
        renderer->volume->transform = glm::rotate(renderer->volume->transform, 1.5f * float(M_PI), glm::vec3(1, 0, 0));
        load_envmap(base_dir / "envmaps/mpumalanga_veld_2k.hdr");
        renderer->show_environment = true;
        renderer->environment->strength = 1.f;
        renderer->albedo = glm::vec3(0.5);
        renderer->density_scale = 500.f;
        renderer->emission_scale = 200.f;
        renderer->phase = 0.f;
        renderer->transferfunc.reset();
        renderer->vol_clip_min = glm::vec3(0.0, 0.0, 0.0);
        renderer->vol_clip_max = glm::vec3(1.0, 1.0, 1.0);
        current_camera()->pos = glm::vec3(0.056, 0.675, 0.114);
        current_camera()->dir = normalize(glm::vec3(-0.08, -0.99, -0.06));
        current_camera()->up = glm::vec3(0, 1, 0);
        animate = true;
        renderer->reset();
    }
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
            else {
                renderer->transferfunc->window_left = glm::clamp(renderer->transferfunc->window_left + (xpos - old_xpos) * (maj - min) * 0.001, -1.0, 1.0);
                // renderer->transferfunc->window_width = 1.0 - renderer->transferfunc->window_left;
            }
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
        if (ImGui::DragFloat3("Albedo", &renderer->albedo.x, 0.01f, 0.f, 1.f)) renderer->reset();
        if (ImGui::DragFloat("Density scale", &renderer->density_scale, 0.1f, 0.f, 1e6f)) renderer->reset();
        if (ImGui::DragFloat("Emission scale", &renderer->emission_scale, 0.1f, 0.f, 1e6f)) renderer->reset();
        if (ImGui::SliderFloat("Phase g", &renderer->phase, -.95f, .95f)) renderer->reset();
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
        if (ImGui::Button("Random TF")) {
            renderer->transferfunc = std::make_shared<TransferFunction>();
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Gradient TF")) {
            renderer->transferfunc = std::make_shared<TransferFunction>(std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1) }));
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("RGB TF")) {
            renderer->transferfunc = std::make_shared<TransferFunction>(std::vector<glm::vec4>({ glm::vec4(0), glm::vec4(1,0,0,0.25), glm::vec4(0,1,0,0.5), glm::vec4(0,0,1,0.75), glm::vec4(1) }));
            renderer->reset();
        }
        if (ImGui::Button("Turbo")) {
            renderer->transferfunc = std::make_shared<TransferFunction>();
            renderer->transferfunc->colormap(tinycolormap::ColormapType::Turbo);
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Viridis")) {
            renderer->transferfunc = std::make_shared<TransferFunction>();
            renderer->transferfunc->colormap(tinycolormap::ColormapType::Viridis);
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cividis")) {
            renderer->transferfunc = std::make_shared<TransferFunction>();
            renderer->transferfunc->colormap(tinycolormap::ColormapType::Cividis);
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Magma")) {
            renderer->transferfunc = std::make_shared<TransferFunction>();
            renderer->transferfunc->colormap(tinycolormap::ColormapType::Magma);
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cubehelix")) {
            renderer->transferfunc = std::make_shared<TransferFunction>();
            renderer->transferfunc->colormap(tinycolormap::ColormapType::Cubehelix);
            renderer->reset();
        }
        if (renderer->transferfunc) {
            if (ImGui::Button("Write TF"))
                    renderer->transferfunc->write_to_file("tf_lut.txt");
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
        glm::mat4 row_maj = glm::transpose(renderer->volume->transform);
        bool modified = false;
        if (ImGui::InputFloat4("row0", &row_maj[0][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row1", &row_maj[1][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row2", &row_maj[2][0], "%.2f")) modified = true;
        if (ImGui::InputFloat4("row3", &row_maj[3][0], "%.2f")) modified = true;
        if (modified) {
            renderer->volume->transform = glm::transpose(row_maj);
            renderer->reset();
        }
        ImGui::Separator();
        ImGui::Text("Rotate VOLUME");
        if (ImGui::Button("90° X##V")) {
            renderer->volume->transform = glm::rotate(renderer->volume->transform, .5f * float(M_PI), glm::vec3(1, 0, 0));
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Y##V")) {
            renderer->volume->transform = glm::rotate(renderer->volume->transform, .5f * float(M_PI), glm::vec3(0, 1, 0));
            renderer->reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("90° Z##V")) {
            renderer->volume->transform = glm::rotate(renderer->volume->transform, .5f * float(M_PI), glm::vec3(0, 0, 1));
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
    params.gl_debug_context = GLFW_FALSE;
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
        else if (arg == "---debug")
            params.gl_debug_context = GLFW_TRUE;
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
            renderer->albedo = glm::vec3(std::stof(argv[++i]));
        else if (arg == "--density")
            renderer->density_scale = std::stof(argv[++i]);
        else if (arg == "--emission")
            renderer->emission_scale = std::stof(argv[++i]);
        else if (arg == "--phase")
            renderer->phase = std::stof(argv[++i]);
        else if (arg == "--env_strength")
            renderer->environment->strength = std::stof(argv[++i]);
        else if (arg == "--env_rot")
            renderer->environment->transform = glm::mat3(glm::rotate(glm::mat4(renderer->environment->transform), glm::radians(std::stof(argv[++i])), glm::vec3(0, 1, 0)));
        else if (arg == "--env_hide")
            renderer->show_environment = false;
        else if (arg == "--heatmap")
            renderer->transferfunc = std::make_shared<TransferFunction>(tinycolormap::ColormapType::Turbo);
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
        else if (arg == "--vol_rot_x")
            renderer->volume->transform = glm::mat3(glm::rotate(glm::mat4(renderer->volume->transform), glm::radians(std::stof(argv[++i])), glm::vec3(1, 0, 0)));
        else if (arg == "--vol_rot_y")
            renderer->volume->transform = glm::mat3(glm::rotate(glm::mat4(renderer->volume->transform), glm::radians(std::stof(argv[++i])), glm::vec3(0, 1, 0)));
        else if (arg == "--vol_rot_z")
            renderer->volume->transform = glm::mat3(glm::rotate(glm::mat4(renderer->volume->transform), glm::radians(std::stof(argv[++i])), glm::vec3(0, 0, 1)));
        else if (fs::is_regular_file(argv[i]) || fs::is_directory(argv[i]))
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
                if (renderer->sample == renderer->sppx)
                    renderer->color->save_ldr(out_filename, true, true); // TODO: apply tonemapping?
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
                Context::swap_buffers(); // sync (this is required for >= 1024spp)
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
            std::string out_fn = fs::path(out_filename).stem().string() + "_" + std::string(n_zero - std::min(n_zero, std::to_string(i).length()), '0') + std::to_string(i) + ".png";
            renderer->color->save_ldr(out_fn);
            std::cout << out_fn << " written." << std::endl;
            Context::swap_buffers();
        }
    }
}
