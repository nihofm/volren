#include <cppgl/context.h>
#include <cppgl/quad.h>
#include <cppgl/camera.h>
#include <cppgl/shader.h>
#include <cppgl/framebuffer.h>
#include <cppgl/imgui/imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>

#include "volume.h"

// ------------------------------------------
// state / variables / settings

static int sample = 0;
static int sppx = 1000;
static int bounces = 1;
static bool show_environment = true;
static bool tonemapping = true;
static float exposure = 3.f;
static std::shared_ptr<Volume> volume;
static std::shared_ptr<Shader> trace_shader;
static std::shared_ptr<Texture2D> environment_tex;
static std::shared_ptr<Framebuffer> fbo;

// ------------------------------------------
// helper funcs and callbacks

void blit(const std::shared_ptr<Texture2D>& tex) {
    static std::shared_ptr<Shader> blit_shader = make_shader("blit", "shader/blit.vs", "shader/blit.fs");
    blit_shader->bind();
    blit_shader->uniform("tex", tex, 0);
    Quad::draw();
    blit_shader->unbind();
}

void tonemap(const std::shared_ptr<Texture2D>& tex) {
    static std::shared_ptr<Shader> tonemap_shader = make_shader("tonemap", "shader/blit.vs", "shader/tonemap.fs");
    tonemap_shader->bind();
    tonemap_shader->uniform("tex", tex, 0);
    tonemap_shader->uniform("exposure", exposure);
    Quad::draw();
    tonemap_shader->unbind();
}

void resize_callback(int w, int h) {
    fbo->resize(w, h);
    sample = 0; // restart rendering
}

void keyboard_callback(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        sample = 0;
    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        Context::screenshot("screenshot.png");
}

void mouse_button_callback(int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
}

// ------------------------------------------
// main

int main(int argc, char** argv) {
    // setup GL
    ContextParameters params;
    params.title = "VolGL";
    params.floating = GLFW_TRUE;
    params.resizable = GLFW_FALSE;
    params.swap_interval = 0;
    Context::init(params);
    Context::set_resize_callback(resize_callback);
    Context::set_keyboard_callback(keyboard_callback);
    Context::set_mouse_button_callback(mouse_button_callback);

    // setup fbo
    const glm::ivec2 res = Context::resolution();
    fbo = make_framebuffer("fbo", res.x, res.y);
    fbo->attach_depthbuffer(make_texture2D("fbo/depth", res.x, res.y, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT));
    fbo->attach_colorbuffer(make_texture2D("fbo/col", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(make_texture2D("fbo/f_pos", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(make_texture2D("fbo/f_norm", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(make_texture2D("fbo/f_alb", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(make_texture2D("fbo/f_vol", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->check();

    // setup trace shader and environment map (i.e. light source)
    trace_shader = make_shader("trace", "shader/trace.glsl");
    //environment_tex = make_texture2D("env_sky", "data/images/envmaps/day_clear.png"); // TODO args
    environment_tex = make_texture2D("env_sky", "data/gi/envmaps/clearsky.hdr"); // TODO args
    //environment_tex = make_texture2D("env_woods", "data/images/envmaps/woods.hdr"); // TODO args

    // setup volume
    std::ifstream bunny_raw("data/volumetric/bunny_512x512x361_uint16.raw", std::ios::binary);
    if (false && bunny_raw.is_open()) {
        // TODO bunny (https://klacansky.com/open-scivis-datasets/)
        std::vector<uint16_t> raw(std::istreambuf_iterator<char>(bunny_raw), {});
        volume = make_volume("bunny", 512, 512, 361, raw.data());
        volume->model[0][0] = 1.f / 0.337891;
        volume->model[1][1] = 1.f / 0.337891;
        volume->model[2][2] = 1.f / 0.5;
    } else {
        // simple cube
        uint32_t N = 128;
        std::vector<float> density(N * N * N);
        for (uint32_t d = 0; d < N; ++d)
            for (uint32_t y = 0; y < N; ++y)
                for (uint32_t x = 0; x < N; ++x)
                    //density[d * N * N + y * N + x] = y / float(N);
                    density[d * N * N + y * N + x] = glm::clamp(sqrtf(100.f / glm::distance(glm::vec3(x, y, d), glm::vec3(N * .5 + 1e-3))), 0.f, 100.f);
        volume = make_volume("volume", N, N, N, density.data());
        volume->model = glm::rotate(volume->model, float(M_PI / 3), glm::vec3(1, 0, 1));
        volume->model = glm::scale(volume->model, glm::vec3(5));
        volume->model[3][0] = 10;
        volume->model[3][1] = 2.5;
        volume->model[3][2] = -2.5;
    }

    // run
    while (Context::running()) {
        // handle input
        glfwPollEvents();
        if (Camera::default_input_handler(Context::frame_time()))
            sample = 0; // restart rendering
        static uint32_t counter = 0;
        if (counter++ % 100 == 0)
            Shader::reload_modified();

        // update
        Camera::current()->update();

        // render
        if (sample < sppx) {
            // bind
            trace_shader->bind();
            for (uint32_t i = 0; i < fbo->color_textures.size(); ++i)
                fbo->color_textures[i]->bind_image(i, GL_READ_WRITE, GL_RGBA32F);

            // uniforms
            trace_shader->uniform("current_sample", ++sample);
            trace_shader->uniform("bounces", bounces);
            trace_shader->uniform("show_environment", show_environment ? 1 : 0);
            trace_shader->uniform("model", volume->model);
            trace_shader->uniform("inv_model", glm::inverse(volume->model));
            trace_shader->uniform("volume_tex", volume->texture, 0);
            trace_shader->uniform("inv_max_density", 1.f / volume->max_density);
            trace_shader->uniform("absorbtion_coefficient", volume->absorbtion_coefficient);
            trace_shader->uniform("scattering_coefficient", volume->scattering_coefficient);
            //trace_shader->uniform("emission", emission);
            trace_shader->uniform("cam_pos", Camera::current()->pos);
            trace_shader->uniform("cam_fov", Camera::current()->fov_degree);
            const glm::vec3 right = glm::normalize(cross(Camera::current()->dir, glm::vec3(1e-4f, 1, 0)));
            const glm::vec3 up = glm::normalize(cross(right, Camera::current()->dir));
            trace_shader->uniform("cam_transform", glm::mat3(right, up, -Camera::current()->dir));
            trace_shader->uniform("environment_tex", environment_tex, 1);

            // trace
            const glm::ivec2 size = Context::resolution();
            trace_shader->dispatch_compute(size.x, size.y);

            // unbind
            for (uint32_t i = 0; i < fbo->color_textures.size(); ++i)
                fbo->color_textures[i]->unbind_image(i);
            trace_shader->unbind();
        } else
            glfwWaitEvents();

        // draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (tonemapping)
            tonemap(fbo->color_textures[0]);
        else
            blit(fbo->color_textures[0]);

        // draw GUI
        const glm::ivec2 size = Context::resolution();
        ImGui::SetNextWindowPos(ImVec2(size.x-260, 20));
        ImGui::SetNextWindowSize(ImVec2(250, 300));
        if (ImGui::Begin("Stuff", 0, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground)) {
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, .9f);
            ImGui::Text("Sample: %i/%i", sample, sppx);
            if (ImGui::InputInt("Sppx", &sppx)) sample = 0;
            if (ImGui::SliderInt("Bounces", &bounces, 1, 100)) sample = 0;
            if (ImGui::Checkbox("Show environment", &show_environment)) sample = 0;
            if (ImGui::SliderFloat("Absorb", &volume->absorbtion_coefficient, 0.001f, 1.f)) sample = 0;
            if (ImGui::SliderFloat("Scatter", &volume->scattering_coefficient, 0.001f, 1.f)) sample = 0;
            //if (ImGui::ColorEdit3("Emission", &emission.x)) sample = 0;
            ImGui::Separator();
            ImGui::Checkbox("Tonemapping", &tonemapping);
            ImGui::SliderFloat("Exposure", &exposure, 0.1f, 25.f);
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
            ImGui::PopStyleVar();
            ImGui::End();
        }

        // finish frame
        Context::swap_buffers();
    }
}
