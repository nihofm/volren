#include <cppgl/context.h>
#include <cppgl/quad.h>
#include <cppgl/camera.h>
#include <cppgl/shader.h>
#include <cppgl/framebuffer.h>
#include <cppgl/imgui/imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void blit(const std::shared_ptr<Texture2D>& tex) {
    static std::shared_ptr<Shader> blit_shader = make_shader("blit", "shader/blit.vs", "shader/blit.fs");
    blit_shader->bind();
    blit_shader->uniform("tex", tex, 0);
    Quad::draw();
    blit_shader->unbind();
}

static std::shared_ptr<Framebuffer> fbo;

void resize_callback(int w, int h) {
    fbo->resize(w, h);
}

void keyboard_callback(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
}

void mouse_button_callback(int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
}

int main(int argc, char** argv) {
    ContextParameters params;
    params.title = "VolGL";
    Context::init(params);
    Context::set_resize_callback(resize_callback);
    Context::set_keyboard_callback(keyboard_callback);
    Context::set_mouse_button_callback(mouse_button_callback);

    const glm::ivec2 res = Context::resolution();
    fbo = make_framebuffer("fbo", res.x, res.y);
    fbo->attach_depthbuffer(make_texture2D("fbo/depth", res.x, res.y, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT));
    fbo->attach_colorbuffer(make_texture2D("fbo/col", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(make_texture2D("fbo/f_pos", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(make_texture2D("fbo/f_norm", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(make_texture2D("fbo/f_alb", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->attach_colorbuffer(make_texture2D("fbo/f_vol", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT));
    fbo->check();

    auto environment = make_texture2D("environment", "data/images/envmaps/day_clear.png");
    auto trace_shader = make_shader("trace", "shader/trace.glsl");

    // TODO volume data
    glm::mat4 model(1);
    model = glm::rotate(model, float(M_PI / 3), glm::vec3(1, 0, 0));
    model = glm::rotate(model, float(M_PI / 3), glm::vec3(0, 0, 1));

    model = glm::scale(model, glm::vec3(3));

    model[3][0] = 1;
    model[3][1] = 1;

    uint32_t N = 128;
    std::vector<float> vol_data(N * N * N);
    for (uint32_t d = 0; d < N; ++d)
        for (uint32_t y = 0; y < N; ++y)
            for (uint32_t x = 0; x < N; ++x)
                vol_data[d * N * N + y * N + x] = 10.f / glm::distance(glm::vec3(x, y, d), glm::vec3(N*.5+1e-3));
    auto vol_tex = make_texture3D("volume", 128, 128, 128, GL_R32F, GL_RED, GL_FLOAT, vol_data.data());

    int sample = 0, sppx = 1000;
    while (Context::running()) {
        // handle input
        if (Camera::default_input_handler(Context::frame_time()))
            sample = 0;
        static uint32_t counter = 0;
        if (counter++ % 100 == 0) Shader::reload_modified();

        // update
        Camera::current()->update();

        if (sample++ < sppx) {
            // bind
            trace_shader->bind();
            for (uint32_t i = 0; i < fbo->color_textures.size(); ++i)
                fbo->color_textures[i]->bind_image(i, GL_READ_WRITE, GL_RGBA32F);

            // uniforms
            trace_shader->uniform("current_sample", sample);
            trace_shader->uniform("model", model);
            trace_shader->uniform("inv_model", glm::inverse(model));
            trace_shader->uniform("volume", vol_tex, 0);
            trace_shader->uniform("cam_pos", Camera::current()->pos);
            trace_shader->uniform("cam_fov", Camera::current()->fov_degree);
            const glm::vec3 right = glm::normalize(cross(Camera::current()->dir, glm::vec3(1e-4f, 1, 0)));
            const glm::vec3 up = glm::normalize(cross(right, Camera::current()->dir));
            trace_shader->uniform("cam_transform", glm::mat3(right, up, -Camera::current()->dir));
            trace_shader->uniform("env_tex", environment, 1);

            // trace
            const glm::ivec2 size = Context::resolution();
            trace_shader->dispatch_compute(size.x, size.y);

            // unbind
            for (uint32_t i = 0; i < fbo->color_textures.size(); ++i)
                fbo->color_textures[i]->unbind_image(i);
            trace_shader->unbind();
        }

        // draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        blit(fbo->color_textures[0]);

        // finish frame
        Context::swap_buffers();
    }
}
