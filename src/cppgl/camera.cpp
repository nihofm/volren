#include "camera.h"
#include "context.h"
#include "imgui/imgui.h"
#include <iostream>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_transform.hpp>

static Camera* current_camera = 0;
float Camera::default_camera_movement_speed = 0.005;

Camera::Camera(const std::string& name) : NamedMap(name), pos(0, 1, 0), dir(1, 0, 0), up(0, 1, 0),
    fov_degree(70), near(0.01), far(1000), left(-100), right(100), bottom(-100), top(100),
    perspective(true), fix_up_vector(true) {
    update();
}

Camera::~Camera() {
    if (current_camera == this) // reset current ptr
        current_camera = 0;
}

Camera* Camera::current() {
    static Camera default_cam("default");
    return current_camera ? current_camera : &default_cam;
}

void Camera::make_current() {
    current_camera = this;
}

void Camera::update() {
    dir = glm::normalize(dir);
    up = glm::normalize(up);
    view = glm::lookAt(pos, pos + dir, up);
    view_normal = glm::transpose(glm::inverse(view));
    proj = perspective ? glm::perspective(fov_degree * float(M_PI / 180), aspect_ratio(), near, far) : glm::ortho(left, right, bottom, top, near, far);
}

void Camera::forward(float by) { pos += by * dir; }
void Camera::backward(float by) { pos -= by * dir; }
void Camera::leftward(float by) { pos -= by * cross(dir, up); }
void Camera::rightward(float by) { pos += by * cross(dir, up); }
void Camera::upward(float by) { pos += by * glm::normalize(cross(cross(dir, up), dir)); }
void Camera::downward(float by) { pos -= by * glm::normalize(cross(cross(dir, up), dir)); }

void Camera::yaw(float angle) { dir = glm::normalize(glm::rotate(dir, angle * float(M_PI) / 180.f, up)); }
void Camera::pitch(float angle) {
    dir = glm::normalize(glm::rotate(dir, angle * float(M_PI) / 180.f, normalize(cross(dir, up))));
    if (not fix_up_vector) up = glm::normalize(glm::cross(glm::cross(dir, up), dir));
}
void Camera::roll(float angle) { up = glm::normalize(glm::rotate(up, angle * float(M_PI) / 180.f, dir)); }

float Camera::aspect_ratio() {
    GLint xywh[4];
    glGetIntegerv(GL_VIEWPORT, xywh);
    return xywh[2] / (float)xywh[3];
}

void Camera::default_input_handler(double dt_ms) {
    if (not ImGui::GetIO().WantCaptureKeyboard) {
        // keyboard
        if (Context::key_pressed(GLFW_KEY_W))
            current()->forward(dt_ms * default_camera_movement_speed);
        if (Context::key_pressed(GLFW_KEY_S))
            current()->backward(dt_ms * default_camera_movement_speed);
        if (Context::key_pressed(GLFW_KEY_A))
            current()->leftward(dt_ms * default_camera_movement_speed);
        if (Context::key_pressed(GLFW_KEY_D))
            current()->rightward(dt_ms * default_camera_movement_speed);
        if (Context::key_pressed(GLFW_KEY_R))
            current()->upward(dt_ms * default_camera_movement_speed);
        if (Context::key_pressed(GLFW_KEY_F))
            current()->downward(dt_ms * default_camera_movement_speed);
        if (Context::key_pressed(GLFW_KEY_Q))
            Camera::current()->roll(dt_ms * -0.1);
        if (Context::key_pressed(GLFW_KEY_E))
            Camera::current()->roll(dt_ms * 0.1);
    }
    // mouse
    static float rot_speed = 0.003;
    static glm::vec2 last_pos(-1);
    const glm::vec2 curr_pos = Context::mouse_pos();
    if (last_pos == glm::vec2(-1)) last_pos = curr_pos;
    const glm::vec2 diff = last_pos - curr_pos;
    if (not ImGui::GetIO().WantCaptureMouse && Context::mouse_button_pressed(GLFW_MOUSE_BUTTON_LEFT)) {
        current()->pitch(dt_ms * diff.y * rot_speed);
        current()->yaw(dt_ms * diff.x * rot_speed);
    }
    last_pos = curr_pos;
}
