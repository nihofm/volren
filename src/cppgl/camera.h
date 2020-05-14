#pragma once

#include <string>
#include <memory>
#include <glm/glm.hpp>
#include "named_map.h"

class Camera : public NamedMap<Camera> {
public:
    Camera(const std::string& name);
    virtual ~Camera();

    // current camera pointer (raw, non-ownership) used for rendering
    static Camera* current();
    void make_current();

    void update();

    // move
    void forward(float by);
    void backward(float by);
    void leftward(float by);
    void rightward(float by);
    void upward(float by);
    void downward(float by);

    // rotate
    void yaw(float angle);
    void pitch(float angle);
    void roll(float angle);

    // compute aspect ratio from current viewport
    static float aspect_ratio();

    // data
    glm::vec3 pos, dir, up;             // camera coordinate system
    float fov_degree, near, far;        // perspective projection
    float left, right, bottom, top;     // orthographic projection
    bool perspective;                   // switch between perspective and orthographic (default: perspective)
    bool fix_up_vector;                 // keep up vector fixed to avoid camera drift
    glm::mat4 view, view_normal, proj;  // camera matrices (computed via a call update())

    // default camera keyboard/mouse handler for basic movement
    static float default_camera_movement_speed;
    static void default_input_handler(double dt_ms);
};

// variadic alias for std::make_shared<>(...)
template <class... Args> std::shared_ptr<Camera> make_camera(Args&&... args) {
    return std::make_shared<Camera>(args...);
}
