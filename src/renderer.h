#pragma once

#include "voldata.h"

struct Renderer {
    // Renderer interface
    virtual void init() = 0;                            // initialize renderer (call once upon initialization)
    virtual void resize(uint32_t w, uint32_t h) = 0;    // resize internal buffers
    virtual void commit() = 0;                          // commit and upload internal data structures (call after changing the scene)
    virtual void trace() = 0;                           // trace one sample per pixel
    virtual void draw() = 0;                            // draw result on screen

    // Camera data TODO actually use this
    /*
    glm::vec3 cam_pos = glm::vec3(0, 0, 0);
    glm::vec3 cam_dir = glm::vec3(1, 0, 0);
    glm::vec3 cam_up = glm::vec3(0, 1, 0);
    float cam_fov = 70.f;
    */

    // Volume data
    std::shared_ptr<voldata::Volume> volume;
    inline void set_volume(const std::shared_ptr<voldata::Volume>& vol) { volume = vol; }

    // Volume clip planes
    glm::vec3 vol_clip_min = glm::vec3(0.f);
    glm::vec3 vol_clip_max = glm::vec3(1.f);

    // Settings
    int sample = 0;
    int sppx = 1024;
    int seed = 42;
    int bounces = 3;
    float tonemap_exposure = 5.f;
    float tonemap_gamma = 2.2f;
    bool tonemapping = true;
    bool show_environment = true;
};