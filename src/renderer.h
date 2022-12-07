#pragma once

#include <voldata/voldata.h>

struct Renderer {
    // Renderer interface
    virtual void init() = 0;                            // initialize renderer (call once upon initialization)
    virtual void resize(uint32_t w, uint32_t h) = 0;    // resize internal buffers
    virtual void commit() = 0;                          // commit and upload internal data structures (call after changing the scene)
    virtual void trace() = 0;                           // trace one sample per pixel
    virtual void draw() = 0;                            // draw result on screen

    virtual void reset() { sample = 0; }                // restart rendering

    // Volume data
    std::shared_ptr<voldata::Volume> volume;

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