#pragma once

#include <glm/glm.hpp>
#include <cppgl.h>
#include <voldata.h>
#include "environment.h"
#include "transferfunc.h"

struct Renderer {
    // interface
    static void init(uint32_t w = 1920, uint32_t h = 1080, bool vsync = false, bool pinned = false, bool visible = true);
    static void commit();
    static void trace();
    static void draw();

    // Settings
    static int sample;
    static int sppx;
    static int bounces;
    static float tonemap_exposure;
    static float tonemap_gamma;
    static bool tonemapping;
    static bool show_convergence;
    static bool show_environment;

    // Scene data
    static Environment environment;
    static TransferFunction transferfunc;
    static std::shared_ptr<voldata::Volume> volume;
    static glm::vec3 vol_crop_min, vol_crop_max;

    // OpenGL data
    static Framebuffer fbo;
    static Shader trace_shader;
    static Texture3D vol_dense;
    static Texture3D vol_indirection, vol_range, vol_atlas;
};
