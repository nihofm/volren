#version 450 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color;

// ---------------------------------------------------
// settings

#define USE_DDA
#include "common.glsl"

// ---------------------------------------------------
// uniforms

uniform int current_sample;
uniform int seed;
uniform ivec2 resolution;

// ---------------------------------------------------
// Quilt stuff

// parameters (https://docs.lookingglassfactory.com/keyconcepts/quilts)
const uint quiltcolumns = 8;
const uint quiltrows = 6;
const uint totalViews = quiltcolumns * quiltrows;
const uint framebufferwidth = resolution.x;
const uint framebufferheight = resolution.y;
const float fov = 14 * M_PI / 180.f;
const float viewCone = 35 * M_PI / 180.f;
// const float cameraSize = 1; // TODO: wtf is this?
const float cameraDistance = 0.5;//-cameraSize / tan(fov / 2.0f);
const bool singleRenderCall = true;
const uint quiltindex = 0;
const vec3 cameraU = cam_transform[0];
const vec3 cameraV = cam_transform[1];
const vec3 cameraW = -cam_transform[2];

void view_ray_quilt(const vec2 pixel, const vec2 screen, const vec2 pixel_sample, out vec3 origin, out vec3 direction) {
    const vec2 fragment = pixel + pixel_sample;              // Jitter the sub-pixel location
    const vec2 ndc      = (fragment / screen) * 2.0f - 1.0f; // Normalized device coordinates in range [-1, 1].
    // render into quilt
    const float viewwidth = min(screen.x, framebufferwidth) / quiltcolumns;
    const float indexx = floor(pixel.x / viewwidth);
    const float viewheight = min(screen.y, framebufferheight) / quiltrows; 
    const float indexy = floor(pixel.y / viewheight);
    const int index = min(max(int(indexy * quiltcolumns + indexx),0), int(totalViews-1));
    const int movex = int(indexx > (quiltcolumns / 2.0 - 1) ? (-(indexx - quiltcolumns / 2.0 + 1) * 2 + (mod(float(quiltcolumns), 2.0) !=0 ? 2 : 1)) : (abs(indexx-quiltcolumns / 2.0 + 1) * 2 + (mod(float(quiltcolumns), 2.0) !=0 ? 2 : 1)));
    const int movey = int(indexy > (quiltrows / 2.0 - 1) ? (-(indexy - quiltrows / 2.0 + 1)*2 + (mod(float(quiltrows), 2.0) != 0 ? 2 : 1)) : (abs(indexy-quiltrows / 2.0 + 1) * 2 + (mod(float(quiltrows), 2.0) !=0 ? 2 : 1)));
    const float offsetAngle = ((!singleRenderCall? quiltindex : index) / (totalViews - 1.0f) - 0.5f) * viewCone;  
    const float offset =  cameraDistance * tan(offsetAngle);
    origin = cam_pos + cameraU * offset;
    direction = normalize(cameraU * (!singleRenderCall? ndc.x : ndc.x*quiltcolumns + movex) + cameraV * (!singleRenderCall? ndc.y : ndc.y*quiltrows + movey) + cameraW - cameraU * tan(offsetAngle));
}

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, resolution))) return;

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * resolution.x + pixel.x), current_sample, 32);
    vec3 pos, dir;
    view_ray_quilt(pixel, resolution, rng2(seed), pos, dir);

    // trace ray
    const vec4 L = trace_path(pos, dir, seed);

    // write result
    imageStore(color, pixel, mix(imageLoad(color, pixel), sanitize(L), 1.f / current_sample));
}
