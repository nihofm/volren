#version 450 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color;

#include "common.glsl"

// ---------------------------------------------------
// path tracing

uniform int current_sample;
uniform int bounces;
uniform int seed;
uniform int show_environment;
uniform ivec2 resolution;

#define USE_DDA

vec4 trace_path(vec3 pos, vec3 dir, inout uint seed) {
    // trace path
    vec3 L = vec3(0);
    vec3 throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
#ifdef USE_DDA
    while (sample_volumeDDA(pos, dir, t, throughput, L, seed)) {
#else
    while (sample_volume(pos, dir, t, throughput, L, seed)) {
#endif
        // advance ray
        pos = pos + t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
#ifdef USE_DDA
            const float Tr = transmittanceDDA(pos, w_i, seed);
#else
            const float Tr = transmittance(pos, w_i, seed);
#endif
            L += throughput * mis_weight * f_p * Tr * Le_pdf.rgb / Le_pdf.w;
        }

        // early out?
        if (++n_paths >= bounces) { free_path = false; break; }
        // russian roulette
        const float rr_val = luma(throughput);
        if (rr_val < .1f) {
            const float prob = 1 - rr_val;
            if (rng(seed) < prob) { free_path = false; break; }
            throughput /= 1 - prob;
        }

        // scatter ray
        const vec3 scatter_dir = sample_phase_henyey_greenstein(dir, vol_phase_g, rng2(seed));
        f_p = phase_henyey_greenstein(dot(-dir, scatter_dir), vol_phase_g);
        dir = scatter_dir;
    }

    // free path? -> add envmap contribution
    if (free_path && show_environment > 0) {
        const vec3 Le = lookup_environment(dir);
        const float mis_weight = n_paths > 0 ? power_heuristic(f_p, pdf_environment(dir)) : 1.f;
        L += throughput * mis_weight * Le;
    }

    return vec4(L, clamp(n_paths, 0.f, 1.f));
}


// ---------------------------------------------------
// quilt helper

// Quilt parameters (https://docs.lookingglassfactory.com/keyconcepts/quilts)
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
