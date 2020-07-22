#version 450 core

#include "common.h"

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color;
layout (binding = 1, rgba32f) uniform image2D even;

// ---------------------------------------------------
// path tracing

uniform int current_sample;
uniform int bounces;
uniform int show_environment;

vec3 trace_path(in vec3 pos, in vec3 dir, inout uint seed) {
    // trace path
    vec3 radiance = vec3(0), throughput = vec3(1);
    int n_paths = 0;
    bool free_path = true;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    while (sample_volume(pos, dir, seed, t, throughput)) {
        // advance ray
        pos = pos + t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(seed), w_i);
        if (Li_pdf.w > 0) {
            const vec3 to_light = world_to_vol(w_i);
            f_p = phase_henyey_greenstein(dot(-dir, to_light), vol_phase_g);
            const float weight = power_heuristic(Li_pdf.w, f_p);
            radiance += throughput * weight * f_p * transmittance(pos, to_light, seed) * env_strength * Li_pdf.rgb / Li_pdf.w;
        }

        // early out?
        if (++n_paths >= bounces) { free_path = false; break; }
        // russian roulette
        const float rr_threshold = .1f;
        const float rr_val = luma(throughput);
        if (rr_val < rr_threshold) {
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
    if (free_path && n_paths >= show_environment) {
        dir = vol_to_world(dir);
        const vec3 Le = environment_lookup(dir);
        const float weight = n_paths > 0 ? power_heuristic(f_p, pdf_environment(Le, dir)) : 1.f;
        radiance += throughput * weight * env_strength * Le;
    }

    return radiance;
}

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 size = imageSize(color);
	if (any(greaterThanEqual(pixel, size))) return;

    // setup random seed and camera ray (in model space!)
    uint seed = tea(pixel.y * size.x + pixel.x, current_sample, 8);
    vec3 pos = world_to_vol(vec4(cam_pos, 1));
    vec3 dir = world_to_vol(view_dir(pixel, size, rng2(seed)));

    // trace ray
    const vec3 radiance = trace_path(pos, dir, seed);

    // write output
    if (any(isnan(radiance)) || any(isinf(radiance))) return;
    imageStore(color, pixel, vec4(mix(imageLoad(color, pixel).rgb, radiance, 1.f / current_sample), 1));
    if (current_sample % 2 == 1)
        imageStore(even, pixel, vec4(mix(imageLoad(even, pixel).rgb, radiance, 1.f / ((current_sample+ 1) / 2)), 1));
}
