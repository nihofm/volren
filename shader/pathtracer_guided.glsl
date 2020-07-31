#version 450 core

#include "common.h"

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color;
layout (binding = 1, rgba32f) uniform image2D even;

/*
// RIS/WRS
// decide on target pdf / use-case?
float target_pdf(const vec3 y, const vec3 dir) {
    return phase_henyey_greenstein(dot(-dir, y), vol_phase_g) * luma(env_strength * environment_lookup(vol_to_world(y)));
}
float stream_ris(inout Reservoir r, const int M, const vec3 dir, inout uint seed) {
    for (int i = 0; i < M; ++i) {
        // generate sample (isotropic)
        const vec3 xi = sample_phase_isotropic(rng2(seed));
        const float wi = phase_isotropic();
        // update reservoir
        wrs_update(r, xi, wi, target_pdf(xi, dir), seed);
    }
    // apply RIS (ReSTIR, eq. 6)
    return r.w_sum / (float(r.y_pt.w) * r.M);
}
float stream_ris_single(inout Reservoir r, const vec3 xi, const float wi, const float pi, inout uint seed) {
    wrs_update(r, xi, wi, pi, seed);
    return r.w_sum / (float(r.y_pt.w) * r.M);
}
// load and clear reservoir?
Reservoir r = reservoirs[pixel.y * size.x + pixel.x];
if (current_sample == 0) wrs_init(reservoirs[pixel.y * size.x + pixel.x]);
// RIS/WRS
const float ris_w = stream_ris(reservoirs[pixel.y * size.x + pixel.x], 32, dir, seed);
const vec3 scatter_dir = reservoirs[pixel.y * size.x + pixel.x].y;
f_p = phase_henyey_greenstein(dot(-dir, scatter_dir), vol_phase_g);
dir = scatter_dir;
throughput *= f_p * ris_w;
*/

// ---------------------------------------------------
// path tracing

uniform int current_sample;
uniform int bounces;
uniform int show_environment;

vec3 trace_path(in vec3 pos, in vec3 dir, inout uint seed) {
    // trace path
    vec3 radiance = vec3(0), throughput = vec3(1), debug = vec3(0);
    uint n_paths = 0, ridx = 0;
    bool free_path = true;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    while (sample_volume(pos, dir, seed, t, throughput)) {
        // advance ray
        pos = pos + t * dir;

        // sample light source (environment)
        const float pt_prev = max(0.f, luma(radiance));
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(seed), w_i);
        if (Li_pdf.w > 0) {
            const vec3 to_light = world_to_vol(w_i);
            f_p = phase_henyey_greenstein(dot(-dir, to_light), vol_phase_g);
            const float weight = power_heuristic(Li_pdf.w, f_p);
            radiance += throughput * weight * f_p * transmittance(pos, to_light, seed) * env_strength * Li_pdf.rgb / Li_pdf.w;
        }

        // update last reservoir?
        const float pt = luma(radiance) - pt_prev;
        if (n_paths > 0 && pt > 0)
            wrs_update_atomic(ridx, dir, f_p, pt, seed);
        // update reservoir index
        const ivec3 idx = ivec3(pos * vol_size);
        ridx = (idx.z * vol_size.y + idx.y) * vol_size.x + idx.x;
        // debug output
        if (n_paths == 0) debug = max(vec3(0), reservoirs[ridx].y_pt.rgb * .5 + .5);
        //if (n_paths == 0) debug = vec3(reservoirs[ridx].M / float(0xFFFFFFFF));

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

        // scatter ray (guided?)
        if (false && reservoirs[ridx].M > 1) {
            const vec3 scatter_dir = reservoirs[ridx].y_pt.xyz;
            f_p = phase_henyey_greenstein(dot(-dir, scatter_dir), vol_phase_g);
            throughput *= f_p * reservoirs[ridx].weight();
            dir = scatter_dir;
        } else {
            const vec3 scatter_dir = sample_phase_henyey_greenstein(dir, vol_phase_g, rng2(seed));
            f_p = phase_henyey_greenstein(dot(-dir, scatter_dir), vol_phase_g);
            dir = scatter_dir;
            // debug output
            //if (n_paths == 1) debug = max(vec3(0), scatter_dir.rgb * .5 + .5);
        }
    }

    // free path? -> add envmap contribution
    if (free_path && n_paths >= show_environment) {
        dir = vol_to_world(dir);
        const vec3 Le = environment_lookup(dir);
        const float weight = n_paths > 0 ? power_heuristic(f_p, pdf_environment(Le, dir)) : 1.f;
        radiance += throughput * weight * (n_paths == 0 ? 1.f : env_strength) * Le;
    }

    // TODO visualize / test / integrate
    if (n_paths > 0) return debug;

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
        imageStore(even, pixel, vec4(mix(imageLoad(even, pixel).rgb, radiance, 1.f / ((current_sample + 1) / 2)), 1));
}
