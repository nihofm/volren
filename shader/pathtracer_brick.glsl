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

vec3 sanitize(const vec3 data) { return mix(data, vec3(0), isnan(data) || isinf(data)); }

vec3 trace_path(inout ray_state ray) {
    // trace path
    vec3 radiance = vec3(0), throughput = vec3(1);
    bool free_path = true;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    while (sample_volumeDDA(ray.pos, ray.dir, t, throughput, ray.seed)) {
        // advance ray
        ray.pos = ray.pos + t * ray.dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(ray.seed), w_i);
        if (Li_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-ray.dir, w_i), vol_phase_g);
            const float weight = power_heuristic(Li_pdf.w, f_p);
            const float Tr = transmittanceDDA(ray.pos, w_i, ray.seed);
            radiance += throughput * weight * f_p * Tr * Li_pdf.rgb / Li_pdf.w;
        }

        // early out?
        if (++ray.n_paths >= bounces) { free_path = false; break; }
        // russian roulette
        const float rr_val = luma(throughput);
        if (rr_val < .1f) {
            const float prob = 1 - rr_val;
            if (rng(ray.seed) < prob) { free_path = false; break; }
            throughput /= 1 - prob;
        }

        // scatter ray
        const vec3 scatter_dir = sample_phase_henyey_greenstein(ray.dir, vol_phase_g, rng2(ray.seed));
        f_p = phase_henyey_greenstein(dot(-ray.dir, scatter_dir), vol_phase_g);
        ray.dir = scatter_dir;
    }

    // free path? -> add envmap contribution
    if (free_path && ray.n_paths >= show_environment) {
        const vec3 Le = lookup_environment(ray.dir);
        const float weight = ray.n_paths > 0 ? power_heuristic(f_p, pdf_environment(ray.dir)) : 1.f;
        radiance += throughput * weight * Le;
    }

    return radiance;
}

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 size = imageSize(color);
	if (any(greaterThanEqual(pixel, size))) return;

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * size.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir(pixel, size, rng2(seed));

    // trace ray
    ray_state ray = { pos, 0.f, dir, 1e+38f, pixel, seed, 0 };
    const vec3 radiance = trace_path(ray);
    // const vec3 radiance = vec3(transmittance_raymarch(ray.pos, ray.dir, seed)); // TODO debug

    // write results
    imageStore(color, pixel, vec4(mix(imageLoad(color, pixel).rgb, sanitize(radiance), 1.f / current_sample), 1));
}
