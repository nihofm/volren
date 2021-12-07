#version 450 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color;

#include "common.glsl"

// ---------------------------------------------------
// path tracing

uniform int bounces;
uniform int seed;
uniform int show_environment;
uniform int current_sample;
uniform ivec2 resolution;

vec3 direct_volume_rendering_irradiance_cache(vec3 pos, vec3 dir, inout uint seed) {
    vec3 L = vec3(0);
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return show_environment > 0 ? lookup_environment(dir) : vec3(0);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(pos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(dir, 0)); // non-normalized!
    const float dt = (near_far.y - near_far.x) / float(RAYMARCH_STEPS);
    // jitter starting position
    const float jitter = rng(seed) * dt;
    near_far.x += jitter;
    float Tr = exp(-lookup_density(ipos + near_far.x * idir, seed) * jitter);
    // ray marching
    for (int i = 0; i < RAYMARCH_STEPS; ++i) {
        const vec3 curr = ipos + min(near_far.x + i * dt, near_far.y) * idir;
#ifdef USE_TRANSFERFUNC
        const float d_real = lookup_density(curr, seed);
        const vec4 rgba = tf_lookup(d_real * vol_inv_majorant);
        const float d = rgba.a * vol_majorant;
        const vec3 Le = vol_albedo * rgba.rgb * irradiance_query(curr, seed);
#else
        const float d = lookup_density(curr, seed);
        const vec3 Le = vol_albedo * irradiance_query(curr, seed);
#endif
        const float dtau = d * dt;
        // accum emission from irradiance cache with geom avg of transmittance along segment
        L += Le * dtau * Tr * exp(-dtau * 0.5);
        // update transmittance
        Tr *= exp(-dtau);
        if (Tr <= 1e-5) break;//return L;
    }
    if (show_environment > 0) L += lookup_environment(dir) * Tr;
    return L;
}

vec3 trace_path(vec3 pos, vec3 dir, inout uint seed) {
    // trace path
    vec3 L = vec3(0), throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t; // t: end of ray segment (i.e. sampled position or out of volume)
    while (sample_volumeDDA(pos, dir, t, throughput, L, seed)) {
        // advance ray
        pos = pos + t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            const float f_p = phase_isotropic();
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittanceDDA(pos, w_i, seed);
            L += throughput * mis_weight * f_p * Tr * Le_pdf.rgb / Le_pdf.w;
        }

        // early out?
        if (++n_paths >= max(0, bounces-1)) { free_path = false; break; }
        // russian roulette
        const float rr_val = luma(throughput);
        if (rr_val < .1f) {
            const float prob = 1 - rr_val;
            if (rng(seed) < prob) { free_path = false; break; }
            throughput /= 1 - prob;
        }

        // scatter ray
        const vec3 scatter_dir = sample_phase_isotropic(rng2(seed));
        dir = scatter_dir;
    }

    // free path? -> add envmap contribution
    if (free_path && show_environment > 0) {
        const vec3 Le = lookup_environment(dir);
        const float f_p = phase_isotropic();
        const float mis_weight = power_heuristic(f_p, pdf_environment(dir));
        L += throughput * mis_weight * Le;
    }

    return L;
}

void update_cache(vec3 pos, vec3 dir, inout uint seed) {
    float t;
    vec3 Li = vec3(0);
    vec3 throughput = vec3(1);
    if (sample_volumeDDA(pos, dir, t, throughput, Li, seed)) {
        pos += t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            const float f_p = phase_isotropic();
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittanceDDA(pos, w_i, seed);
            Li += mis_weight * f_p * Tr * Le_pdf.rgb / Le_pdf.w;
        }

        const vec3 scatter_dir = sample_phase_isotropic(rng2(seed));
        Li += trace_path(pos, scatter_dir, seed);
        irradiance_update(vec3(vol_inv_model * vec4(pos, 1)), Li);
    }
}

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, resolution))) return;

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * resolution.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir(pixel, resolution, rng2(seed));

    // const vec3 L = trace_path(pos, dir, 1.f, seed);
    // const vec3 L = direct_lighting(pos, dir, seed);
    // const vec3 L = vec3(transmittanceDDA(pos, dir, seed));

    update_cache(pos, dir, seed);
    vec3 L = direct_volume_rendering_irradiance_cache(pos, dir, seed);

    // write result
    imageStore(color, pixel, vec4(mix(imageLoad(color, pixel).rgb, sanitize(L), 1.f / current_sample), 1));
}
