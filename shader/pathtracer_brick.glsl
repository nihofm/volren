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

// DDA-based volume sampling
bool sample_volumeDDA_emission(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, out vec3 Le, inout uint seed) {
    Le = vec3(0);
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    const vec3 ri = 1.f / idir;
    // march brick grid
    t = near_far.x + 1e-6f;
    float tau = -log(1.f - rng(seed)), mip = MIP_START;
    while (t < near_far.y) {
        const vec3 curr = ipos + t * idir;
#ifdef USE_TRANSFERFUNC
        const float majorant = vol_majorant * tf_lookup(lookup_majorant(curr, int(round(mip))) * vol_inv_majorant).a;
#else
        const float majorant = lookup_majorant(curr, int(round(mip)));
#endif
        const float dt = stepDDA(curr, ri, int(round(mip)));
        t += dt;
        tau -= majorant * dt;
        mip = min(mip + MIP_SPEED_UP, 3.f);
        if (tau > 0) continue; // no collision, step ahead
        t += tau / majorant; // step back to point of collision
        if (t >= near_far.y) break;
#ifdef USE_TRANSFERFUNC
        const vec4 rgba = tf_lookup(lookup_density(ipos + t * idir, seed) * vol_inv_majorant);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density(ipos + t * idir, seed);
        Le += throughput * lookup_emission(ipos + t * idir, seed) * d * vol_inv_majorant;
#endif
        if (rng(seed) * majorant < d) { // check if real or null collision
            throughput *= vol_albedo;
#ifdef USE_TRANSFERFUNC
            throughput *= rgba.rgb;
#endif
            return true;
        }
        tau = -log(1.f - rng(seed));
        mip = max(0.f, mip - MIP_SPEED_DOWN);
    }
    return false;
}

vec3 trace_path(vec3 pos, vec3 dir, inout uint seed) {
    // trace path
    vec3 L = vec3(0), throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
    while (sample_volumeDDA(pos, dir, t, throughput, L, seed)) {
        // advance ray
        pos = pos + t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
            const float Tr = transmittanceDDA(pos, w_i, seed);
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

    return L;
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

    // trace ray
    const vec3 L = trace_path(pos, dir, seed);
    // const vec3 L = direct_volume_rendering(pos, dir, seed);

    // write result
    imageStore(color, pixel, vec4(mix(imageLoad(color, pixel).rgb, sanitize(L), 1.f / current_sample), 1));
}
