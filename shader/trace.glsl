#version 450 core

#include "random.h"

layout (local_size_x = 32, local_size_y = 32) in;

layout (binding = 0, rgba32f) uniform image2D color;
layout (binding = 1, rgba32f) uniform image2D f_pos;
layout (binding = 2, rgba32f) uniform image2D f_norm;
layout (binding = 3, rgba32f) uniform image2D f_alb;
layout (binding = 4, rgba32f) uniform image2D f_vol;
layout (binding = 5, rgba32f) uniform image2D even;

uniform int current_sample;
uniform int bounces;
uniform int show_environment;

// --------------------------------------------------------------
// constants and helper funcs

#define PI float(3.14159265358979323846)
#define inv_4PI 1.f / PI

float sqr(float x) { return x * x; }

float luma(const vec3 col) { return dot(col, vec3(0.212671f, 0.715160f, 0.072169f)); }

vec3 align(const vec3 N, const vec3 v) {
    // build tangent frame
    const vec3 T = abs(N.x) > abs(N.y) ? vec3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z) : vec3(0, N.z, -N.y) / sqrt(N.y * N.y + N.z * N.z);
    const vec3 B = cross(N, T);
    // tangent to world
    return normalize(v.x * T + v.y * B + v.z * N);
}

float power_heuristic(const float a, const float b) { return sqr(a) / (sqr(a) + sqr(b)); }

// --------------------------------------------------------------
// camera helper

uniform vec3 cam_pos;
uniform float cam_fov;
uniform mat3 cam_transform;

vec3 view_dir(const ivec2 xy, const ivec2 wh, const vec2 pixel_sample) {
    const vec2 pixel = (xy + pixel_sample - wh * .5f) / vec2(wh.y);
    const float z = -.5f / tan(.5f * PI * cam_fov / 180.f);
    return normalize(cam_transform * normalize(vec3(pixel.x, pixel.y, z)));
}

// --------------------------------------------------------------
// environment helper (vectors all in world space!)

uniform mat4 env_model;
uniform mat4 env_inv_model;
uniform sampler2D env_texture;
uniform float env_integral;
uniform float env_strength;

layout(std430, binding = 0) buffer env_cdf_U {
    float cdf_U[];
};
layout(std430, binding = 1) buffer env_cdf_V {
    float cdf_V[];
};

vec3 world_to_env(const vec4 world) { return vec3(env_inv_model * world); } // position
vec3 world_to_env(const vec3 world) { return normalize(mat3(env_inv_model) * world); } // direction
vec3 env_to_world(const vec4 model) { return vec3(env_model * model); } // position
vec3 env_to_world(const vec3 model) { return normalize(mat3(env_model) * model); } // direction

vec3 environment_lookup(const vec3 dir) {
    const float u = atan(dir.z, dir.x) / (2 * PI);
    const float v = -acos(dir.y) / PI;
    return texture(env_texture, vec2(u, v)).rgb;
}

vec4 sample_environment(const vec2 env_sample, out vec3 w_i) {
    const ivec2 size = textureSize(env_texture, 0);
    ivec2 index;
    // sample V coordinate index (row) using binary search
    int ilo = 0, ihi = size.y;
    while (ilo != ihi - 1) {
        const int i = (ilo + ihi) >> 1;
        const float cdf = cdf_V[i];
        if (env_sample.y < cdf)
            ihi = i;
        else
            ilo = i;
    }
    index.y = ilo;
    // sample U coordinate index (column) using binary search
    ilo = 0, ihi = size.x;
    while (ilo != ihi - 1) {
        const int i = (ilo + ihi) >> 1;
        const float cdf = cdf_U[index.y * (size.x + 1) + i];
        if (env_sample.y < cdf)
            ihi = i;
        else
            ilo = i;
    }
    index.x = ilo;
    // continuous sampling of texture coordinates
    const float cdf_U_lo = cdf_U[index.y * (size.x + 1) + index.x];
    const float cdf_U_hi = cdf_U[index.y * (size.x + 1) + index.x + 1];
    const float du = (env_sample.x - cdf_U_lo) / (cdf_U_hi - cdf_U_lo);
    const float cdf_V_lo = cdf_V[index.y];
    const float cdf_V_hi = cdf_V[index.y + 1];
    const float dv = (env_sample.x - cdf_V_lo) / (cdf_V_hi - cdf_V_lo);
    const float u = (index.x + du) / size.x;
    const float v = (index.y + dv) / size.y;
    // convert to direction
    const float theta = v * PI;
    const float phi   = u * 2.f * PI;
    const float sin_t = sin(theta);
    w_i = vec3(sin_t * cos(phi), sin_t * sin(phi), cos(theta));
    // compute emission and pdf
    const vec3 emission = environment_lookup(w_i);
    const float pdf = (luma(emission) / env_integral) / (2.f * PI * PI * sin_t);
    return vec4(env_strength * emission, pdf);
}

float pdf_environment(const vec3 emission, const vec3 dir) {
    const float theta = acos(dir.y);
    return (luma(emission) / env_integral) / (2.f * PI * PI * sin(theta));
}

// --------------------------------------------------------------
// box intersect helper

bool intersect_box(const vec3 pos, const vec3 dir, const vec3 bb_min, const vec3 bb_max, out vec2 near_far) {
    // TODO fix inside
    const vec3 inv_dir = 1.f / dir;
    const vec3 lo = (bb_min - pos) * inv_dir;
    const vec3 hi = (bb_max - pos) * inv_dir;
    const vec3 tmin = min(lo, hi), tmax = max(lo, hi);
    near_far.x = max(tmin.x, max(tmin.y, tmin.z));
    near_far.y = min(tmax.x, min(tmax.y, tmax.z));
    return max(0.f, near_far.x) <= near_far.y;
}

// --------------------------------------------------------------
// transfer function helper

uniform float tf_window_center;
uniform float tf_window_width;
uniform sampler2D tf_lut_texture;

vec4 tf_lookup(float d) {
    return texture(tf_lut_texture, vec2((d - tf_window_center) / tf_window_width, 0));
}

// --------------------------------------------------------------
// phase function helpers (vectors all in model space!)

float phase_isotropic() { return inv_4PI; }
float phase_henyey_greenstein(const float cos_t, const float g) {
    const float denom = 1 + sqr(g) + 2 * g * cos_t;
    return inv_4PI * (1 - sqr(g)) / (denom * sqrt(denom));
}

vec3 sample_phase_isotropic(const vec2 phase_sample) {
    const float cos_t = 1.f - 2.f * phase_sample.x;
    const float sin_t = sqrt(max(0.f, 1.f - sqr(cos_t)));
    const float phi = 2.f * PI * phase_sample.y;
    return normalize(vec3(sin_t * cos(phi), sin_t * sin(phi), cos_t));
}
vec3 sample_phase_henyey_greenstein(const vec3 dir, const float g, const vec2 phase_sample) {
    const float cos_t = abs(g) < 1e-4f ? 1.f - 2.f * phase_sample.x :
        (1 + sqr(g) - sqr((1 - sqr(g)) / (1 - g + 2 * g * phase_sample.x))) / (2 * g);
    const float sin_t = sqrt(max(0.f, 1.f - sqr(cos_t)));
    const float phi = 2.f * PI * phase_sample.y;
    return align(dir, vec3(sin_t * cos(phi), sin_t * sin(phi), cos_t));
}

// --------------------------------------------------------------
// volume sampling helpers (vectors all in model space!)

uniform mat4 vol_model;
uniform mat4 vol_inv_model;
uniform sampler3D vol_texture;
uniform float vol_absorb;
uniform float vol_scatter;
uniform float vol_phase_g;

float density(const vec3 tc) { return texture(vol_texture, tc).r; }
vec3 world_to_vol(const vec4 world) { return vec3(vol_inv_model * world); } // position
vec3 world_to_vol(const vec3 world) { return normalize(mat3(vol_inv_model) * world); } // direction
vec3 vol_to_world(const vec4 model) { return vec3(vol_model * model); } // position
vec3 vol_to_world(const vec3 model) { return normalize(mat3(vol_model) * model); } // direction

// pos and dir in model (volume) space
float transmittance(const vec3 pos, const vec3 dir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vec3(0), vec3(1), near_far)) return 1.f;
    // ratio tracking
    float t = near_far.x, Tr = 1.f;
    const float sigma_t = vol_absorb + vol_scatter;
    while (t < near_far.y) {
        t -= log(1 - rng(seed)) / sigma_t;
        Tr *= 1 - max(0.f, tf_lookup(density(pos + t * dir)).a);
        // russian roulette
        const float rr_threshold = .1f;
        if (Tr < rr_threshold) {
            const float q = max(.05f, 1 - Tr);
            if (rng(seed) < q) return 0.f;
            Tr /= 1 - q;
        }
    }
    return Tr;
}

// pos and dir in model (volume) space
bool sample_volume(const vec3 pos, const vec3 dir, inout uint seed, out float t, inout vec3 throughput) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vec3(0), vec3(1), near_far)) return false;
    // delta tracking
    t = near_far.x;
    float Tr = 1.f;
    const float sigma_t = vol_absorb + vol_scatter;
     while (t < near_far.y) {
        t -= log(1 - rng(seed)) / sigma_t;
        const vec4 rgba = tf_lookup(density(pos + t * dir));
        if (rgba.a > rng(seed)) {
            throughput *= rgba.rgb * vol_scatter / sigma_t;
            return true;
        }
     }
     return false;
}

// --------------------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 size = imageSize(color);
	if (any(greaterThanEqual(pixel, size))) return;

    // setup random seed and view ray (in model space!)
    uint seed = tea(pixel.y * size.x + pixel.x, current_sample, 8);
    vec3 pos = world_to_vol(vec4(cam_pos, 1));
    vec3 dir = world_to_vol(view_dir(pixel, size, rng2(seed)));

    // trace path
    vec3 radiance = vec3(0), throughput = vec3(1);
    int n_paths = 0;
    float t, f_p = 1.f; // t: end of ray segment (i.e. sampled position or out of volume), f_p: phase function of last bounce
    while (sample_volume(pos, dir, seed, t, throughput)) {
        // advance ray
        pos = pos + t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(seed), w_i);
        if (Li_pdf.w > 0) {
            const vec3 to_light = world_to_vol(w_i);
            f_p = phase_henyey_greenstein(dot(-dir, to_light), vol_phase_g);
            const float weight = 1.f;//power_heuristic(Li_pdf.w, f_p); // TODO check MIS
            radiance += throughput * weight * f_p * transmittance(pos, to_light, seed) * Li_pdf.rgb / Li_pdf.w;
        }
        if (++n_paths >= bounces) break;

        // scatter ray
        const vec3 scatter_dir = sample_phase_henyey_greenstein(dir, vol_phase_g, rng2(seed));
        f_p = phase_henyey_greenstein(dot(-dir, scatter_dir), vol_phase_g);
        dir = scatter_dir;

        // russian roulette
        const float rr_threshold = .1f;
        const float rr_val = luma(throughput);
        if (rr_val < rr_threshold) {
            const float q = max(.05f, 1 - rr_val);
            if (rng(seed) < q) break;
            throughput /= 1 - q;
        }
    }

    // free path? -> add envmap contribution
    if (n_paths < bounces && n_paths >= show_environment) {
        const vec3 Li = env_strength * environment_lookup(vol_to_world(dir));
        const float weight = 1.f;//n_paths > 0 ? power_heuristic(f_p, pdf_environment(Li, dir)) : 1.f; // TODO check MIS
        radiance += throughput * weight * Li;
    }

    // write output
    if (any(isnan(radiance)) || any(isinf(radiance))) return;
    imageStore(color, pixel, vec4(mix(imageLoad(color, pixel).rgb, radiance, 1.f / current_sample), 1));
    if (current_sample % 2 == 1)
        imageStore(even, pixel, vec4(mix(imageLoad(even, pixel).rgb, radiance, 1.f / ((current_sample+ 1) / 2)), 1));
}
