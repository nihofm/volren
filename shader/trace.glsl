#version 450 core

#include "random.h"

layout (local_size_x = 32, local_size_y = 32) in;

layout (binding = 0, rgba32f) uniform image2D color;
layout (binding = 1, rgba32f) uniform image2D f_pos;
layout (binding = 2, rgba32f) uniform image2D f_norm;
layout (binding = 3, rgba32f) uniform image2D f_alb;
layout (binding = 4, rgba32f) uniform image2D f_vol;

uniform int current_sample;
uniform int bounces;
uniform float stride;
uniform int show_environment;

#define PI float(3.14159265358979323846)

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

uniform sampler2D environment_tex;

vec3 environment(const vec3 dir) {
    return vec3(.5); // XXX test uniform env
    const float u = atan(dir.z, dir.x) / (2 * PI);
    const float v = -acos(dir.y) / PI;
    return texture(environment_tex, vec2(u, v)).rgb;
}

vec4 sample_environment(vec2 env_sample, vec3 w_i) {
    // TODO envmap importance sampling (CDFs)
    const float z = 1.f - 2.f * env_sample.x;
    const float r = sqrt(max(0.f, 1.f - z * z));
    const float phi = 2.f * PI * env_sample.y;
    w_i = vec3(r * cos(phi), r * sin(phi), z);
    const float pdf = 1.f / (4.f * PI);
    return vec4(environment(w_i), pdf);
}

// --------------------------------------------------------------
// box intersect helper

bool intersect_box(const vec3 pos, const vec3 dir, const vec3 bb_min, const vec3 bb_max, out vec2 near_far) {
    const vec3 inv_dir = 1.f / dir;
    const vec3 lo = (bb_min - pos) * inv_dir;
    const vec3 hi = (bb_max - pos) * inv_dir;
    const vec3 tmin = min(lo, hi), tmax = max(lo, hi);
    near_far.x = max(tmin.x, max(tmin.y, tmin.z));
    near_far.y = min(tmax.x, min(tmax.y, tmax.z));
    return max(0.f, near_far.x) <= near_far.y;
}

// --------------------------------------------------------------
// phase function helpers (vectors all in model space!)

float phase_isotropic() { return 1.f / (4.f * PI); }
vec3 sample_phase_isotropic(const vec2 phase_sample) {
    const float z = 1.f - 2.f * phase_sample.x;
    const float r = sqrt(max(0.f, 1.f - z * z));
    const float phi = 2.f * PI * phase_sample.y;
    return vec3(r * cos(phi), r * sin(phi), z);
}

// --------------------------------------------------------------
// volume sampling helpers (vectors all in model space!)

uniform mat4 model;
uniform mat4 inv_model;
uniform sampler3D volume_tex;
//uniform float density_scale;
uniform float inv_max_density;
uniform float absorbtion_coefficient;
uniform float scattering_coefficient;
uniform vec3 emission;

// TODO uniform?
const float density_scale = 100.f;
const float inv_density_scale = 1.f / density_scale;
const ivec3 voxels = textureSize(volume_tex, 0);
const float stepsize = max(1e-4f, stride / max(1e-4f, min(voxels.x, min(voxels.y, voxels.z))));

float density(const vec3 tc) { return density_scale * texture(volume_tex, tc).r; }

float transmittance(const vec3 pos, const vec3 dir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vec3(0), vec3(1), near_far)) return 1.f;
    // ratio tracking
    float t = near_far.x, Tr = 1.f;
    const float sigma_t = absorbtion_coefficient + scattering_coefficient;
    while (t < near_far.y && Tr > 0.f) {
        t -= log(1 - rng(seed)) * inv_max_density * inv_density_scale / sigma_t;
        Tr *= 1 - max(0.f, density(pos + t * dir) * inv_max_density * inv_density_scale);
        // russian roulette
        const float rr_threshold = .1f;
        if (Tr < rr_threshold) {
            const float q = max(.05f, 1 - Tr);
            if (rng(seed) < q) return 0;
            Tr /= 1 - q;
        }
    }
    return Tr;
    /*
    // ray marching
    float t = near_far.x + tr_sample * stepsize, tau = 0.f;
    const float sigma_t = absorbtion_coefficient + scattering_coefficient;
    int i = 0;
    while (t < near_far.y && tau < 50.f && i++ < 1000) {
        tau += sigma_t * density(pos + t * dir) * stepsize;
        t += stepsize;
    }
    return exp(-tau);
    */
}

bool sample_volume(const vec3 pos, const vec3 dir, inout uint seed, out float t, inout vec3 throughput) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vec3(0), vec3(1), near_far)) return false;
    // delta tracking
    t = near_far.x;
    float Tr = 1.f;
    const float sigma_t = absorbtion_coefficient + scattering_coefficient;
     while (t < near_far.y) {
        t -= log(1 - rng(seed)) * inv_max_density * inv_density_scale / sigma_t;
        if (density(pos + t * dir) * inv_max_density * inv_density_scale > rng(seed)) {
            throughput *= scattering_coefficient / sigma_t;
            return true;
        }
     }
     return false;
    /*
    // ray marching
    t = near_far.x + vol_sample.x * stepsize;
    float tau = 0.f;
    const float sigma_t = absorbtion_coefficient + scattering_coefficient;
    while (t < near_far.y) {
        tau += sigma_t * density(pos + t * dir) * stepsize;
        if (tau > vol_sample.y) {
            throughput *= scattering_coefficient / sigma_t;
            return true;
        }
        t += stepsize;
    }
    return false;
    */
}

// --------------------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 size = imageSize(color);
	if (any(greaterThanEqual(pixel, size))) return;

    // setup view ray (in model space!)
    uint seed = tea(pixel.y * size.x + pixel.x, current_sample, 8);
    vec3 pos = vec3(inv_model * vec4(cam_pos, 1));
    vec3 dir = normalize(mat3(inv_model) * view_dir(pixel, size, rng2(seed)));

    // trace path
    vec3 radiance = vec3(0), throughput = vec3(1);
    int n_paths = 0;
    float t; // t: end of ray segment (i.e. sampled position or out of volume)
    while (all(greaterThan(throughput, vec3(0))) && sample_volume(pos, dir, seed, t, throughput)) {
        // advance ray and evaluate medium
        pos = pos + t * dir;
        const float d = density(pos);
        const float mu_a = absorbtion_coefficient * d;
        const float mu_s = scattering_coefficient * d;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(seed), w_i);
        const vec3 to_light = normalize(mat3(inv_model) * w_i);
        radiance += throughput * (mu_a * emission + mu_s * phase_isotropic() * transmittance(pos, to_light, seed) * Li_pdf.rgb / Li_pdf.w);
        if (++n_paths >= bounces) break;

        // scatter ray
        dir = normalize(mat3(inv_model) * sample_phase_isotropic(rng2(seed)));

        // russian roulette
        const float rr_threshold = .1f;
        const float tr_val = dot(throughput, vec3(0.212671f, 0.715160f, 0.072169f));
        if (tr_val < rr_threshold) {
            const float q = max(.05f, 1 - tr_val);
            if (rng(seed) < q) break;
            throughput /= 1 - q;
        }
    }

    // free path?
    if (n_paths < bounces && n_paths >= show_environment)
        radiance += throughput * environment(normalize(mat3(model) * dir));

    // write output
    if (any(isnan(radiance)) || any(isinf(radiance))) return;
    const vec3 prev = imageLoad(color, pixel).rgb;
    imageStore(color, pixel, vec4(mix(prev, radiance, 1.f / current_sample), 1));
}
