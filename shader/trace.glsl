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
uniform float inv_max_density;
uniform float absorbtion_coefficient;
uniform float scattering_coefficient;
//uniform vec3 emission;

float density(const vec3 tc) { return texture(volume_tex, tc).r; }

float sigma_t() { return scattering_coefficient + absorbtion_coefficient; }
float sigma_a() { return scattering_coefficient / sigma_t(); }

const ivec3 voxels = textureSize(volume_tex, 0);
const float stepsize = max(1e-4f, stride / max(1e-4f, min(voxels.x, min(voxels.y, voxels.z))));

float transmittance(const vec3 pos, const vec3 dir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vec3(0), vec3(1), near_far)) return 1.f;
    // ratio tracking
    float t = near_far.x, Tr = 1.f;
    int i = 0;
    while (t < near_far.y) {
        t -= log(1 - rng(seed)) * inv_max_density / sigma_t();
        Tr *= 1 - max(0.f, density(pos + t * dir) * inv_max_density);
    }
    return Tr;
}

bool sample_volume(const vec3 pos, const vec3 dir, inout uint seed, out float t) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vec3(0), vec3(1), near_far)) return false;
    // delta tracking
    t = near_far.x;
    float Tr = 1.f;
    int i = 0;
    while (t < near_far.y) {
        t -= log(1 - rng(seed)) * inv_max_density / sigma_t();
        if (density(pos + t * dir) * inv_max_density > rng(seed))
            return true;
    }
    return false;
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
    vec3 dir = normalize(vec3(inv_model * vec4(view_dir(pixel, size, rng2(seed)), 0)));

    // trace path
    vec3 radiance = vec3(0), throughput = vec3(1);
    int n_scatter = 0;
    float t, Tr; // t: end of ray segment, Tr: transmittance along ray segment
    while (n_scatter++ < bounces && sample_volume(pos, dir, seed, t)) {
        // advance ray and adjust throughput
        pos = pos + t * dir;
        throughput *= sigma_a();

        // sample light source (environment)
        vec3 w_i;
        const vec4 Li_pdf = sample_environment(rng2(seed), w_i);

        // hacky (model space) point light
        //const vec3 w_pos = vec3(0, 1, 0);
        //w_i = w_pos - vec3(model * vec4(pos, 1));
        //const float r = length(w_i);
        //w_i = normalize(w_i);
        //const vec4 Li_pdf = vec4(vec3(10) / (r*r), 1);

        const vec3 to_light = normalize(mat3(inv_model) * w_i);
        radiance += throughput * transmittance(pos, to_light, seed) * phase_isotropic() * Li_pdf.rgb / Li_pdf.w;

        // sample phase function for scattered direction
        dir = normalize(mat3(inv_model) * sample_phase_isotropic(rng2(seed)));
    }
    // free path
    if (n_scatter > (show_environment > 0 ? 0 : 1))
        radiance += throughput * environment(normalize(mat3(model) * dir));

    // write output
    if (any(isnan(radiance)) || any(isinf(radiance))) return;
    const vec3 old = imageLoad(color, pixel).rgb;
    imageStore(color, pixel, vec4(mix(old, radiance, 1.f / current_sample), 1));
}
