#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_atomic_float : enable
#extension GL_NV_shader_atomic_fp16_vector : enable

// --------------------------------------------------------------
// constants and helper funcs

#define PI float(3.14159265358979323846)
#define inv_4PI 1.f / PI

float sqr(float x) { return x * x; }

float luma(const vec3 col) { return dot(col, vec3(0.212671f, 0.715160f, 0.072169f)); }

vec3 align(const vec3 N, const vec3 v) {
    // build tangent frame
    const vec3 T = abs(N.x) > abs(N.y) ?
        vec3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z) :
        vec3(0, N.z, -N.y) / sqrt(N.y * N.y + N.z * N.z);
    const vec3 B = cross(N, T);
    // tangent to world
    return normalize(v.x * T + v.y * B + v.z * N);
}

float power_heuristic(const float a, const float b) { return sqr(a) / (sqr(a) + sqr(b)); }

// --------------------------------------------------------------
// random number generation helpers

uint tea(const uint val0, const uint val1, const uint N) { // tiny encryption algorithm (TEA) to calculate a seed per launch index and iteration
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;
    for (uint n = 0; n < N; ++n) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
        v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
    }
    return v0;
}

float rng(inout uint previous) { // return a random sample in the range [0, 1) with a simple linear congruential generator
    previous = previous * 1664525u + 1013904223u;
    return float(previous & 0x00FFFFFF) / float(0x01000000u);
}
vec2 rng2(inout uint previous) {
    return vec2(rng(previous), rng(previous));
}
vec3 rng3(inout uint previous) {
    return vec3(rng(previous), rng(previous), rng(previous));
}
vec4 rng4(inout uint previous) {
    return vec4(rng(previous), rng(previous), rng(previous), rng(previous));
}

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
// environment helper (input vectors assumed in world space!)
// TODO mipmap based sample warping scheme

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
    return vec4(emission, pdf);
}

float pdf_environment(const vec3 emission, const vec3 dir) {
    const float theta = acos(dir.y);
    return (luma(emission) / env_integral) / (2.f * PI * PI * sin(theta));
}

// --------------------------------------------------------------
// box intersect helper

bool intersect_box(const vec3 pos, const vec3 dir, const vec3 bb_min, const vec3 bb_max, out vec2 near_far) {
    const vec3 inv_dir = 1.f / dir;
    const vec3 lo = (bb_min - pos) * inv_dir;
    const vec3 hi = (bb_max - pos) * inv_dir;
    const vec3 tmin = min(lo, hi), tmax = max(lo, hi);
    near_far.x = max(0.f, max(tmin.x, max(tmin.y, tmin.z)));
    near_far.y = min(tmax.x, min(tmax.y, tmax.z));
    return near_far.x <= near_far.y;
}

// --------------------------------------------------------------
// transfer function helper

uniform float tf_window_center;
uniform float tf_window_width;
uniform sampler2D tf_texture;

// TODO stochastic lookup
vec4 tf_lookup(float d) {
    const vec4 lut = texture(tf_texture, vec2((d - tf_window_center) / tf_window_width, 0));
    return vec4(lut.rgb, d * lut.a);
}

// --------------------------------------------------------------
// phase function helpers

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
// volume sampling helpers (input vectors assumed in model space!)

uniform mat4 vol_model;
uniform mat4 vol_inv_model;
uniform vec3 vol_bb_min;
uniform vec3 vol_bb_max;
uniform vec3 vol_albedo;
uniform float vol_phase_g;
uniform float vol_density_scale;

// dense grid
//uniform sampler3D vol_texture;
uniform float vol_inv_majorant;

// brick grid
uniform usampler3D vol_indirection;
uniform sampler3D vol_range;
uniform sampler3D vol_atlas;

// brick voxel lookup
float lookup_voxel(const vec3 ipos) {
    const ivec3 brick = ivec3(ipos) >> 3;
    const uvec3 ptr = texelFetch(vol_indirection, brick, 0).xyz;
    const vec2 range = texelFetch(vol_range, brick, 0).xy;
    const float value_unorm = texelFetch(vol_atlas, ivec3(ptr << 3) + (ivec3(ipos) & 7), 0).x;
    return value_unorm * (range.y - range.x) + range.x;
}

// brick majorant lookup
float lookup_majorant(const vec3 ipos) {
    return vol_density_scale * texelFetch(vol_range, ivec3(ipos) >> 3, 0).y;
}

// density lookup with stochastic lerp
float density(const vec3 ipos, inout uint seed) {
    return vol_density_scale * lookup_voxel(ivec3((ipos + rng3(seed) - .5f)));
}

// pos and dir assumed in model (volume) space
float transmittance(in vec3 pos, in vec3 dir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // to index-space
    pos = vec3(vol_inv_model * vec4(pos, 1));
    dir = vec3(vol_inv_model * vec4(dir, 0)); // non-normalized!
    // ratio tracking
    float t = near_far.x, Tr = 1.f;
    while (t < near_far.y) {
        t -= log(1 - rng(seed)) * vol_inv_majorant;
        Tr *= max(0.f, 1 - tf_lookup(density(pos + t * dir, seed) * vol_inv_majorant).a);
        // russian roulette
        const float rr_threshold = .1f;
        if (Tr < rr_threshold) {
            const float prob = 1 - Tr;
            if (rng(seed) < prob) return 0.f;
            Tr /= 1 - prob;
        }
    }
    return Tr;
}

// pos and dir assumed in model (volume) space
bool sample_volume(in vec3 pos, in vec3 dir, out float t, inout vec3 throughput, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    pos = vec3(vol_inv_model * vec4(pos, 1));
    dir = vec3(vol_inv_model * vec4(dir, 0)); // non-normalized!
    // delta tracking
    t = near_far.x;
    float Tr = 1.f;
     while (t < near_far.y) {
        t -= log(1 - rng(seed)) * vol_inv_majorant;
        const vec4 rgba = tf_lookup(density(pos + t * dir, seed) * vol_inv_majorant);
        if (rng(seed) < rgba.a) {
            throughput *= rgba.rgb * vol_albedo;
            return true;
        }
     }
     return false;
}

float stepDDA(in vec3 ro, in vec3 ri, in vec3 pos, in int mip) {
    const float dim = 8 << mip;
    const vec3 ofs = ri * (mix(vec3(-0.5f), vec3(dim + 0.5f), greaterThanEqual(ri, vec3(0))) - ro);
    const vec3 tmax = floor(pos * (1.f / dim)) * dim * ri + ofs;
    return min(tmax.x, min(tmax.y , tmax.z));
}

// TODO debug DDA
vec3 transmittanceDDA(in vec3 pos, in vec3 dir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return vec3(1.f);
    // to index-space
    pos = vec3(vol_inv_model * vec4(pos, 1));
    dir = vec3(vol_inv_model * vec4(dir, 0)); // non-normalized!
    const vec3 ri = 1.f / dir;
    float t = near_far.x - 0.01f, Tr = 1.f, tau = -log(1.f - rng(seed));
    while (t < near_far.y) {
        const vec3 curr = pos + t * dir;
        const float majorant = lookup_majorant(ivec3(curr));
        const float next_t = stepDDA(pos, ri, curr, 0);
        const float dt = next_t - t;
        if (dt < 0.f) return vec3(1, 0, 0);
        const float dtau = majorant * dt;
        t = next_t;
        tau -= dtau;
        if (tau > 0) continue; // no collision, step ahead
        t += dt * tau / dtau ; // step back to point of collision
        const float d = density(pos + t * dir, seed);
        if (d > majorant) return vec3(1, 1, 0);
        if (rng(seed) * majorant < d) { // check if real or null collision
            Tr *= 1.f - d / majorant;
            // russian roulette
            if (Tr < .1f) {
                const float prob = 1 - Tr;
                if (rng(seed) < prob) return vec3(0.f);
                Tr /= 1 - prob;
            }
        }
        tau = -log(1.f - rng(seed));
    }
    return vec3(Tr);
}
