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
// random number generation

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
uniform sampler3D vol_texture;
uniform float vol_absorb;
uniform float vol_scatter;
uniform float vol_phase_g;
uniform vec3 vol_bb_min;
uniform vec3 vol_bb_max;

float density(const vec3 tc) { return texture(vol_texture, tc).r; }
vec3 world_to_vol(const vec4 world) { return vec3(vol_inv_model * world); } // position
vec3 world_to_vol(const vec3 world) { return normalize(mat3(vol_inv_model) * world); } // direction
vec3 vol_to_world(const vec4 model) { return vec3(vol_model * model); } // position
vec3 vol_to_world(const vec3 model) { return normalize(mat3(vol_model) * model); } // direction

// pos and dir assumed in model (volume) space
float transmittance(const vec3 pos, const vec3 dir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // ratio tracking
    float t = near_far.x, Tr = 1.f;
    const float sigma_t = vol_absorb + vol_scatter;
    while (t < near_far.y) {
        t -= log(1 - rng(seed)) / sigma_t;
        Tr *= 1 - max(0.f, tf_lookup(density(pos + t * dir)).a);
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
bool sample_volume(const vec3 pos, const vec3 dir, inout uint seed, out float t, inout vec3 throughput) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return false;
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


// ---------------------------------------------------
// path tracing

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
// weighted reservoir sampling / resampled importance sampling

struct Reservoir {
    vec3 y;         // the output sample
    float p_hat;    // the output sample target pdf
    float w_sum;    // the sum of weights
    uint M;         // the number of samples seen so far
};
layout(std430, binding = 2) buffer ReservoirBuffer {
    Reservoir reservoirs[];
};

void wrs_init(inout Reservoir r) {
    r.y = vec3(0);
    r.p_hat = 0;
    r.w_sum = 0;
    r.M = 0;
}

void wrs_update(inout Reservoir r, const vec3 xi, const float wi, const float pi, inout uint seed) {
    const float w = pi / wi;
    r.M += 1;
    r.w_sum += w;
    if (rng(seed) < w / r.w_sum) {
        r.y = xi;
        r.p_hat = pi;
    }
}

// TODO decide on target pdf / use-case
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
    return r.w_sum / (r.p_hat * r.M);
}

float stream_ris_single(inout Reservoir r, const vec3 xi, const float wi, const float pi, inout uint seed) {
    wrs_update(r, xi, wi, pi, seed);
    return r.w_sum / (r.p_hat * r.M);
}

/*
// RIS/WRS
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
