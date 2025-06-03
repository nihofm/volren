// --------------------------------------------------------------
// constants and helper funcs

#define M_PI float(3.14159265358979323846)
#define inv_PI (1.f / M_PI)
#define inv_2PI (1.f / (2 * M_PI))
#define inv_4PI (1.f / (4 * M_PI))
#define FLT_MAX float(3.402823466e+38)

float sqr(float x) { return x * x; }
vec3 sqr(vec3 x) { return x * x; }

float sum(const vec3 x) { return x.x + x.y + x.z; }

float mean(const vec3 x) { return sum(x) / 3.f; }

float sanitize(const float x) { return isnan(x) || isinf(x) ? 0.f : x; }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }
vec4 sanitize(const vec4 x) { return mix(x, vec4(0), isnan(x) || isinf(x)); }

float luma(const vec3 col) { return dot(col, vec3(0.212671f, 0.715160f, 0.072169f)); }

float saturate(const float x) { return clamp(x, 0.f, 1.f); }

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
    return float(previous & 0x00FFFFFFu) / float(0x01000000u);
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
    const float z = -.5f / tan(.5f * M_PI * cam_fov / 180.f);
    return normalize(cam_transform * normalize(vec3(pixel.x, pixel.y, z)));
}

// --------------------------------------------------------------
// environment helper (input vectors assumed in world space!)

uniform mat3 env_transform;
uniform mat3 env_inv_transform;
uniform float env_strength;
uniform vec2 env_imp_inv_dim;
uniform int env_imp_base_mip;
uniform sampler2D env_envmap;
uniform sampler2D env_impmap;

vec3 lookup_environment(const vec3 dir) {
    const vec3 idir = env_inv_transform * dir;
    const float u = atan(idir.z, idir.x) / (2 * M_PI) + 0.5f;
    const float v = 1.f - acos(idir.y) / M_PI;
    return env_strength * texture(env_envmap, vec2(u, v)).rgb;
}

vec4 sample_environment(const vec2 rng, out vec3 w_i) {
    ivec2 pos = ivec2(0);   // pixel position
    vec2 p = rng;           // sub-pixel position
    // warp sample over mip hierarchy
    for (int mip = env_imp_base_mip - 1; mip >= 0; mip--) {
        pos *= 2; // scale to mip
        float w[4]; // four relevant texels
        w[0] = texelFetch(env_impmap, pos + ivec2(0, 0), mip).r;
        w[1] = texelFetch(env_impmap, pos + ivec2(1, 0), mip).r;
        w[2] = texelFetch(env_impmap, pos + ivec2(0, 1), mip).r;
        w[3] = texelFetch(env_impmap, pos + ivec2(1, 1), mip).r;
        float q[2]; // bottom / top
        q[0] = w[0] + w[2];
        q[1] = w[1] + w[3];
        // horizontal
        int off_x;
        const float d = q[0] / max(1e-8f, q[0] + q[1]);
        if (p.x < d) { // left
            off_x = 0;
            p.x = p.x / d;
        } else { // right
            off_x = 1;
            p.x = (p.x - d) / (1.f - d);
        }
        pos.x += off_x;
        // vertical
        float e = w[off_x] / q[off_x];
        if (p.y < e) { // bottom
            //pos.y += 0;
            p.y = p.y / e;
        } else { // top
            pos.y += 1;
            p.y = (p.y - e) / (1.f - e);
        }
    }
    // compute sample uv coordinate and (world-space) direction
    const vec2 uv = (vec2(pos) + p) * env_imp_inv_dim;
    const float theta = saturate(1.f - uv.y) * M_PI;
    const float phi   = (saturate(uv.x) * 2.f - 1.f) * M_PI;
    const float sin_t = sin(theta);
    w_i = env_transform * vec3(sin_t * cos(phi), cos(theta), sin_t * sin(phi));
    // sample envmap and compute pdf
    const vec3 Le = env_strength * texture(env_envmap, uv).rgb;
    const float avg_w = texelFetch(env_impmap, ivec2(0, 0), env_imp_base_mip).r;
    const float pdf = texelFetch(env_impmap, pos, 0).r / avg_w;
    return vec4(Le, pdf * inv_4PI);
}

float pdf_environment(const vec3 dir) {
    const float avg_w = texelFetch(env_impmap, ivec2(0, 0), env_imp_base_mip).r;
    const float pdf = luma(lookup_environment(dir)) / avg_w;
    return pdf * inv_4PI;
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
// phase function helpers

float phase_isotropic() { return inv_4PI; }

float phase_henyey_greenstein(const float cos_t, const float g) {
    const float denom = 1 + sqr(g) + 2 * g * cos_t;
    return inv_4PI * (1 - sqr(g)) / (denom * sqrt(denom));
}

vec3 sample_phase_isotropic(const vec2 phase_sample) {
    const float cos_t = 1.f - 2.f * phase_sample.x;
    const float sin_t = sqrt(max(0.f, 1.f - sqr(cos_t)));
    const float phi = 2.f * M_PI * phase_sample.y;
    return normalize(vec3(sin_t * cos(phi), sin_t * sin(phi), cos_t));
}

vec3 sample_phase_henyey_greenstein(const vec3 dir, const float g, const vec2 phase_sample) {
    const float cos_t = abs(g) < 1e-4f ? 1.f - 2.f * phase_sample.x :
        (1 + sqr(g) - sqr((1 - sqr(g)) / (1 - g + 2 * g * phase_sample.x))) / (2 * g);
    const float sin_t = sqrt(max(0.f, 1.f - sqr(cos_t)));
    const float phi = 2.f * M_PI * phase_sample.y;
    return align(dir, vec3(sin_t * cos(phi), sin_t * sin(phi), cos_t));
}

// --------------------------------------------------------------
// transfer function helper

layout(std430, binding = 4) buffer LUTBuffer {
    vec4 tf_lut[];
};

uniform uint tf_size;
uniform float tf_window_left;
uniform float tf_window_width;

float tf_window(float d) {
    return clamp((d - tf_window_left) / tf_window_width, 0.0, 1.0 - 1e-6);
}

vec4 tf_lookup(float d) {
    const float tc = tf_window(d);
    const int idx = int(floor(tc * tf_size));
    const float f = fract(tc * tf_size);
    return mix(tf_lut[idx], tf_lut[min(idx + 1, tf_size - 1)], f);
}

// --------------------------------------------------------------
// stochastic filter helpers

ivec3 stochastic_trilinear_filter(const vec3 ipos, inout uint seed) {
    return ivec3(ipos - 0.5 + rng3(seed));
}

ivec3 stochastic_tricubic_filter(const vec3 ipos, inout uint seed) {
    // from "Stochastic Texture Filtering": https://arxiv.org/pdf/2305.05810.pdf
    const ivec3 iipos = ivec3(floor(ipos - 0.5));
    const vec3 t = (ipos - 0.5) - iipos;
    const vec3 t2 = t * t;
    // weighted reservoir sampling, first tap always accepted
    vec3 w = (1.f / 6.f) * (-t * t2 + 3 * t2 - 3 * t + 1);
    vec3 sumWt = w;
    ivec3 idx = ivec3(0);
    // sample second tap
    w = (1.f / 6.f) * (3 * t * t2 - 6 * t2 + 4);
    sumWt = w + sumWt;
    idx = mix(idx, ivec3(1), lessThan(rng3(seed), w / max(vec3(1e-3), sumWt)));
    // sample third tap
    w = (1.f / 6.f) * (-3 * t * t2 + 3 * t2 + 3 * t + 1);
    sumWt = w + sumWt;
    idx = mix(idx, ivec3(2), lessThan(rng3(seed), w / max(vec3(1e-3), sumWt)));
    // sample fourth tap
    w = (1.f / 6.f) * t * t2;
    sumWt = w + sumWt;
    idx = mix(idx, ivec3(3), lessThan(rng3(seed), w / max(vec3(1e-3), sumWt)));
    // return tap location
    return iipos + idx - 1;
}

// --------------------------------------------------------------
// volume sampling helpers (input vectors assumed in index space!)

uniform vec3 vol_bb_min;
uniform vec3 vol_bb_max;
uniform float vol_minorant;
uniform float vol_majorant;
uniform float vol_inv_majorant;
uniform vec3 vol_albedo;
uniform float vol_phase_g;
uniform float vol_density_scale;
uniform float vol_emission_scale;
uniform float vol_emission_norm;

// density brick grid stored as textures
uniform mat4 vol_density_transform;
uniform mat4 vol_density_inv_transform;
uniform usampler3D vol_density_indirection;
uniform sampler3D vol_density_range;
uniform sampler3D vol_density_atlas;

// brick grid voxel density lookup (nearest neighbor)
float lookup_density_brick(const vec3 ipos) {
    const ivec3 iipos = ivec3(floor(ipos));
    const ivec3 brick = iipos >> 3;
    const uvec3 ptr = texelFetch(vol_density_indirection, brick, 0).xyz;
    const vec2 range = texelFetch(vol_density_range, brick, 0).xy;
    const float value_unorm = texelFetch(vol_density_atlas, ivec3(ptr << 3) + (iipos & 7), 0).x;
    return range.x + value_unorm * (range.y - range.x);
}

// brick majorant lookup (nearest neighbor)
float lookup_majorant(const vec3 ipos, int mip) {
    const ivec3 brick = ivec3(floor(ipos)) >> (3 + mip);
    return vol_density_scale * texelFetch(vol_density_range, brick, mip).y;
}

// density lookup (nearest neighbor)
float lookup_density(const vec3 ipos) {
    return vol_density_scale * lookup_density_brick(ipos);
}

// density lookup (trilinear filter)
float lookup_density_trilinear(const vec3 ipos) {
    const vec3 f = fract(ipos - 0.5);
    const ivec3 iipos = ivec3(floor(ipos - 0.5));
    const float lx0 = mix(lookup_density_brick(iipos + ivec3(0, 0, 0)), lookup_density_brick(iipos + ivec3(1, 0, 0)), f.x);
    const float lx1 = mix(lookup_density_brick(iipos + ivec3(0, 1, 0)), lookup_density_brick(iipos + ivec3(1, 1, 0)), f.x);
    const float hx0 = mix(lookup_density_brick(iipos + ivec3(0, 0, 1)), lookup_density_brick(iipos + ivec3(1, 0, 1)), f.x);
    const float hx1 = mix(lookup_density_brick(iipos + ivec3(0, 1, 1)), lookup_density_brick(iipos + ivec3(1, 1, 1)), f.x);
    return vol_density_scale * mix(mix(lx0, lx1, f.y), mix(hx0, hx1, f.y), f.z);
}

// density lookup (stochastic tricubic filter)
float lookup_density_stochastic(const vec3 ipos, inout uint seed) {
    // return lookup_density(ivec3(ipos));
    // return lookup_density(stochastic_trilinear_filter(ipos, seed));
    return lookup_density(stochastic_tricubic_filter(ipos, seed));
}

// temperature brick grid stored as textures
uniform mat4 vol_emission_transform;
uniform mat4 vol_emission_inv_transform;
uniform usampler3D vol_emission_indirection;
uniform sampler3D vol_emission_range;
uniform sampler3D vol_emission_atlas;

// brick grid voxel temperature lookup (nearest neighbor)
float lookup_temperature_brick(const vec3 ipos) {
    const ivec3 iipos = ivec3(floor(ipos));
    const ivec3 brick = iipos >> 3;
    const uvec3 ptr = texelFetch(vol_emission_indirection, brick, 0).xyz;
    const vec2 range = texelFetch(vol_emission_range, brick, 0).xy;
    const float value_unorm = texelFetch(vol_emission_atlas, ivec3(ptr << 3) + (iipos & 7), 0).x;
    return range.x + value_unorm * (range.y - range.x);
}

// emission lookup (stochastic tricubic filter)
vec3 lookup_emission(const vec3 ipos, inout uint seed) {
    const vec3 ipos_emission = vec3(vol_emission_inv_transform * vol_density_transform * vec4(ipos, 1));
    const float t = lookup_temperature_brick(stochastic_tricubic_filter(ipos_emission, seed)) * vol_emission_norm;
    return vol_emission_scale * sqr(vec3(t, sqr(t), sqr(sqr(t))));
}

// --------------------------------------------------------------
// null-collision methods

float transmittance(const vec3 wpos, const vec3 wdir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // to index-space
    const vec3 ipos = vec3(vol_density_inv_transform * vec4(wpos, 1));
    const vec3 idir = vec3(vol_density_inv_transform * vec4(wdir, 0)); // non-normalized!
    // ratio tracking
    float t = near_far.x - log(1 - rng(seed)) * vol_inv_majorant, Tr = 1.f;
    while (t < near_far.y) {
#ifdef USE_TRANSFERFUNC
        const vec4 rgba = tf_lookup(lookup_density_trilinear(ipos + t * idir) * vol_inv_majorant);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density_stochastic(ipos + t * idir, seed);
#endif
        // track ratio of real to null particles
        Tr *= 1 - d * vol_inv_majorant;
        // russian roulette
        if (Tr < .1f) {
            const float prob = 1 - Tr;
            if (rng(seed) < prob) return 0.f;
            Tr /= 1 - prob;
        }
        // advance
        t -= log(1 - rng(seed)) * vol_inv_majorant;
    }
    return Tr;
}

bool sample_volume(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, inout vec3 Le, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_density_inv_transform * vec4(wpos, 1));
    const vec3 idir = vec3(vol_density_inv_transform * vec4(wdir, 0)); // non-normalized!
    // delta tracking
    t = near_far.x - log(1 - rng(seed)) * vol_inv_majorant;
    while (t < near_far.y) {
#ifdef USE_TRANSFERFUNC
        const vec4 rgba = tf_lookup(lookup_density_trilinear(ipos + t * idir) * vol_inv_majorant);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density_stochastic(ipos + t * idir, seed);
#endif
        const float P_real = d * vol_inv_majorant;
        Le += throughput * (1 - vol_albedo) * lookup_emission(ipos + t * idir, seed) * P_real;
        // classify as real or null collison
        if (rng(seed) < P_real) {
#ifdef USE_TRANSFERFUNC
            throughput *= rgba.rgb * vol_albedo;
#else
            throughput *= vol_albedo;
#endif
            return true;
        }
        // advance
        t -= log(1 - rng(seed)) * vol_inv_majorant;
    }
    return false;
}

// --------------------------------------------------------------
// DDA-based null-collision methods

#define MIP_START 3
#define MIP_SPEED_UP 0.25
#define MIP_SPEED_DOWN 2

// perform DDA step on given mip level
float stepDDA(const vec3 pos, const vec3 inv_dir, const int mip) {
    const float dim = 8 << mip;
    const vec3 offs = mix(vec3(-0.5f), vec3(dim + 0.5f), greaterThanEqual(inv_dir, vec3(0)));
    const vec3 tmax = (floor(pos * (1.f / dim)) * dim + offs - pos) * inv_dir;
    return min(tmax.x, min(tmax.y, tmax.z));
}

// DDA-based transmittance
float transmittanceDDA(const vec3 wpos, const vec3 wdir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // to index-space
    const vec3 ipos = vec3(vol_density_inv_transform * vec4(wpos, 1));
    const vec3 idir = vec3(vol_density_inv_transform * vec4(wdir, 0)); // non-normalized!
    const vec3 ri = 1.f / idir;
    // march brick grid
    float t = near_far.x + 1e-6f, Tr = 1.f, tau = -log(1.f - rng(seed)), mip = MIP_START;
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
        const vec4 rgba = tf_lookup(lookup_density_trilinear(ipos + t * idir) * vol_inv_majorant);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density_stochastic(ipos + t * idir, seed);
#endif
        if (rng(seed) * majorant < d) { // check if real or null collision
            Tr *= max(0.f, 1.f - vol_majorant / majorant); // adjust by ratio of global to local majorant
            // russian roulette
            if (Tr < .1f) {
                const float prob = 1 - Tr;
                if (rng(seed) < prob) return 0.f;
                Tr /= 1 - prob;
            }
        }
        tau = -log(1.f - rng(seed));
        mip = max(0.f, mip - MIP_SPEED_DOWN);
    }
    return Tr;
}

// DDA-based volume sampling
bool sample_volumeDDA(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, inout vec3 Le, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_density_inv_transform * vec4(wpos, 1));
    const vec3 idir = vec3(vol_density_inv_transform * vec4(wdir, 0)); // non-normalized!
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
        const vec4 rgba = tf_lookup(lookup_density_trilinear(ipos + t * idir) * vol_inv_majorant);
        const float d = vol_majorant * rgba.a;
#else
        const float d = lookup_density_stochastic(ipos + t * idir, seed);
#endif
        Le += throughput * (1.f - vol_albedo) * lookup_emission(ipos + t * idir, seed) * d * vol_inv_majorant;
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

// --------------------------------------------------------------
// ray-marching

#define RAYMARCH_STEPS 64

float transmittance_raymarch(const vec3 wpos, const vec3 wdir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // to index-space
    const vec3 ipos = vec3(vol_density_inv_transform * vec4(wpos, 1));
    const vec3 idir = vec3(vol_density_inv_transform * vec4(wdir, 0)); // non-normalized!
    // ray marching
    const float dt = (near_far.y - near_far.x) / float(RAYMARCH_STEPS);
    near_far.x += rng(seed) * dt; // jitter starting position
    float tau = 0.f;
    for (int i = 0; i < RAYMARCH_STEPS; ++i) {
#ifdef USE_TRANSFERFUNC
        tau += tf_lookup(lookup_density_stochastic(ipos + min(near_far.x + i * dt, near_far.y) * idir, seed) * vol_inv_majorant).a * vol_majorant * dt;
#else
        tau += lookup_density_stochastic(ipos + min(near_far.x + i * dt, near_far.y) * idir, seed) * dt;
#endif
    }
    return exp(-tau);
}

bool sample_volume_raymarch(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, out float pdf, inout uint seed) {
    pdf = 1.f;
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_density_inv_transform * vec4(wpos, 1));
    const vec3 idir = vec3(vol_density_inv_transform * vec4(wdir, 0)); // non-normalized!
    // ray marching
    const float tau_target = -log(1.f - rng(seed));
    const float dt = (near_far.y - near_far.x) / float(RAYMARCH_STEPS);
    near_far.x += rng(seed) * dt; // jitter starting position
    float tau = 0.f;
    for (int i = 0; i < RAYMARCH_STEPS; ++i) {
        t = min(near_far.x + i * dt, near_far.y);
#ifdef USE_TRANSFERFUNC
        const float d = lookup_density_stochastic(ipos + t * idir, seed);
        const vec4 rgba = tf_lookup(d * vol_inv_majorant);
        tau += rgba.a * vol_majorant * dt;
#else
        const float d = lookup_density_stochastic(ipos + t * idir, seed);
        tau += d * dt;
#endif
        if (tau >= tau_target) {
            // TODO revert to exact hit pos
#ifdef USE_TRANSFERFUNC
            const vec3 albedo = rgba.rgb * vol_albedo;
#else
            const vec3 albedo = vec3(vol_albedo);
#endif
            pdf = mean(albedo) * d * exp(-tau_target);
            throughput *= albedo;
            return true;
        }
    }
    pdf = exp(-tau);
    return false;
}

// --------------------------------------------------------------
// simple direct volume rendering

vec3 direct_volume_rendering(vec3 pos, vec3 dir, inout uint seed) {
    vec3 L = vec3(0);
    // clip volume
    vec2 near_far;
    if (!intersect_box(pos, dir, vol_bb_min, vol_bb_max, near_far)) return lookup_environment(dir);
    // to index-space
    const vec3 ipos = vec3(vol_density_inv_transform * vec4(pos, 1));
    const vec3 idir = vec3(vol_density_inv_transform * vec4(dir, 0)); // non-normalized!
    // ray marching
    const float dt = (near_far.y - near_far.x) / float(RAYMARCH_STEPS);
    near_far.x += rng(seed) * dt; // jitter starting position
    float Tr = 1.f;
    for (int i = 0; i < RAYMARCH_STEPS; ++i) {
        const vec4 rgba = tf_lookup(lookup_density_trilinear(ipos + min(near_far.x + i * dt, near_far.y) * idir) * vol_inv_majorant);
        const float dtau = rgba.a * vol_majorant * dt;
        L += rgba.rgb * dtau * Tr;
        Tr *= exp(-dtau);
        if (Tr <= 1e-6) return L;
    }
    return L + lookup_environment(dir) * Tr;
}

// --------------------------------------------------------------
// volumetric path tracing

uniform int bounces;
uniform int show_environment;

vec4 trace_path(vec3 pos, vec3 dir, inout uint seed) {
    // trace path
    vec3 L = vec3(0);
    vec3 throughput = vec3(1);
    bool free_path = true;
    uint n_paths = 0;
    float t, f_p; // t: end of ray segment (i.e. sampled position or out of volume), f_p: last phase function sample for MIS
#ifdef USE_DDA
    while (sample_volumeDDA(pos, dir, t, throughput, L, seed)) {
#else
    while (sample_volume(pos, dir, t, throughput, L, seed)) {
#endif
        // advance ray
        pos = pos + t * dir;

        // sample light source (environment)
        vec3 w_i;
        const vec4 Le_pdf = sample_environment(rng2(seed), w_i);
        if (Le_pdf.w > 0) {
            f_p = phase_henyey_greenstein(dot(-dir, w_i), vol_phase_g);
            const float mis_weight = show_environment > 0 ? power_heuristic(Le_pdf.w, f_p) : 1.f;
#ifdef USE_DDA
            const float Tr = transmittanceDDA(pos, w_i, seed);
#else
            const float Tr = transmittance(pos, w_i, seed);
#endif
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

    return vec4(L, clamp(n_paths, 0.f, 1.f));
}
