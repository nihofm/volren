// --------------------------------------------------------------
// common ray state struct

struct ray_state {
    vec3 pos;
    float near;
    vec3 dir;
    float far;
    ivec2 pixel;
    uint seed;
    uint n_paths;
};

// --------------------------------------------------------------
// constants and helper funcs

#define M_PI float(3.14159265358979323846)
#define inv_PI (1.f / M_PI)
#define inv_2PI (1.f / (2 * M_PI))
#define inv_4PI (1.f / (4 * M_PI))
#define FLT_MAX float(3.402823466e+38)

float sqr(float x) { return x * x; }

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

uniform mat3 env_model;
uniform mat3 env_inv_model;
uniform float env_strength;
//uniform float env_integral; // TODO precompute
uniform vec2 env_imp_inv_dim;
uniform int env_imp_base_mip;
uniform sampler2D env_envmap;
uniform sampler2D env_impmap;

vec3 lookup_environment(const vec3 dir) {
    const vec3 idir = env_inv_model * dir;
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
    w_i = env_model * vec3(sin_t * cos(phi), cos(theta), sin_t * sin(phi));
    // sample envmap and compute pdf
    const vec3 Le = env_strength * texture(env_envmap, uv).rgb;
    const float avg_w = texelFetch(env_impmap, ivec2(0, 0), env_imp_base_mip).r; // TODO precompute (uniform)
    const float pdf = texelFetch(env_impmap, pos, 0).r / avg_w;
    return vec4(Le, pdf * inv_4PI);
}

float pdf_environment(const vec3 dir) {
    const float avg_w = texelFetch(env_impmap, ivec2(0, 0), env_imp_base_mip).r; // TODO precompute (uniform)
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

uniform float tf_window_left;
uniform float tf_window_width;
uniform sampler2D tf_texture;

// TODO stochastic lookup filter for transferfunc
vec4 tf_lookup(float d) {
    const vec4 lut = texture(tf_texture, vec2(max(0.f, d - tf_window_left) / tf_window_width, 0));
    return vec4(lut.rgb, lut.a);
}

// --------------------------------------------------------------
// volume sampling helpers (input vectors assumed in model space!)

uniform mat4 vol_model;
uniform mat4 vol_inv_model;
uniform vec3 vol_bb_min;
uniform vec3 vol_bb_max;
uniform float vol_minorant;
uniform float vol_inv_minorant;
uniform float vol_majorant;
uniform float vol_inv_majorant;
uniform vec3 vol_albedo;
uniform float vol_phase_g;
uniform float vol_density_scale;
uniform int vol_grid_type; // underlying grid data type

// dense grid (only valid if vol_grid_type == 1)
uniform sampler3D vol_dense;

// dense grid voxel lookup
float lookup_voxel_dense(const vec3 ipos) {
    return texelFetch(vol_dense, ivec3(floor(ipos)), 0).r;
    return vol_minorant + texelFetch(vol_dense, ivec3(floor(ipos)), 0).r * (vol_majorant - vol_minorant);
}

// brick grid (only valid if vol_grid_type == 0)
uniform usampler3D vol_indirection;
uniform sampler3D vol_range;
uniform sampler3D vol_atlas;

// brick grid voxel lookup
float lookup_voxel_brick(const vec3 ipos) {
    const ivec3 iipos = ivec3(floor(ipos));
    const ivec3 brick = iipos >> 3;
    const uvec3 ptr = texelFetch(vol_indirection, brick, 0).xyz;
    const vec2 range = texelFetch(vol_range, brick, 0).xy;
    const float value_unorm = texelFetch(vol_atlas, ivec3(ptr << 3) + (iipos & 7), 0).x;
    return range.x + value_unorm * (range.y - range.x);
}

// brick majorant lookup
float lookup_majorant(const vec3 ipos, int mip) {
    const ivec3 brick = ivec3(floor(ipos)) >> (3 + mip);
    return vol_density_scale * texelFetch(vol_range, brick, mip).y;
}

// density lookup
// TODO better filter on axis-aligned view and high density scale?
float lookup_density(const vec3 ipos, inout uint seed) {
    if (vol_grid_type > 0)
        return vol_density_scale * lookup_voxel_dense(ipos + rng3(seed) - .5f);
    else
        return vol_density_scale * lookup_voxel_brick(ipos + rng3(seed) - .5f);
}

// ---------------------------------
// ray marching based methods

float integrate_density(const vec3 ipos, const vec3 idir, const vec2 near_far, inout uint seed, out float last_interpolant) {
    const int steps = 32;
    const float dt = (near_far.y - near_far.x) / float(steps);
    // first step
    const float t0 = near_far.x + rng(seed) * dt;
    float last_value = lookup_density(ipos + t0 * idir, seed),  tau = 0.f;
    // integrate density
    for (int i = 1; i < steps; ++i) {
        const vec3 curr_pos = ipos + min(t0 + i * dt, near_far.y) * idir;
        const float curr_value = lookup_density(curr_pos, seed);
        last_interpolant = (last_value + curr_value) * 0.5f;
        tau += last_interpolant * dt;
        last_value = curr_value;
    }
    return tau;
}

float integrate_density(const vec3 ipos, const vec3 idir, const vec2 near_far, inout uint seed) {
    float dummy;
    return integrate_density(ipos, idir, near_far, seed, dummy);
}

// TODO DEBUG
float transmittance_raymarch(const vec3 wpos, const vec3 wdir, inout uint seed, const float t_max = FLT_MAX) {
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    near_far.y = min(near_far.y, t_max);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    const float tau = integrate_density(ipos, idir, near_far, seed);
    return exp(-tau);
}

// TODO DEBUG
bool sample_volume_raymarch(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // compute step size and jitter starting point
    const int steps = 32;
    const float dt = (near_far.y - near_far.x) / float(steps);
    const float sampled_tau = -log(1.f - rng(seed));
    // first step
    const float t0 = near_far.x + rng(seed) * dt;
    float last_d = lookup_density(ipos + min(t0, near_far.y) * idir, seed), tau = 0.f;
    // raymarch
    for (int i = 1; i < steps; ++i) {
        const vec3 curr_p = ipos + min(t0 + i * dt, near_far.y) * idir;
        const float curr_d = lookup_density(curr_p, seed);
        const float d = (last_d + curr_d) * 0.5f;
        tau += d * dt;
        last_d = curr_d;
        if (tau >= sampled_tau) {
            const float f = max(0.f, tau - sampled_tau) / d;
            t = t0 + (i - f) * dt;
            throughput *= vol_albedo;
            return true;
        }
    }
    return false;
}

// TODO DEBUG
bool sample_volume_raymarch_pdf(const vec3 wpos, const vec3 wdir, out float t, out float tr_pdf, inout vec3 throughput, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // compute step size and jitter starting point
    const int steps = 32;
    const float dt = (near_far.y - near_far.x) / float(steps);
    const float sampled_tau = -log(1.f - rng(seed));
    t = near_far.x + rng(seed) * dt, tr_pdf = 0.f;
    // raymarch
    float tau = 0.f, density = 0.f;
    for (int i = 0; i < steps; ++i) {
        const vec3 curr_p = ipos + min(t, near_far.y) * idir;
        density = lookup_density(curr_p, seed);
        tau += density * dt;
        t = min(t + dt, near_far.y);
        if (tau >= sampled_tau) {
            // solve for exact collision
            const float f = (tau - sampled_tau) / density;
            t -= f * dt;
            tr_pdf = density * exp(-sampled_tau);
            throughput *= vol_albedo;
            return true;
        }
    }
    t = near_far.y;
    tr_pdf = exp(-tau);
    return false;
}

float pdf_distance(const vec3 wpos, const vec3 wdir, inout uint seed, const float t) {
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 0.f;
    near_far.y = min(near_far.y, t);
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    float last_interpolant;
    const float tau = integrate_density(ipos, idir, near_far, seed, last_interpolant);
    return last_interpolant * exp(-tau);
}

// ---------------------------------
// null-collision methods

float transmittance(const vec3 wpos, const vec3 wdir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // ratio tracking
    float t = near_far.x, Tr = 1.f;
    while (t < near_far.y) {
        t -= log(1 - rng(seed)) * vol_inv_majorant;
        Tr *= max(0.f, 1 - lookup_density(ipos + t * idir, seed) * vol_inv_majorant);
        // russian roulette
        if (Tr < .1f) {
            const float prob = 1 - Tr;
            if (rng(seed) < prob) return 0.f;
            Tr /= 1 - prob;
        }
    }
    return Tr;
}

bool sample_volume(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // delta tracking
    t = near_far.x;
    while (t < near_far.y) {
        t -= log(1 - rng(seed)) * vol_inv_majorant;
        const float d = lookup_density(ipos + t * idir, seed);
        if (rng(seed) * vol_majorant < d) {
            throughput *= vol_albedo;
            return true;
        }
    }
    return false;
}

// ---------------------------------
// DDA-based null-collision methods

// #define USE_TRANSFERFUNC
#define MIP_START 3
#define MIP_SPEED_UP 0.25
#define MIP_SPEED_DOWN 2

// perform DDA step on given mip level
float stepDDA(const vec3 pos, const vec3 inv_dir, const int mip) {
    const float dim = 8 << mip;
    const vec3 offs = mix(vec3(-0.5f), vec3(dim + 0.5f), greaterThanEqual(inv_dir, vec3(0)));
    const vec3 tmax = (floor(pos * (1.f / dim)) * dim + offs - pos) * inv_dir;
    return min(tmax.x, min(tmax.y , tmax.z));
}

// DDA-based transmittance
float transmittanceDDA(const vec3 wpos, const vec3 wdir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 1.f;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = (vec3(vol_inv_model * vec4(wdir, 0))); // non-normalized!
    const vec3 ri = 1.f / idir;
    // march brick grid
    float t = near_far.x + 1e-4f, Tr = 1.f, tau = -log(1.f - rng(seed)), mip = MIP_START;
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
bool sample_volumeDDA(const vec3 wpos, const vec3 wdir, out float t, inout vec3 throughput, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return false;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    const vec3 ri = 1.f / idir;
    // march brick grid
    t = near_far.x + 1e-4f;
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
