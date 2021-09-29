#version 430

in vec2 tc;
out vec4 out_col;

uniform sampler2D color_prediction;
uniform sampler2D color_reference;
uniform sampler2D color_backprop;
// uniform sampler3D vol_gradients;

// TODO SSBO

uniform int seed;
uniform int sppx;
uniform int current_sample;
uniform ivec2 resolution;

#include "common.glsl"

// ---------------------------------------------------
// helper funcs

float sum(const vec3 x) { return x.x + x.y + x.z; }
float mean(const vec3 x) { return sum(x) * (1.f / 3.f); }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }
vec3 visualize_grad(const float grad) { return abs(grad) * (abs(grad) <= 0.001f ? vec3(0) : (sign(grad) > 0.f ? vec3(1, 0, 0) : vec3(0, 0, 1))); }

float lookup_grad(const vec3 ipos) {
    const ivec3 iipos = ivec3(floor(ipos));
    const uint idx = iipos.z * vol_size.x * vol_size.y + iipos.y * vol_size.x + iipos.x;
    return parameters[idx].y;
}

float raymarch_gradients(const vec3 wpos, const vec3 wdir, inout uint seed) {
    // clip volume
    vec2 near_far;
    if (!intersect_box(wpos, wdir, vol_bb_min, vol_bb_max, near_far)) return 0.f;
    // to index-space
    const vec3 ipos = vec3(vol_inv_model * vec4(wpos, 1));
    const vec3 idir = vec3(vol_inv_model * vec4(wdir, 0)); // non-normalized!
    // raymarch
    const float step_size = max(0.01f, (near_far.y - near_far.x) / 8.f);
    float t = near_far.x + rng(seed) * step_size;
    float grad = 0.f;
    while (t < near_far.y) {
        grad += step_size * lookup_grad(ipos + t * idir);
        t += step_size;
    }
    return grad;
}

// ---------------------------------------------------
// main

void main() {
    out_col = vec4(0, 0, 0, 1);
    if (tc.y < 0.5) {
        if (tc.x < 0.5) {
            // bottom left: backprop
            // out_col.rgb = abs(texture(color_backprop, tc * 2).rgb); return;
            // bottom left: gradient visualization
            const ivec2 pixel_adj = ivec2(gl_FragCoord.xy * 2);
            uint seed = tea(seed * (pixel_adj.y * resolution.x + pixel_adj.x), current_sample, 32);
            const vec3 pos = cam_pos;
            const vec3 dir = view_dir(pixel_adj, resolution, rng2(seed));
            out_col.rgb = visualize_grad(raymarch_gradients(pos, dir, seed) * 1e-4);
        } else {
            // bottom right: l2 grad
            const vec2 tc_adj = vec2((tc.x - 0.5) * 2, tc.y * 2);
            const vec3 col_adj = texture(color_prediction, tc_adj).rgb;
            const vec3 col_ref = texture(color_reference, tc_adj).rgb;
            const vec3 l2_grad = 2 * (col_adj - col_ref);
            out_col.rgb = visualize_grad(sum(l2_grad));
            // out_col.rgb = abs(texture(color_backprop, tc_adj).rgb - col_adj);
        }
    } else {
        if (tc.x < 0.5) {
            // top left: prediction
            const vec2 tc_adj = vec2(tc.x * 2, (tc.y - 0.5) * 2);
            out_col.rgb = texture(color_prediction, tc_adj).rgb;
        } else {
            // top right: reference
            const vec2 tc_adj = vec2((tc.x - 0.5) * 2, (tc.y - 0.5) * 2);
            out_col.rgb = texture(color_reference, tc_adj).rgb;
        }
    }
}
