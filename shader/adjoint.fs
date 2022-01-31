#version 430

in vec2 tc;
out vec4 out_col;

uniform sampler2D color_prediction;
uniform sampler2D color_reference;
uniform sampler2D color_backprop;

#include "common.glsl"

// ---------------------------------------------------
// helper funcs

vec3 visualize_grad(const float grad) { return abs(grad) * (abs(grad) <= 0.001f ? vec3(0) : (sign(grad) > 0.f ? vec3(1, 0, 0) : vec3(0, 0, 1))); }
/*
vec3 visualize_tf(const vec2 tc, bool use_ref = true) {
    const float xi = tc.x * tf_size - 0.5;
    const int x0 = max(0, int(floor(xi)));
    const int x1 = min(int(ceil(xi)), int(tf_size)-1);
    const vec4 rgba = use_ref ? mix(tf_lut[x0], tf_lut[x1], fract(xi)) : mix(parameters[x0], parameters[x1], fract(xi));
    // return rgba.rgb * rgba.a;
    const vec3 color = smoothstep(vec3(0.1), vec3(0), abs(tc.y - rgba.rgb));
    const float alpha = smoothstep(0.1, 0.0, tc.y - rgba.a) * 0.5;
    return color + alpha;
}
*/

// ---------------------------------------------------
// main

void main() {
    out_col = vec4(0, 0, 0, 1);
    if (tc.y < 0.5) {
        if (tc.x < 0.5) {
            // bottom left: gradients visualization
            const vec2 tc_adj = tc * 2;
            // out_col.rgb += texture(color_backprop, tc_adj).rgb;
            // bottom left: negative l2 grad
            const vec3 col_adj = texture(color_prediction, tc_adj).rgb;
            const vec3 col_ref = texture(color_reference, tc_adj).rgb;
            const vec3 l2_grad = 2 * (col_adj - col_ref);
            out_col.rgb = -l2_grad;
            out_col.rgb = -texture(color_backprop, tc_adj).rgb;

        } else {
            // bottom right: positive l2 grad
            const vec2 tc_adj = vec2((tc.x - 0.5) * 2, tc.y * 2);
            const vec3 col_adj = texture(color_prediction, tc_adj).rgb;
            const vec3 col_ref = texture(color_reference, tc_adj).rgb;
            const vec3 l2_grad = 2 * (col_adj - col_ref);
            out_col.rgb = l2_grad;
            // out_col.rgb = visualize_grad(mean(l2_grad));
            out_col.rgb = texture(color_backprop, tc_adj).rgb;
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
