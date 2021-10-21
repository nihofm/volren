#version 430

in vec2 tc;
out vec4 out_col;

uniform sampler2D color_prediction;
uniform sampler2D color_reference;
uniform sampler2D color_backprop;

#include "common.glsl"

// ---------------------------------------------------
// helper funcs

float sum(const vec3 x) { return x.x + x.y + x.z; }
float mean(const vec3 x) { return sum(x) * (1.f / 3.f); }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }
vec3 visualize_grad(const float grad) { return abs(grad) * (abs(grad) <= 0.001f ? vec3(0) : (sign(grad) > 0.f ? vec3(1, 0, 0) : vec3(0, 0, 1))); }

// ---------------------------------------------------
// main

void main() {
    out_col = vec4(0, 0, 0, 1);
    if (tc.y < 0.5) {
        if (tc.x < 0.5) {
            // bottom left: transferfunc and gradients visualization
            const vec2 tc_adj = tc * 2;
            if (tc_adj.y < 0.33)
                out_col.rgb = abs(gradients[int(tc_adj.x * n_parameters)].rgb);
                // out_col.rgb = visualize_grad(sum(gradients[int(tc_adj.x * n_parameters)].rgb));
            else if (tc_adj.y < 0.66)
                out_col.rgb = texelFetch(tf_texture, ivec2(tc_adj.x * n_parameters, 0), 0).rgb;
            else
                out_col.rgb = parameters[int(tc_adj.x * n_parameters)].rgb;
            // out_col.rgb = texture(color_backprop, tc_adj).rgb;
        } else {
            // bottom right: l2 grad
            const vec2 tc_adj = vec2((tc.x - 0.5) * 2, tc.y * 2);
            const vec3 col_adj = texture(color_prediction, tc_adj).rgb;
            const vec3 col_ref = texture(color_reference, tc_adj).rgb;
            const vec3 l2_grad = 2 * (col_adj - col_ref);
            out_col.rgb = abs(l2_grad);
            // out_col.rgb = visualize_grad(sum(l2_grad));
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
