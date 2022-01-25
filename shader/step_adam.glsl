#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 512) in;

#include "common.glsl"

uniform float learning_rate;
uniform float gradient_normalization;
uniform int reset;
uniform int solve;

uniform float param_min;
uniform float param_max;

const float b1 = 0.9;
const float b2 = 0.999;
const float eps = 1e-4;

// ---------------------------------------------------
// main

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= n_parameters) return;

    // debug: reset experiment?
    if (reset > 0) {
         //parameters[idx] = vec4(1, 1, 1, tf_lut[idx].a); // optimize for color
        //parameters[idx] = vec4(tf_lut[idx].rgb, 1);//idx / float(n_parameters)); // optimize for extinction
        // parameters[idx] = vec4(1); // optimize everything
        parameters[idx] = 0.1;
        gradients[idx] = 0;
        first_moments[idx] = 0;
        second_moments[idx] = 1;
        return;
    }

    // debug: solve experiment?
    if (solve > 0) {
        parameters[idx] = lookup_density_brick(vec3(idx % grid_size.x, (idx / grid_size.x) % grid_size.y, idx / (grid_size.x * grid_size.y)));
        gradients[idx] = 0;
        first_moments[idx] = 0;
        second_moments[idx] = 1;
        return;
    }

    // load parameters
    const float x = parameters[idx];
    const float dx = clamp(gradients[idx] * gradient_normalization, -1, 1);
    float m1 = first_moments[idx];
    float m2 = second_moments[idx];

    // update moments
    m1 = b1 * m1 + (1.f - b1) * dx;
    m2 = b2 * m2 + (1.f - b2) * dx * dx;
    // bias correction
    const float m1_c = m1 / (1.f - b1);
    const float m2_c = m2 / (1.f - b2);
    // update parameter
    const float y = clamp(x - learning_rate * m1_c / (sqrt(m2_c) + eps), param_min + eps, param_max);

    // store updated parameters and zero gradients
    parameters[idx] = y;
    gradients[idx] = 0;
    first_moments[idx] = m1;
    second_moments[idx] = m2;
}
