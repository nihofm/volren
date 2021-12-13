#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 512) in;

#include "common.glsl"

uniform float learning_rate;
uniform float gradient_normalization;
uniform int reset;
uniform int solve;

const float b1 = 0.9;
const float b2 = 0.999;
const float eps = 1e-8;

// ---------------------------------------------------
// main

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= n_parameters) return;

    // debug: reset experiment?
    if (reset > 0) {
        parameters[idx] = vec4(1, 1, 1, tf_lut[idx].a);
        gradients[idx] = vec4(0);
        first_moments[idx] = vec4(0);
        second_moments[idx] = vec4(1);
        return;
    }

    // debug: solve experiment?
    if (solve > 0) {
        parameters[idx] = tf_lut[idx];
        gradients[idx] = vec4(0);
        first_moments[idx] = vec4(0);
        second_moments[idx] = vec4(1);
        return;
    }

    // load parameters
    const vec4 x = parameters[idx];
    const vec4 dx = clamp(gradients[idx] * gradient_normalization, vec4(-1), vec4(1));
    vec4 m1 = first_moments[idx];
    vec4 m2 = second_moments[idx];

    // update moments
    m1 = b1 * m1 + (1.f - b1) * dx;
    m2 = b2 * m2 + (1.f - b2) * dx * dx;
    // bias correction
    const vec4 m1_c = m1 / (1.f - b1);
    const vec4 m2_c = m2 / (1.f - b2);
    // update parameter
    const vec4 y = clamp(x - learning_rate * m1_c / (sqrt(m2_c) + eps), 0.0 + eps, 1.0);

    // store updated parameters and zero gradients
    parameters[idx] = y;
    gradients[idx] = vec4(0);
    first_moments[idx] = m1;
    second_moments[idx] = m2;

    /*
    // TODO: ensure monotonic function
    barrier();
    const float lower = parameters[max(0, idx-1)].a;
    const float upper = parameters[min(idx+1, n_parameters-1)].a;
    barrier();
    parameters[idx].a = clamp(parameters[idx].a, lower, upper);
    */
}
