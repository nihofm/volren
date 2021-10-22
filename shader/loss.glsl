#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

#include "common.glsl"

layout(std430, binding = 1) buffer LossBuffer {
    float loss[];
};

// ---------------------------------------------------
// main

void main() {
	const uint idx = gl_GlobalInvocationID.x;
    if (idx >= n_parameters) return;

    const vec4 x = parameters[idx];
    const vec4 y = tf_lut[idx];
    const vec4 diff = abs(x - y);
    const float L = diff.x + diff.y + diff.z + diff.w;
    atomicAdd(loss[0], L);
}
