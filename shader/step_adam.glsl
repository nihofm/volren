#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout (binding = 0, r32f) uniform image3D volume;
layout (binding = 1, r32f) uniform image3D grads;
layout (binding = 2, rg32f) uniform image3D adam;

layout(std430, binding = 0) buffer Buffer0 {
    vec4 gradients[]; // vec4(grad, N, m1, m2)
};

uniform float learning_rate;
uniform float vol_majorant;
uniform ivec3 size;

const float b1 = 0.9;
const float b2 = 0.999;
const float eps = 1e-8;

// ---------------------------------------------------
// main

void main() {
	const ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(gid, size))) return;

    // gradient descent
    const float x = imageLoad(volume, gid).x;
    const float dx = clamp(imageLoad(grads, gid).x, -1, 1);
    // Adam optimizer
    vec2 m1_m2 = imageLoad(adam, gid).xy;
    m1_m2.x = b1 * m1_m2.x + (1.f - b1) * dx;
    m1_m2.y = b2 * m1_m2.y + (1.f - b2) * dx * dx;
    // bias correction
    const float m1_c = m1_m2.x / (1.f - b1);
    const float m2_c = m1_m2.y / (1.f - b2);
    // apply update
    const float y = clamp(x - learning_rate * m1_c / (sqrt(m2_c) + eps), eps, vol_majorant - eps);

    // write updated grid and adam params
    imageStore(volume, gid, vec4(y));
    imageStore(adam, gid, vec4(m1_m2, 0.f, 0.f));
}
