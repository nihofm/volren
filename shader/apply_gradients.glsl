#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, r32f) uniform image3D volume;
layout (binding = 1, r32f) uniform image3D gradients;
layout (binding = 2, rg32f) uniform image3D adam;

uniform float learning_rate;
uniform float vol_majorant;
uniform ivec3 size;

const float b1 = 0.9f;
const float b2 = 0.999f;
const float eps = 1e-8f;

// ---------------------------------------------------
// main

void main() {
	const ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(gid, size))) return;

    // gradient descent
    const float x = imageLoad(volume, gid).x;
    const float dx = clamp(imageLoad(gradients, gid).x, -1, 1);
    // Adam optimizer
    vec2 m1_m2 = imageLoad(adam, gid).xy;
    m1_m2.x = b1 * m1_m2.x + (1.f - b1) * dx;
    m1_m2.y = b2 * m1_m2.y + (1.f - b2) * dx * dx;
    // bias correction
    const float m1_c = m1_m2.x / (1.f - b1);
    const float m2_c = m1_m2.y / (1.f - b2);
    // apply update
    const float y = clamp(x - learning_rate * m1_c / (sqrt(m2_c) + eps), 0.f, vol_majorant);

    // write updated grid and zero gradients
    imageStore(volume, gid, vec4(y));
    imageStore(gradients, gid, vec4(0.f));
    imageStore(adam, gid, vec4(m1_m2, 0.f, 0.f));
}
