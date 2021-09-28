#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout (binding = 0, r32f) uniform image3D volume;
layout (binding = 1, r32f) uniform image3D grads;
layout (binding = 2, rg32f) uniform image3D adam;

layout(std430, binding = 0) buffer Buffer0 {
    vec4 gradients[]; // vec4(grad, N, m1, m2)
};

uniform ivec3 size;
uniform int reset;

// ---------------------------------------------------
// main

void main() {
	const ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(gid, size))) return;
    const uint idx = gid.z * size.x * size.y + gid.y * size.x + gid.x;

    // zero gradients
    imageStore(grads, gid, vec4(0.f));
    gradients[idx].x = 0.f;
    gradients[idx].y = 0.f;

    // reset?
    if (reset > 0) {
        imageStore(volume, gid, vec4(1.f));
        imageStore(adam, gid, vec4(0, 1, 0, 0));
        gradients[idx].z = 0.f;
        gradients[idx].w = 1.f;
    }
}
