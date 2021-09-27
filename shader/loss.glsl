#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform ivec3 size;

layout(std430, binding = 0) buffer Buffer {
    float loss[];
};

#include "common.glsl"

// ---------------------------------------------------
// main

void main() {
	const ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(gid, size))) return;

    const float A = lookup_voxel_dense(gid);
    const float B = lookup_voxel_brick(gid);
    atomicAdd(loss[0], abs(A - B));
}
