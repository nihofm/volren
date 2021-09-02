#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, r32f) uniform image3D volume;
layout (binding = 1, r32f) uniform image3D gradients;

uniform float learning_rate;
uniform float vol_majorant;
uniform ivec3 size;

// ---------------------------------------------------
// main

void main() {
	const ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(gid, size))) return;

    // gradient descent
    const float x = imageLoad(volume, gid).x;
    const float dx = clamp(imageLoad(gradients, gid).x, -1, 1);
    // TODO adam optimizer
    const float y = clamp(x - learning_rate * dx, 0.f, vol_majorant);

    // write updated grid and zero gradients
    imageStore(volume, gid, vec4(y));
    imageStore(gradients, gid, vec4(0.f));
}
