#version 450 core

#include "common.h"

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// ---------------------------------------------------
// main

uniform int current_sample;

void main() {
	const ivec3 voxel = ivec3(gl_GlobalInvocationID.xyz);
	if (any(greaterThanEqual(voxel, vol_size))) return;
    const uint ridx_target = (voxel.z * vol_size.y + voxel.y) * vol_size.x + voxel.x;
    uint seed = tea(current_sample, ridx_target, 16);

    // copy target reservoir
    reservoirs_flipflop[ridx_target].y_pt = reservoirs[ridx_target].y_pt;
    reservoirs_flipflop[ridx_target].w_sum = reservoirs[ridx_target].w_sum;
    reservoirs_flipflop[ridx_target].M = reservoirs[ridx_target].M;

    // dropout?
    if (rng(seed) < .0) {
        reservoirs_flipflop[ridx_target].y_pt = f16vec4(0);
        reservoirs_flipflop[ridx_target].w_sum = 0;
        reservoirs_flipflop[ridx_target].M = 0;
        return;
    }

    // gather randomly from neighborhood
    const int K = 3;
    const float radius = 10;
    for (int i = 0; i < K; ++i) {
        // select reservoir
        const ivec3 dir = ivec3(sample_phase_isotropic(rng2(seed)) * (1 + rng(seed) * radius));
        const uint ridx_source = ((voxel.z + dir.z) * vol_size.y + (voxel.y + dir.y)) * vol_size.x + voxel.x + dir.x;
        if (ridx_source == ridx_target) continue;
        // combine reservoirs
        const uint M = reservoirs[ridx_source].M;
        if (M <= 0) continue;
        const float pt = reservoirs[ridx_source].y_pt.w;
        if (pt <= 0) continue;
        const float wi = pt * reservoirs[ridx_source].weight() * M;
        // update target
        reservoirs_flipflop[ridx_target].w_sum += wi;
        reservoirs_flipflop[ridx_target].M += M;
        if (rng(seed) < wi / reservoirs_flipflop[ridx_target].w_sum)
            reservoirs_flipflop[ridx_target].y_pt = reservoirs[ridx_source].y_pt;
    }
}
