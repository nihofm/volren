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

    // clear target reservoir
    reservoirs_flipflop[ridx_target].y_pt = f16vec4(0);
    reservoirs_flipflop[ridx_target].w_sum = 0;
    reservoirs_flipflop[ridx_target].M = 0;

    // gather 3x3x3
    for (int z = -1; z <= 1; ++z) {
        if (voxel.z + z < 0 || voxel.z + z >= vol_size.z) continue;
        for (int y = -1; y <= 1; ++y) {
            if (voxel.y + y < 0 || voxel.y + y >= vol_size.y) continue;
            for (int x = -1; x <= 1; ++x) {
                if (voxel.x + x < 0 || voxel.x + x >= vol_size.x) continue;
                // query source
                const uint ridx_source = ((voxel.z + z) * vol_size.y + (voxel.y + y)) * vol_size.x + voxel.x + x;
                const uint M = reservoirs[ridx_source].M;
                if (M == 0) continue;
                const float pt = reservoirs[ridx_source].y_pt.w;
                if (pt <= 0) continue;
                const float wi = pt * reservoirs[ridx_source].weight() * M;
                // update target TODO FIXME artifacts/biased/unbiased
                reservoirs_flipflop[ridx_target].w_sum += wi;
                reservoirs_flipflop[ridx_target].M += M;
                if (rng(seed) < wi / reservoirs_flipflop[ridx_target].w_sum)
                    reservoirs_flipflop[ridx_target].y_pt = reservoirs[ridx_source].y_pt;
            }
        }
    }
}
