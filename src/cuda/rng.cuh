#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

// tiny encryption algorithm (TEA) to calculate a seed per launch index and iteration
__host__ __device__ inline uint32_t tea(const uint32_t val0, const uint32_t val1, const uint32_t N) {
    uint32_t v0 = val0;
    uint32_t v1 = val1;
    uint32_t s0 = 0;
    for (uint32_t n = 0; n < N; ++n) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
        v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
    }
    return v0;
}

// return a random sample in the range [0, 1) with a simple linear congruential generator
__host__ __device__ inline float rng(uint32_t& previous) {
    previous = previous * 1664525u + 1013904223u;
    return float(previous & 0x00FFFFFFu) / float(0x01000000u);
}

__host__ __device__ inline glm::vec2 rng2(uint32_t& previous) {
    return glm::vec2(rng(previous), rng(previous));
}

__host__ __device__ inline glm::vec3 rng3(uint32_t& previous) {
    return glm::vec3(rng(previous), rng(previous), rng(previous));
}

__host__ __device__ inline glm::vec4 rng4(uint32_t& previous) {
    return glm::vec4(rng(previous), rng(previous), rng(previous), rng(previous));
}