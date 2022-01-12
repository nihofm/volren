#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "buffer.cuh"
#include "aabb.cuh"
#include "ray.cuh"
#include "rng.cuh"

// ------------------------------------------
// Dense Grid

struct DenseGridCUDA {
    DenseGridCUDA() : to_world(1), to_local(1), bounds(), minorant(FLT_MAX), majorant(FLT_MIN), grid() {}
    DenseGridCUDA(const glm::mat4& transform, const float max_value, BufferCUDA<float> grid) {
        to_world = transform;
        to_local = glm::inverse(transform);
        bounds.bb_min = glm::vec3(transform * glm::vec4(0, 0, 0, 1));
        bounds.bb_max = glm::vec3(transform * glm::vec4(grid.size.x, grid.size.y, grid.size.z, 1));
        minorant = 0.f;
        majorant = max_value;
        this->grid = grid;
    }

    __host__ __device__ inline float lookup(const glm::uvec3& ipos) {
        return grid[glm::clamp(ipos, glm::uvec3(0), grid.size - 1u)];
    }
    __host__ __device__ inline float lookup(const glm::uvec3& ipos, uint32_t& seed) {
        const glm::uvec3 jpos = glm::uvec3(glm::floor(glm::vec3(ipos) + rng3(seed) - .5f));
        return grid[glm::clamp(jpos, glm::uvec3(0), grid.size - 1u)];
    }

    __host__ __device__ inline bool intersect(const Ray& ray, glm::vec2& near_far) {
        return bounds.intersect(ray, near_far);
    }

    __host__ __device__ inline float transmittance(const Ray& ray, uint32_t& seed) {
        glm::vec2 near_far;
        if (!bounds.intersect(ray, near_far)) return 1.f;
        // to index-space
        const glm::vec3 ipos = glm::vec3(to_local * glm::vec4(ray.pos, 1));
        const glm::vec3 idir = glm::vec3(to_local * glm::vec4(ray.dir, 0));
        // ratio tracking
        float Tr = 1.f;
        float t = near_far.x - log(1.f - rng(seed)) / majorant;
        while (t < near_far.y) {
            const float d = lookup(ipos + t * idir, seed);
            Tr *= 1.f - d / majorant;
            // russian roulette
            if (Tr < 1.f) {
                const float prob = 1 - Tr;
                if (rng(seed) < prob) return 0.f;
                Tr /= 1 - prob;
            }
            // advance
            t -= log(1.f - rng(seed)) / majorant;
        }
        return Tr;
    }

    __host__ __device__ inline bool sample(const Ray& ray, float& t, uint32_t& seed) {
        glm::vec2 near_far;
        if (!bounds.intersect(ray, near_far)) return false;
        // to index-space
        const glm::vec3 ipos = glm::vec3(to_local * glm::vec4(ray.pos, 1));
        const glm::vec3 idir = glm::vec3(to_local * glm::vec4(ray.dir, 0));
        // delta tracking
        t = near_far.x - log(1.f - rng(seed)) / majorant;
        while (t < near_far.y) {
            const float d = lookup(ipos + t * idir, seed);
            if (rng(seed) * majorant < d)
                return true;
            // advance
            t -= log(1.f - rng(seed)) / majorant;
        }
        return false;
    }

    // data
    glm::mat4 to_world, to_local;   // transform from index- to world-space and vice-versa
    AABB bounds;                    // axis-aligned world-space bounds
    float minorant, majorant;       // minorant and majorant of density
    BufferCUDA<float> grid;         // dense grid data
};