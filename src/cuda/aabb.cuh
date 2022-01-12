#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "ray.cuh"

class AABB {
public:
    __host__ __device__ AABB() : bb_min(FLT_MIN), bb_max(FLT_MAX) {}
    __host__ __device__ AABB(const glm::vec3& bb_min, const glm::vec3& bb_max) : bb_min(bb_min), bb_max(bb_max) {}

    inline __host__ __device__ void include(const glm::vec3& point) {
        bb_min = glm::min(bb_min, point);
        bb_max = glm::max(bb_max, point);
    }

    inline __host__ __device__ void include(const AABB& other) {
        bb_min = glm::min(bb_min, other.bb_min);
        bb_max = glm::max(bb_max, other.bb_max);
    }

    inline __host__ __device__ void intersect(const AABB& other) {
        bb_min = glm::max(bb_min, other.bb_min);
        bb_max = glm::min(bb_max, other.bb_max);
    }

    inline __host__ __device__ bool intersect(const Ray& ray, glm::vec2& near_far) {
        const glm::vec3 inv_dir = 1.f / ray.dir;
        const glm::vec3 lo = (bb_min - ray.pos) * inv_dir;
        const glm::vec3 hi = (bb_max - ray.pos) * inv_dir;
        const glm::vec3 tmin = min(lo, hi), tmax = max(lo, hi);
        near_far.x = fmaxf(ray.near, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z)));
        near_far.y = fminf(ray.far , fminf(tmax.x, fminf(tmax.y, tmax.z)));
        return near_far.x < near_far.y;
    }

    // data
    glm::vec3 bb_min, bb_max;
};

class Sphere {
public:
    __host__ __device__ Sphere() : origin(0), radius(0) {}
    __host__ __device__ Sphere(const glm::vec3& origin, const float radius) : origin(origin), radius(radius) {}

    inline __host__ __device__ void include(const glm::vec3& point) {
        radius = fminf(radius, glm::distance(origin, point));
    }

    inline __host__ __device__ void include(const Sphere& other) {
        radius = fminf(radius, glm::distance(origin, other.origin) + other.radius);
    }

    inline __host__ __device__ bool intersect(const Ray& ray, glm::vec2& near_far) {
        const glm::vec3 pdir = ray.pos - origin;
        const float b = glm::dot(ray.dir, pdir);
        const float c = glm::dot(pdir, pdir) - radius * radius;
        const float h = sqrtf(b * b - c);
        near_far.x = fmaxf(0.f, -b - h);
        near_far.y = fmaxf(0.f, -b + h);
        return near_far.x < near_far.y;
    }

    // data
    glm::vec3 origin;
    float radius;
};