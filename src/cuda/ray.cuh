#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

// ------------------------------------------
// Ray

struct Ray {
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const glm::vec3& origin, const glm::vec3& direction, float near = 0.f, float far = 1e+38f) : pos(origin), near(near), dir(direction), far(far) {}

    __host__ __device__ inline glm::vec3 operator()(float t) const {
        return pos + t * dir;
    }

    __host__ __device__ inline Ray clip(const glm::vec2& near_far) const {
        return Ray(pos, dir, fmaxf(near, near_far.x), fminf(far , near_far.y));
    }

    __host__ __device__ inline Ray transform(const glm::mat4& mat) const {
        return Ray(glm::vec3(mat * glm::vec4(pos, 1)), glm::mat3(mat) * dir, near, far);
    }

    __host__ __device__ inline Ray transform_normalize(const glm::mat4& mat) const {
        const glm::vec3 tpos = glm::vec3(mat * glm::vec4(pos, 1));
        const glm::vec3 tdir = glm::mat3(mat) * dir;
        const float scale = glm::length(tdir);
        return Ray(tpos, tdir / scale, near * scale, far * scale);
    }

    // data
    glm::vec3 pos;
    float near;
    glm::vec3 dir;
    float far;
};