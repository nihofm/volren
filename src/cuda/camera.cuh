#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "ray.cuh"

// ------------------------------------------
// Camera CUDA

struct CameraCUDA {
    __host__ __device__ CameraCUDA() {}
    __host__ __device__ CameraCUDA(const glm::vec3& pos, const glm::vec3& dir, float fovy) : pos(pos), dir(dir), fovy(fovy), flags(0) {}

    inline __host__ __device__ glm::vec3 to_world(const glm::vec3& v) const {
        const glm::vec3 ndir = glm::normalize(dir);
        const glm::vec3 left = glm::normalize(glm::cross(ndir, glm::vec3(1e-7f, 1, 1e-6f)));
        const glm::vec3 up = glm::cross(left, ndir);
        return glm::normalize(v.x * left + v.y * up + v.z * -dir);
    }

    inline __host__ __device__ glm::vec3 to_tangent(const glm::vec3& v) const {
        const glm::vec3 ndir = glm::normalize(dir);
        const glm::vec3 left = glm::cross(ndir, glm::vec3(1e-7f, 1, 1e-6f));
        const glm::vec3 up = glm::cross(left, ndir);
        return glm::vec3(dot(v, left), dot(v, up), dot(v, -dir));
    }

    inline __host__ __device__ Ray view_ray_pinhole(const glm::vec2& pixel, const glm::vec2& wh, const glm::vec2& pixel_sample = glm::vec2(.5f)) const {
        const glm::vec2 xy = (pixel + pixel_sample - wh * .5f) / wh.y;
        const float z = -.5f / tanf(.5f * M_PI * fovy / 180.f);
        return Ray(pos, to_world(glm::vec3(xy.x, xy.y, z)));
    }

    // data
    glm::vec3 pos;  // camera position
    glm::vec3 dir;  // view direction
    float fovy;     // fov in y-direction in degrees
    uint32_t flags; // TODO: flags
};