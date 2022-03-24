#pragma once

#include <vector>
#include <cppgl.h>
#include <glm/glm.hpp>

class Environment {
public:
    Environment(const std::string& path);
    Environment(const cppgl::Texture2D& envmap);
    virtual ~Environment();

    explicit inline operator bool() const  { return envmap->operator bool() && impmap->operator bool(); }

    uint32_t num_mip_levels() const;
    uint32_t dimension() const;
    void set_uniforms(const cppgl::Shader& shader, uint32_t& texture_unit) const;

    // data
    glm::mat3 model;
    float strength;
    cppgl::Texture2D envmap, impmap;
};
