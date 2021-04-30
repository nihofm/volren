#pragma once

#include <vector>
#include <cppgl.h>
#include <glm/glm.hpp>

class EnvironmentImpl {
public:
    EnvironmentImpl(const std::string& name, const Texture2D& envmap);
    virtual ~EnvironmentImpl();

    explicit inline operator bool() const  { return envmap->operator bool() && impmap->operator bool(); }

    int num_mip_levels() const;
    void set_uniforms(const Shader& shader, uint32_t& texture_unit) const;

    // data
    const std::string name;
    glm::mat3 model;
    float strength;
    Texture2D envmap, impmap;
};

using Environment = NamedHandle<EnvironmentImpl>;
