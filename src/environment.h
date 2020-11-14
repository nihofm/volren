#pragma once

#include <vector>
#include <cppgl.h>
#include <glm/glm.hpp>

class EnvironmentImpl {
public:
    EnvironmentImpl(const std::string& name, const Texture2D& texture);
    virtual ~EnvironmentImpl();

    explicit inline operator bool() const  { return texture->operator bool(); }
    inline operator GLuint() const { return texture->operator GLuint(); }

    // build cdf in given vectors and return integral
    static float build_cdf_1D(const float* f, uint32_t N, std::vector<float>& cdf);
    static float build_cdf_2D(const Texture2D& tex, std::vector<std::vector<float>>& conditional, std::vector<float>& marginal);

    // data
    const std::string name;
    glm::mat4 model;
    Texture2D texture;
    float strength;
    // cdf
    float integral;
    SSBO cdf_U, cdf_V;
};

using Environment = NamedHandle<EnvironmentImpl>;
