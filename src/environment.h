#pragma once

#include <vector>
#include <cppgl/texture.h>
#include <cppgl/ssbo.h>
#include <glm/glm.hpp>

class Environment {
public:
    Environment(const std::string& name, const std::shared_ptr<Texture2D>& texture);
    virtual ~Environment();

    explicit inline operator bool() const  { return texture->operator bool(); }
    inline operator GLuint() const { return texture->operator GLuint(); }

    // build cdf in given vectors and return integral
    static float build_cdf_1D(const float* f, uint32_t N, std::vector<float>& cdf);
    static float build_cdf_2D(const std::shared_ptr<Texture2D>& tex, std::vector<std::vector<float>>& conditional, std::vector<float>& marginal);

    // data
    glm::mat4 model;
    std::shared_ptr<Texture2D> texture;
    // cdf
    float integral;
    std::shared_ptr<SSBO> cdf_U, cdf_V;
};

// variadic alias for std::make_shared<>(...)
template <class... Args> std::shared_ptr<Environment> make_environment(Args&&... args) {
    return std::make_shared<Environment>(args...);
}
