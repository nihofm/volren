#pragma once

#include <vector>
#include <cppgl.h>
#include <glm/glm.hpp>

class TransferFunction {
public:
    TransferFunction();
    TransferFunction(const fs::path& path);
    TransferFunction(const std::vector<glm::vec4>& lut);
    virtual ~TransferFunction();

    void set_uniforms(const Shader& shader, uint32_t& texture_unit) const;

    // compute density-CDF lut from given lut
    static std::vector<glm::vec4> compute_lut_cdf(const std::vector<glm::vec4>& lut);

    // push cdf lut data to GPU texture
    void upload_gpu();

    // data
    float window_left, window_width;
    std::vector<glm::vec4> lut;
    Texture2D texture;
};
