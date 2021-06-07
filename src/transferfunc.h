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

    // push cpu LUT data to GPU texture
    void upload_gpu();

    // data
    float window_left, window_width;
    std::vector<glm::vec4> lut;
    Texture2D texture;
};
