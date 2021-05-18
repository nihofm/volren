#pragma once

#include <vector>
#include <cppgl.h>
#include <glm/glm.hpp>

class TransferFunctionImpl {
public:
    TransferFunctionImpl(const std::string& name);
    TransferFunctionImpl(const std::string& name, const fs::path& path);
    TransferFunctionImpl(const std::string& name, const std::vector<glm::vec4>& lut);
    virtual ~TransferFunctionImpl();

    void set_uniforms(const Shader& shader, uint32_t& texture_unit) const;

    // push cpu LUT data to GPU texture
    void upload_gpu();

    // data
    const std::string name;
    float window_left, window_width;
    std::vector<glm::vec4> lut;
    Texture2D texture;
};

using TransferFunction = NamedHandle<TransferFunctionImpl>;
