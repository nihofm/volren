#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <cppgl/texture.h>

class TransferFunctionImpl {
public:
    TransferFunctionImpl();
    TransferFunctionImpl(const fs::path& path);
    TransferFunctionImpl(const std::vector<glm::vec4>& lut);
    virtual ~TransferFunctionImpl();

    // push cpu LUT data to GPU texture
    void upload_gpu();

    // data
    float window_center, window_width;
    std::vector<glm::vec4> lut;
    Texture2D texture;
};

using TransferFunction = NamedHandle<TransferFunctionImpl>;