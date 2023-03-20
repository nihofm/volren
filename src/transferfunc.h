#pragma once

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <cppgl.h>

class TransferFunction {
public:
    TransferFunction();
    TransferFunction(const fs::path& path);
    TransferFunction(const std::vector<glm::vec4>& lut);
    virtual ~TransferFunction();

    // bind uniforms to given shader
    void set_uniforms(const cppgl::Shader& shader, uint32_t buffer_binding) const;

    // compute density-CDF lut from given lut
    static std::vector<glm::vec4> compute_lut_cdf(const std::vector<glm::vec4>& lut);

    // push cdf lut data to GPU texture
    void upload_gpu();

    // randomize contents
    void randomize(size_t n_bins = 8);

    // write current LUT to (text-)file
    void write_to_file(const std::string& filename);

    // data
    float window_left, window_width;
    std::vector<glm::vec4> lut;
    cppgl::SSBO lut_ssbo;
};
