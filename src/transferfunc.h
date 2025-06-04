#pragma once

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <tinycolormap.hpp>
#include <cppgl.h>

class TransferFunction {
public:
    TransferFunction();
    TransferFunction(const std::filesystem::path& path);
    TransferFunction(tinycolormap::ColormapType type);
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

    // set to colormap: Parula, Heat, Jet, Turbo, Hot, Gray, Magma, Inferno, Plasma, Viridis, Cividis, Github, Cubehelix, HSV
    void colormap(tinycolormap::ColormapType type, size_t n_bins = 256);

    // load LUT from file
    void load_from_file(const std::filesystem::path& path);

    // write current LUT to (text-)file
    void write_to_file(const std::string& filename);

    // data
    float window_left, window_width;
    std::vector<glm::vec4> lut;
    cppgl::SSBO lut_ssbo;
};
