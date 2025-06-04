#include "transferfunc.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

TransferFunction::TransferFunction() : window_left(0), window_width(1) {
    randomize();
}

TransferFunction::TransferFunction(const std::filesystem::path& path) : TransferFunction() {
    load_from_file(path);
}

TransferFunction::TransferFunction(tinycolormap::ColormapType type) : TransferFunction() {
    colormap(type);
}

TransferFunction::TransferFunction(const std::vector<glm::vec4>& lut) : TransferFunction() {
    this->lut = lut;
    upload_gpu();
}

TransferFunction::~TransferFunction() {}

void TransferFunction::set_uniforms(const cppgl::Shader& shader, uint32_t buffer_binding) const {
    lut_ssbo->bind_base(buffer_binding);
    shader->uniform("tf_size", uint32_t(lut_ssbo->size_bytes / sizeof(glm::vec4)));
    shader->uniform("tf_window_left", window_left);
    shader->uniform("tf_window_width", window_width);
}

std::vector<glm::vec4> TransferFunction::compute_lut_cdf(const std::vector<glm::vec4>& lut) {
    // copy
    auto lut_cdf = lut;
    // build density CDF to ensure alpha monotonically nondecreasing
    for (uint32_t i = 1; i < lut_cdf.size(); ++i)
        lut_cdf[i].a += lut_cdf[i-1].a;
    const float integral = lut_cdf[lut_cdf.size()-1].a;
    for (uint32_t i = 0; i < lut_cdf.size(); ++i)
        lut_cdf[i].a = integral <= 0.f ? (i + 1) / float(lut_cdf.size()) : lut_cdf[i].a / integral;
    return lut_cdf;
}

void TransferFunction::upload_gpu() {
    // check if lut is monotonically nondecreasing (hard requirement of DDA optimization)
    bool needs_cdf = false;
    for (uint i = 1; i < lut.size(); ++i) {
        if (lut[i-1].a > lut[i].a) {
            needs_cdf = true;
            break;
        }
    }
    // setup and fill lut SSBO
    lut_ssbo = cppgl::SSBO("transferfunc_ssbo");
    const auto lut_gpu = needs_cdf ? compute_lut_cdf(lut) : lut;
    lut_ssbo->upload_data(lut_gpu.data(), lut_gpu.size() * sizeof(glm::vec4));
}

static inline float randf() { return rand() / (RAND_MAX + 1.f); }

void TransferFunction::randomize(size_t n_bins) {
    lut.clear();
    for (int i = 0; i < n_bins; ++i)
        lut.push_back(i == 0 ? glm::vec4(0) : glm::vec4(randf(), randf(), randf(), randf()));
    upload_gpu();
}

void TransferFunction::colormap(tinycolormap::ColormapType type, size_t n_bins) {
    lut.clear();
    for (int i = 0; i < n_bins; ++i) {
        const float f = float(i) / n_bins;
        const tinycolormap::Color color = tinycolormap::GetColor(f, type);
        lut.push_back(glm::vec4(color.r(), color.g(), color.b(), f));
    }
    upload_gpu();
}

void TransferFunction::load_from_file(const std::filesystem::path& path) {
    // load lut from file (format: %f, %f, %f, %f)
    std::ifstream lut_file(path);
    if (!lut_file.is_open())
        throw std::runtime_error("Unable to read file: " + path.string());
    lut.clear();
    std::cout << "Loading LUT: " << path << std::endl;
    char tmp[256];
    while (lut_file.getline(tmp, 256)) {
        float r, g, b, a;
        sscanf(tmp, "%f, %f, %f, %f", &r, &g, &b, &a);
        lut.emplace_back(r, g, b, a);
    }
    upload_gpu();
}

void TransferFunction::write_to_file(const std::string& filename) {
    std::filesystem::path filepath = filename;
    filepath.replace_extension(".txt"); // ensure text-based file
    std::ofstream file(filepath);
    if (file.is_open()) {
        char tmp[256];
        for (const auto& rgba : lut) {
            snprintf(tmp, 256, "%f, %f, %f, %f", rgba.x, rgba.y, rgba.z, rgba.a);
            file << tmp << std::endl;
        }
    }
}
