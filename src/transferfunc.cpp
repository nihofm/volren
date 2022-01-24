#include "transferfunc.h"
#include <fstream>
#include <iostream>

TransferFunction::TransferFunction() : window_left(0), window_width(1) {}

TransferFunction::TransferFunction(const fs::path& path) : TransferFunction() {
    // load lut from file (format: %f, %f, %f, %f)
    std::ifstream lut_file(path);
    if (!lut_file.is_open())
        throw std::runtime_error("Unable to read file: " + path.string());
    std::cout << "Loading LUT: " << path << std::endl;
    char tmp[256];
    while (lut_file.getline(tmp, 256)) {
        float r, g, b, a;
        sscanf(tmp, "%f, %f, %f, %f", &r, &g, &b, &a);
        lut.emplace_back(r, g, b, a);
    }
    upload_gpu();
}

TransferFunction::TransferFunction(const std::vector<glm::vec4>& lut) : TransferFunction() {
    this->lut = lut;
    upload_gpu();
}

TransferFunction::~TransferFunction() {}

void TransferFunction::set_uniforms(const Shader& shader, uint32_t& texture_unit, uint32_t buffer_binding) const {
    lut_ssbo->bind_base(buffer_binding);
    shader->uniform("tf_size", uint32_t(lut_ssbo->size_bytes / sizeof(glm::vec4)));
    shader->uniform("tf_window_left", window_left);
    shader->uniform("tf_window_width", window_width);
}

std::vector<glm::vec4> TransferFunction::compute_lut_cdf(const std::vector<glm::vec4>& lut) {
    //return lut; // TODO XXX debug
    // copy
    auto lut_cdf = lut;
    // build density CDF (ensure a monotonic nondecreasing function)
    for (uint32_t i = 1; i < lut_cdf.size(); ++i)
        lut_cdf[i].a += lut_cdf[i-1].a;
    const float integral = lut_cdf[lut_cdf.size()-1].a;
    for (uint32_t i = 0; i < lut_cdf.size(); ++i)
        lut_cdf[i].a = integral <= 0.f ? (i + 1) / float(lut_cdf.size()) : lut_cdf[i].a / integral;
    return lut_cdf;
}

void TransferFunction::upload_gpu() {
    // prepare lut
    const std::vector<glm::vec4> lut_cdf = compute_lut_cdf(lut);
    // setup SSBO
    lut_ssbo = SSBO("transferfunc_ssbo");
    lut_ssbo->upload_data(lut_cdf.data(), lut_cdf.size() * sizeof(glm::vec4));
}
