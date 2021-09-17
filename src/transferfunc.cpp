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

void TransferFunction::set_uniforms(const Shader& shader, uint32_t& texture_unit) const {
    shader->uniform("tf_window_left", window_left);
    shader->uniform("tf_window_width", window_width);
    shader->uniform("tf_texture", texture, texture_unit++);
}

void TransferFunction::upload_gpu() {
    // copy
    auto lut_cdf = lut;
    // build density CDF
    for (uint32_t i = 1; i < lut_cdf.size(); ++i)
        lut_cdf[i].a += lut_cdf[i-1].a;
    const float integral = lut_cdf[lut_cdf.size()-1].a;
    for (uint32_t i = 0; i < lut_cdf.size(); ++i)
        lut_cdf[i].a = integral <= 0.f ? (i+1) / float(lut_cdf.size()) : lut_cdf[i].a / integral;
    // setup GL texture
    texture = Texture2D("transferfunc_lut", lut_cdf.size(), 1, GL_RGBA16F, GL_RGBA, GL_FLOAT, lut_cdf.data(), false);
    texture->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);//GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);//GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    texture->unbind();
}
