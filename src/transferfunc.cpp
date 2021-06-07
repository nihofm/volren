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
    // setup GL texture
    texture = Texture2D("transferfunc_lut", lut.size(), 1, GL_RGBA32F, GL_RGBA, GL_FLOAT, lut.data(), false);
    texture->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    texture->unbind();
}
