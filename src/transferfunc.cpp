#include "transferfunc.h"
#include <fstream>
#include <iostream>

TransferFunctionImpl::TransferFunctionImpl(const std::string& name) : name(name), window_center(0), window_width(1) {}

TransferFunctionImpl::TransferFunctionImpl(const std::string& name, const fs::path& path) : TransferFunctionImpl(name) {
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

TransferFunctionImpl::TransferFunctionImpl(const std::string& name, const std::vector<glm::vec4>& lut) : TransferFunctionImpl(name) {
    this->lut = lut;
    upload_gpu();
}

TransferFunctionImpl::~TransferFunctionImpl() {}

void TransferFunctionImpl::upload_gpu() {
    // setup GL texture
    texture = Texture2D("TODO_transferfunc_lut", lut.size(), 1, GL_RGBA32F, GL_RGBA, GL_FLOAT, lut.data(), false);
    texture->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    texture->unbind();
}
