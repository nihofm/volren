#include "volume.h"
#include <vector>

Volume::Volume() : model(1), absorbtion_coefficient(0.01), scattering_coefficient(0.05), density_scale(100), max_density(0) {}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, float density) : Volume() {
    std::vector<float> data(w * h * d, density);
    load(name, w, h, d, data.data());
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const float* data) : Volume() {
    load(name, w, h, d, data);
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const uint8_t* data) : Volume() {
    load(name, w, h, d, data);
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const uint16_t* data) : Volume() {
    load(name, w, h, d, data);
}

Volume::~Volume() {

}

void Volume::load(const std::string& name, size_t w, size_t h, size_t d, const uint8_t* data) {
    // setup GL texture
    texture = make_texture3D(name, w, h, d, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data, false);
    // fetch max density for delta tracking
    max_density = 0.f;
    for (size_t i = 0; i < w * h * d; ++i)
        max_density = std::max(data[i] / 255.f, max_density);
}

void Volume::load(const std::string& name, size_t w, size_t h, size_t d, const uint16_t* data) {
    // setup GL texture
    texture = make_texture3D(name, w, h, d, GL_R16, GL_RED, GL_UNSIGNED_SHORT, data, false);
    // fetch max density for delta tracking
    max_density = 0.f;
    for (size_t i = 0; i < w * h * d; ++i)
        max_density = std::max(data[i] / 65535.f, max_density);
}

void Volume::load(const std::string& name, size_t w, size_t h, size_t d, const float* data) {
    // setup GL texture
    texture = make_texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, data, false);
    // fetch max density for delta tracking
    max_density = 0.f;
    for (size_t i = 0; i < w * h * d; ++i)
        max_density = std::max(data[i], max_density);
}
