#include "volume.h"
#include <vector>

Volume::Volume() : model(1), absorbtion_coefficient(0.01), scattering_coefficient(0.05), emission(0), phase_g(0), density_scale(100), max_density(0) {}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, float density) : Volume() {
    // setup GL texture
    std::vector<float> data(w * h * d, density);
    texture = make_texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, data.data(), false);
    // fetch (normalized) max density for delta tracking
    max_density = density;
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const uint8_t* data) : Volume() {
    // setup GL texture
    texture = make_texture3D(name, w, h, d, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data, false);
    // fetch (normalized) max density for delta tracking
    max_density = 0.f;
    for (size_t i = 0; i < w * h * d; ++i)
        max_density = std::max(data[i] / 255.f, max_density);
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const uint16_t* data) : Volume() {
    // setup GL texture
    texture = make_texture3D(name, w, h, d, GL_R16, GL_RED, GL_UNSIGNED_SHORT, data, false);
    // fetch (normalized) max density for delta tracking
    max_density = 0.f;
    for (size_t i = 0; i < w * h * d; ++i)
        max_density = std::max(data[i] / 65535.f, max_density);
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const float* data) : Volume() {
    // setup GL texture
    texture = make_texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, data, false);
    // fetch (normalized) max density for delta tracking
    max_density = 0.f;
    for (size_t i = 0; i < w * h * d; ++i)
        max_density = std::max(data[i], max_density);
}

Volume::~Volume() {

}
