#pragma once

#include <cppgl/texture.h>
#include <glm/glm.hpp>

class VolumeImpl {
public:
    VolumeImpl(); // default construct as invalid volume
    VolumeImpl(const fs::path& path); // heterogeneous from file on disk
    // TODO remove name arg?
    VolumeImpl(const std::string& name, size_t w, size_t h, size_t d, float density); // homogeneous (very inefficient, no closed form!)
    VolumeImpl(const std::string& name, size_t w, size_t h, size_t d, const uint8_t* data); // heterogeneous (linear array of w x h x d uchars)
    VolumeImpl(const std::string& name, size_t w, size_t h, size_t d, const uint16_t* data); // heterogeneous (linear array of w x h x d ushorts)
    VolumeImpl(const std::string& name, size_t w, size_t h, size_t d, const float* data); // heterogeneous (linear array of w x h x d floats)
    virtual ~VolumeImpl();

    explicit inline operator bool() const  { return texture->operator bool(); }
    //inline operator GLuint() const { return texture->operator GLuint(); }

    // data
    glm::mat4 model;
    float absorbtion_coefficient;
    float scattering_coefficient;
    float phase_g;
    glm::vec3 slice_thickness;
    Texture3D texture;
};

using Volume = NamedHandle<VolumeImpl>;
