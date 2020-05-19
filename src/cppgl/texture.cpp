#include "texture.h"
#include <vector>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ----------------------------------------------------
// helper funcs

inline uint32_t format_to_channels(GLint format) {
    return format == GL_RGBA ? 4 : format == GL_RGB ? 3 : format == GL_RG ? 2 : 1;
}
inline GLint channels_to_format(uint32_t channels) {
    return channels == 4 ? GL_RGBA : channels == 3 ? GL_RGB : channels == 2 ? GL_RG : GL_RED;
}
inline GLint channels_to_float_format(uint32_t channels) {
    return channels == 4 ? GL_RGBA32F : channels == 3 ? GL_RGB32F : channels == 2 ? GL_RG32F : GL_R32F;
}
inline GLint channels_to_ubyte_format(uint32_t channels) {
    return channels == 4 ? GL_RGBA8 : channels == 3 ? GL_RGB8 : channels == 2 ? GL_RG8 : GL_R8;
}

// ----------------------------------------------------
// Texture2D

Texture2D::Texture2D(const std::string& name, const fs::path& path, bool mipmap) : NamedMap(name), id(0) {
    // load image from disk
    stbi_set_flip_vertically_on_load(1);
    int channels;
    uint8_t* data = 0;
    if (stbi_is_hdr(path.c_str())) {
        data = (uint8_t*)stbi_loadf(path.c_str(), &w, &h, &channels, 0);
        internal_format = channels_to_float_format(channels);
        format = channels_to_format(channels);
        type = GL_FLOAT;
    } else {
        data = stbi_load(path.c_str(), &w, &h, &channels, 0);
        internal_format = channels_to_ubyte_format(channels);
        format = channels_to_format(channels);
        type = GL_UNSIGNED_BYTE;
    }
    if (!data)
        throw std::runtime_error("Failed to load image file: " + path.string());
    // init GL texture
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, type, data);
    if (mipmap) glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    // free data
    stbi_image_free(data);
}

Texture2D::Texture2D(const std::string& name, uint32_t w, uint32_t h, GLint internal_format, GLenum format, GLenum type, void* data, bool mipmap)
    : NamedMap(name), id(0), w(w), h(h), internal_format(internal_format), format(format), type(type) {
    // init GL texture
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, format == GL_DEPTH_COMPONENT ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
            mipmap ? GL_LINEAR_MIPMAP_LINEAR : format == GL_DEPTH_COMPONENT ? GL_NEAREST : GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, type, data);
    if (mipmap && data != 0) glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture2D::~Texture2D() {
    if (glIsTexture(id))
        glDeleteTextures(1, &id);
}

void Texture2D::resize(uint32_t w, uint32_t h) {
    this->w = w;
    this->h = h;
    glBindTexture(GL_TEXTURE_2D, id);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, type, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture2D::bind(uint32_t unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, id);
}

void Texture2D::unbind() const {
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture2D::bind_image(uint32_t unit, GLenum access, GLenum format) const {
    glBindImageTexture(unit, id, 0, GL_FALSE, 0, access, format);
}

void Texture2D::unbind_image(uint32_t unit) const {
    glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
}

void Texture2D::save_png(const fs::path& path, bool flip) const {
    stbi_flip_vertically_on_write(flip);
    std::vector<uint8_t> pixels(w * h * format_to_channels(format));
    glBindTexture(GL_TEXTURE_2D, id);
    glGetTexImage(GL_TEXTURE_2D, 0, format, GL_UNSIGNED_BYTE, &pixels[0]);
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_write_png(path.c_str(), w, h, format_to_channels(format), pixels.data(), 0);
    std::cout << path << " written." << std::endl;
}

void Texture2D::save_jpg(const fs::path& path, int quality, bool flip) const {
    stbi_flip_vertically_on_write(flip);
    std::vector<uint8_t> pixels(w * h * format_to_channels(format));
    glBindTexture(GL_TEXTURE_2D, id);
    glGetTexImage(GL_TEXTURE_2D, 0, format, GL_UNSIGNED_BYTE, &pixels[0]);
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_write_jpg(path.c_str(), w, h, format_to_channels(format), pixels.data(), quality);
    std::cout << path << " written." << std::endl;
}

// ----------------------------------------------------
// Texture3D TODO test

Texture3D::Texture3D(const std::string& name, const fs::path& path, bool mipmap) : NamedMap(name), id(0) {
    // TODO
    throw std::runtime_error("not implemented!");
}

Texture3D::Texture3D(const std::string& name, uint32_t w, uint32_t h, uint32_t d, GLint internal_format, GLenum format, GLenum type, void* data, bool mipmap)
    : NamedMap(name), id(0), w(w), h(h), d(d), internal_format(internal_format), format(format), type(type) {
    // init GL texture
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_3D, id);
    // default border color is (0, 0, 0, 0)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexImage3D(GL_TEXTURE_3D, 0, internal_format, w, h, d, 0, format, type, data);
    if (mipmap && data != 0) glGenerateMipmap(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, 0);
}

Texture3D::~Texture3D() {
    if (glIsTexture(id))
        glDeleteTextures(1, &id);
}

void Texture3D::resize(uint32_t w, uint32_t h, uint32_t d) {
    this->w = w;
    this->h = h;
    this->d = d;
    glBindTexture(GL_TEXTURE_3D, id);
    glTexImage2D(GL_TEXTURE_3D, 0, internal_format, w, h, 0, format, type, 0);
    glBindTexture(GL_TEXTURE_3D, 0);
}

void Texture3D::bind(uint32_t unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_3D, id);
}

void Texture3D::unbind() const {
    glBindTexture(GL_TEXTURE_3D, 0);
}

void Texture3D::bind_image(uint32_t unit, GLenum access, GLenum format) const {
    glBindImageTexture(unit, id, 0, GL_FALSE, 0, access, format);
}

void Texture3D::unbind_image(uint32_t unit) const {
    glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
}
