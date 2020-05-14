#pragma once

#include <memory>
#include <filesystem>
namespace fs = std::filesystem;
#include <GL/glew.h>
#include <GL/gl.h>

#include "named_map.h"

class Texture2D : public NamedMap<Texture2D> {
public:
    // construct from image on disk
    Texture2D(const std::string& name, const fs::path& path, bool mipmap = true);
    // construct empty texture or from raw data
    Texture2D(const std::string& name, uint32_t w, uint32_t h, GLint internal_format, GLenum format, GLenum type, void *data = 0, bool mipmap = false);
    virtual ~Texture2D();

    // prevent copies and moves, since GL buffers aren't reference counted
    Texture2D(const Texture2D&) = delete;
    Texture2D& operator=(const Texture2D&) = delete;
    Texture2D& operator=(const Texture2D&&) = delete;

    explicit inline operator bool() const  { return glIsTexture(id); }
    inline operator GLuint() const { return id; }

    // resize (discards all data!)
    void resize(uint32_t w, uint32_t h);

    // bind/unbind to/from OpenGL
    void bind(uint32_t uint) const;
    void unbind() const;
    void bind_image(uint32_t unit, GLenum access, GLenum format) const;
    void unbind_image(uint32_t unit) const;

    // save to disk
    void save_png(const fs::path& path, bool flip = true) const;
    void save_jpg(const fs::path& path, int quality = 100, bool flip = true) const; // quality: [1, 100]

    // data
    GLuint id;
    int w, h;
    GLint internal_format;
    GLenum format, type;
};

// variadic alias for std::make_shared<>(...)
template <class... Args> std::shared_ptr<Texture2D> make_texture(Args&&... args) {
    return std::make_shared<Texture2D>(args...);
}
