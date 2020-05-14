#pragma once

#include <string>
#include <vector>
#include <memory>
#include <GL/glew.h>
#include <GL/gl.h>
#include "named_map.h"
#include "texture.h"

class Framebuffer : public NamedMap<Framebuffer> {
public:
    Framebuffer(const std::string& name, uint32_t w, uint32_t h);
    virtual ~Framebuffer();

    // prevent copies and moves, since GL buffers aren't reference counted
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&&) = delete;

    inline operator GLuint() const { return id; }

    void bind();
    void unbind();

    void check() const;
    void resize(uint32_t w, uint32_t h);

    void attach_depthbuffer(std::shared_ptr<Texture2D> tex = std::shared_ptr<Texture2D>());
    void attach_colorbuffer(const std::shared_ptr<Texture2D>& tex);

    // data
    GLuint id;
    uint32_t w, h;
    std::vector<std::shared_ptr<Texture2D>> color_textures;
    std::vector<GLenum> color_targets;
    std::shared_ptr<Texture2D> depth_texture;
    GLint prev_vp[4];
};

// variadic alias for std::make_shared<>(...)
template <class... Args> std::shared_ptr<Framebuffer> make_framebuffer(Args&&... args) {
    return std::make_shared<Framebuffer>(args...);
}
