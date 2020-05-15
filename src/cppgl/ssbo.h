#pragma once

#include <string>
#include <memory>
#include <vector>
#include <GL/glew.h>
#include <GL/gl.h>

#include "named_map.h"

class SSBO : public NamedMap<SSBO> {
public:
    SSBO(const std::string& name);
    SSBO(const std::string& name, size_t size_bytes);
    virtual ~SSBO();

    // prevent copies and moves, since GL buffers aren't reference counted
    SSBO(const SSBO&) = delete;
    SSBO& operator=(const SSBO&) = delete;
    SSBO& operator=(const SSBO&&) = delete;

    explicit inline operator bool() const  { return glIsBuffer(id) && size_bytes > 0; }
    inline operator GLuint() const { return id; }

    // resize (discards all data!)
    void resize(size_t size_bytes, GLenum hint = GL_DYNAMIC_DRAW);

    // bind/unbind to/from OpenGL
    void bind(uint32_t unit) const;
    void unbind(uint32_t unit) const;

    // map/unmap from GPU mem
    void* map(GLenum access = GL_READ_WRITE) const;
    void unmap() const;

    // data
    GLuint id;
    size_t size_bytes;
};

// variadic alias for std::make_shared<>(...)
template <class... Args> std::shared_ptr<SSBO> make_ssbo(Args&&... args) {
    return std::make_shared<SSBO>(args...);
}
