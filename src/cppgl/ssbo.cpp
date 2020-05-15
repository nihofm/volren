#include "ssbo.h"

SSBO::SSBO(const std::string& name) : NamedMap(name), size_bytes(0) {
    glGenBuffers(1, &id);
}

SSBO::SSBO(const std::string& name, size_t size_bytes) : SSBO(name) {
    resize(size_bytes);
}

SSBO::~SSBO() {
    glDeleteBuffers(1, &id);
}

void SSBO::resize(size_t size_bytes, GLenum hint) {
    this->size_bytes = size_bytes;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size_bytes, 0, hint);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SSBO::bind(uint32_t unit) const {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, unit, id);
}

void SSBO::unbind(uint32_t unit) const {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, unit, 0);
}

void* SSBO::map(GLenum access) const {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
    return glMapBuffer(GL_SHADER_STORAGE_BUFFER, access);
}

void SSBO::unmap() const {
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}
