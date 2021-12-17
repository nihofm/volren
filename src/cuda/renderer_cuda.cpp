#include "renderer_cuda.h"
#include "buffer.cuh"
#include "cppgl.h"

#include "kernels/render.h"

RendererCUDA::RendererCUDA() {}

RendererCUDA::~RendererCUDA() {}

void RendererCUDA::init() {
    // TODO
}

void RendererCUDA::resize(uint32_t w, uint32_t h) {
    fbo.resize({w, h});
}

void RendererCUDA::commit() {
    // TODO
}

void RendererCUDA::trace() {
    // TODO: use cuda texture/surface!
    call_trace_kernel(fbo.map_cuda(), { fbo.size.x, fbo.size.y }, cast(current_camera()->pos), cast(current_camera()->dir), current_camera()->fov_degree);
    fbo.unmap_cuda();

    // DEBUG: check for errors
    cudaCheckError(cudaDeviceSynchronize());
}

void RendererCUDA::draw() {
    static Shader tonemap_shader = Shader("tonemap", "shader/quad.vs", "shader/tonemap_buf.fs");
    tonemap_shader->bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fbo.gl_buf);
    tonemap_shader->uniform("size", glm::ivec2(fbo.size.x, fbo.size.y));
    tonemap_shader->uniform("exposure", tonemap_exposure);
    tonemap_shader->uniform("gamma", tonemap_gamma);
    Quad::draw();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    tonemap_shader->unbind();

    /* ImageCUDAGL version
    static Shader tonemap_shader = Shader("tonemap", "shader/quad.vs", "shader/tonemap.fs");
    tonemap_shader->bind();
    tonemap_shader->uniform("size", glm::ivec2(fbo.size.x, fbo.size.y));
    tonemap_shader->uniform("exposure", tonemap_exposure);
    tonemap_shader->uniform("gamma", tonemap_gamma);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fbo.gl_img);
    glUniform1i(glGetUniformLocation(tonemap_shader->id, "tex"), 0);
    Quad::draw();
    glBindTexture(GL_TEXTURE_2D, 0);
    tonemap_shader->unbind();
    */
}
