#include "renderer_cuda.h"
#include "buffer.cuh"
#include "cppgl.h"

#include "kernels/render.h"

RendererCUDA::RendererCUDA() {
    fbo.resize(Context::resolution().x, Context::resolution().y);
}

RendererCUDA::~RendererCUDA() {
}

void RendererCUDA::init() {
    // TODO
}

void RendererCUDA::resize(uint32_t w, uint32_t h) {
    fbo.resize(w, h);
}

void RendererCUDA::commit() {
    // TODO
    const auto size = volume->current_grid()->index_extent();
    grid_data.resize(size.x, size.y, size.z);
    for (uint32_t z = 0; z < size.z; ++z)
        for (uint32_t y = 0; y < size.y; ++y)
            for (uint32_t x = 0; x < size.x; ++x)
                grid_data(x, y, z) = volume->current_grid()->lookup(glm::uvec3(x, y, z));
    const auto [min, maj] = volume->current_grid()->minorant_majorant();
    const auto cuda_buf = BufferCUDA<float>(grid_data.size, grid_data.ptr);
    grid = DenseGridCUDA(volume->get_transform(), maj, cuda_buf);
}

void RendererCUDA::trace() {
    // TODO: use cuda texture/surface!
    const auto cam = CameraCUDA(current_camera()->pos, current_camera()->dir, current_camera()->fov_degree);
    call_trace_kernel(fbo, cam, grid, sample);

    // DEBUG: check for errors
    // cudaCheckError(cudaDeviceSynchronize());
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
