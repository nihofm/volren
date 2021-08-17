#include "renderer.h"

inline float3 cast(const glm::vec3& v) { return make_float3(v.x, v.y, v.z); }
inline float4 cast(const glm::vec4& v) { return make_float4(v.x, v.y, v.z, v.w); }
inline uint3 cast(const glm::uvec3& v) { return make_uint3(v.x, v.y, v.z); }
inline dim3 cast_dim(const glm::uvec3& v) { return dim3(v.x, v.y, v.z); }

// render kernel
void render_cuda(float4* fbo, size_t w, size_t h, const BufferCUDA<CameraCUDA>& cam, const BufferCUDA<VolumeCUDA>& vol);

void RendererCUDA::init() {
    // load default volume
    if (!volume)
        volume = std::make_shared<voldata::Volume>();
    
    // setup fbo
    const glm::ivec2 res = Context::resolution();
    fbo.resize(dim3(res.x, res.y));
}

void RendererCUDA::resize(uint32_t w, uint32_t h) {
    fbo.resize(dim3(w, h));
}

void RendererCUDA::commit() {
    // copy transform
    const auto mat = glm::transpose(glm::inverse(volume->get_transform()));
    vol->transform[0] = cast(mat[0]);
    vol->transform[1] = cast(mat[1]);
    vol->transform[2] = cast(mat[2]);
    vol->transform[3] = cast(mat[3]);
    // copy parameters
    const auto [bb_min, bb_max] = volume->AABB();
    vol->bb_min = cast(bb_min); 
    vol->bb_max = cast(bb_max); 
    vol->albedo = cast(volume->albedo);
    vol->phase = volume->phase;
    vol->density_scale = volume->density_scale;
    vol->majorant = volume->minorant_majorant().second;
    // copy grid data TODO paralellize?
    vol->grid.resize(cast_dim(volume->current_grid()->index_extent()));
    for (uint32_t z = 0; z < vol->grid.size.z; ++z)
        for (uint32_t y = 0; y < vol->grid.size.y; ++y)
            for (uint32_t x = 0; x < vol->grid.size.x; ++x)
                vol->grid[make_uint3(x, y, z)] = volume->current_grid()->lookup(glm::uvec3(x, y, z));
}

void RendererCUDA::trace(uint32_t spp) {
    // update camera
    cam->pos = cast(current_camera()->pos);
    cam->dir = cast(current_camera()->dir);
    cam->fov = current_camera()->fov_degree;
    // trace
    for (uint32_t i = 0; i < spp; ++i)
        render_cuda(fbo.map_cuda(), fbo.size.x, fbo.size.y, cam, vol);
    fbo.unmap_cuda();
}

void RendererCUDA::draw() {
    fbo.draw(1.f, 1.f);
}
