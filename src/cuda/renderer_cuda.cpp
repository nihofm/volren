#include "../renderer.h"
#include "common.cuh"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <fstream>

// ----------------------------------------------------------
// helper functions

std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

static void optix_log_cb(unsigned int level, const char* tag, const char* message, void* /*data */) {
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

inline float3 cast(const glm::vec3& v) { return make_float3(v.x, v.y, v.z); }
inline float4 cast(const glm::vec4& v) { return make_float4(v.x, v.y, v.z, v.w); }
inline uint3 cast(const glm::uvec3& v) { return make_uint3(v.x, v.y, v.z); }
inline dim3 cast_dim(const glm::uvec3& v) { return dim3(v.x, v.y, v.z); }

// ----------------------------------------------------------
// render kernel

void render_cuda(float4* fbo, size_t w, size_t h, const BufferCUDA<CameraCUDA>& cam, const BufferCUDA<VolumeCUDA>& vol);

// ----------------------------------------------------------
// RendererOptix

void RendererOptix::initOptix() {
    static bool is_init = false;
    if (is_init) return;

    // Initialize CUDA
    cudaCheckError(cudaFree(0));

    // Initialize Optix
    optixCheckError(optixInit());

    is_init = true;
}

void RendererOptix::init() {
    initOptix();

    // init Optix device
    context = 0;
    CUcontext cuCtx = 0;
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &optix_log_cb;
    options.logCallbackLevel          = 4;
    optixCheckError(optixDeviceContextCreate(cuCtx, &options, &context));

    // init Optix module
    module = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues      = 2;
        pipeline_compile_options.numAttributeValues    = 2;
        pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        const std::string input = read_file("ptx/draw_solid_color.cu");
        std::cout << "input: " << input << std::endl;

        char log[2048];
        size_t sizeof_log = sizeof(log);

        optixCheckError(optixModuleCreateFromPTX(context,
                    &module_compile_options,
                    &pipeline_compile_options,
                    input.c_str(),
                    input.size(),
                    log,
                    &sizeof_log,
                    &module
                    ));
    }

    // load default volume
    if (!volume)
        volume = std::make_shared<voldata::Volume>();
    
    // setup fbo
    const glm::ivec2 res = Context::resolution();
    fbo.resize(dim3(res.x, res.y));
}

void RendererOptix::resize(uint32_t w, uint32_t h) {
    fbo.resize(dim3(w, h));
}

void RendererOptix::commit() {
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

void RendererOptix::trace(uint32_t spp) {
    // update camera
    cam->pos = cast(current_camera()->pos);
    cam->dir = cast(current_camera()->dir);
    cam->fov = current_camera()->fov_degree;
    // trace
    for (uint32_t i = 0; i < spp; ++i)
        render_cuda(fbo.map_cuda(), fbo.size.x, fbo.size.y, cam, vol);
    fbo.unmap_cuda();
}

void RendererOptix::draw() {
    fbo.draw(1.f, 1.f);
}
