#include "../renderer.h"
#include "common.cuh"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <filesystem>
namespace fs = std::filesystem;
#include <fstream>

// ----------------------------------------------------------
// helper functions

static void optix_log_cb(unsigned int level, const char* tag, const char* message, void* /*data */) {
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

inline std::string read_file(const fs::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

inline std::string compile_ptx(const fs::path& path) {
    // read source from file
    const std::string cu_source = read_file(path);
    // create program
    nvrtcProgram program = 0;
    nvrtcCheckError(nvrtcCreateProgram(&program, cu_source.c_str(), path.filename().c_str(), 0, NULL, NULL)); // TODO options
    // set compiler options // TODO decide on options
    std::vector<std::string> options;
    options.push_back("-I" + path.parent_path().string());
    options.push_back(std::string("-I") + NVRTC_CUDA_INCLUDE);
    options.push_back(std::string("-I") + NVRTC_OPTIX_INCLUDE);
    options.push_back("-std=c++11");
    options.push_back("-use_fast_math");
    options.push_back("-lineinfo");
    // prepare options array
    std::vector<const char*> compiler_options;
    for (const auto& opt : options)
        compiler_options.push_back(opt.c_str());
    // jit compile to ptx
    const nvrtcResult result = nvrtcCompileProgram(program, compiler_options.size(), compiler_options.data());
    if (result != NVRTC_SUCCESS) {
        // fetch log
        std::string log;
        size_t log_size = 0;
        nvrtcCheckError(nvrtcGetProgramLogSize(program, &log_size));
        log.resize(log_size);
        if (log_size) nvrtcCheckError(nvrtcGetProgramLog(program, &log[0]));
        throw std::runtime_error(std::string("NVRTC Compilation failed: ") + nvrtcGetErrorString(result) + "\n" + log);
    }
    // fetch ptx
    std::string ptx;
    size_t ptx_size = 0;
    nvrtcCheckError(nvrtcGetPTXSize(program, &ptx_size));
    ptx.resize(ptx_size);
    nvrtcCheckError(nvrtcGetPTX(program, &ptx[0]));
    // cleanup
    nvrtcCheckError(nvrtcDestroyProgram(&program));
    return ptx;
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

RendererOptix::RendererOptix() {}
RendererOptix::~RendererOptix() {}

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
    {
        OptixModuleCompileOptions module_options = {};
        module_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
        
        OptixPipelineCompileOptions pipeline_options = {};
        pipeline_options.usesMotionBlur        = false;
        pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_options.numPayloadValues      = 2;
        pipeline_options.numAttributeValues    = 2;
        pipeline_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_options.pipelineLaunchParamsVariableName = "params";


        char log[2048];
        size_t sizeof_log = sizeof(log);
        const std::string ptx_source = compile_ptx("ptx/draw_solid_color.cu");

        optixCheckError(optixModuleCreateFromPTX(
            context,
            &module_options,
            &pipeline_options,
            ptx_source.c_str(),
            ptx_source.size(),
            log,
            &sizeof_log,
            &module));
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

void RendererOptix::trace() {
    // update camera
    cam->pos = cast(current_camera()->pos);
    cam->dir = cast(current_camera()->dir);
    cam->fov = current_camera()->fov_degree;
    // trace
    render_cuda(fbo.map_cuda(), fbo.size.x, fbo.size.y, cam, vol);
    fbo.unmap_cuda();
}

void RendererOptix::draw() {
    fbo.draw(1.f, 1.f);
}
