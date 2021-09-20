#include "renderer_optix.h"
#include "host_helpers.h"

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

template <typename T> struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord = SbtRecord<float3>;
using MissSbtRecord = SbtRecord<int>;

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

// ----------------------------------------------------------
// RendererOptix

static void initOptix() {
    static bool is_init = false;
    if (is_init) return;

    // Initialize CUDA
    cudaCheckError(cudaFree(0));

    // Initialize Optix
    optixCheckError(optixInit());

    is_init = true;
}

RendererOptix::RendererOptix() : params(0) {
    initOptix();

    // init optix device
    context = 0;
    CUcontext cuCtx = 0;
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &optix_log_cb;
    options.logCallbackLevel          = 3;//4;
    optixCheckError(optixDeviceContextCreate(cuCtx, &options, &context));

    // init optix module
    module = 0;
    OptixPipelineCompileOptions pipeline_options = {};
    {
        OptixModuleCompileOptions module_options = {};
        module_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
        
        pipeline_options.usesMotionBlur        = false;
        pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_options.numPayloadValues      = 2;
        pipeline_options.numAttributeValues    = 2;
        pipeline_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_options.pipelineLaunchParamsVariableName = "params";

        const std::string ptx_source = compile_ptx("src/cuda/ptx/raygen_pinhole.cu");

        optixCheckError(optixModuleCreateFromPTX(
            context,
            &module_options,
            &pipeline_options,
            ptx_source.c_str(),
            ptx_source.size(),
            0,
            0,
            &module));
    }

    // init optix program groups
    raygen_group = 0;
    miss_group = 0;
    {
        OptixProgramGroupOptions group_options   = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_group_desc  = {};
        raygen_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_group_desc.raygen.module            = module;
        raygen_group_desc.raygen.entryFunctionName = "__raygen__pinhole";
        optixCheckError(optixProgramGroupCreate(context, &raygen_group_desc, 1, &group_options, 0, 0, &raygen_group));

        // Leave miss group's module and entryfunc name null
        OptixProgramGroupDesc miss_group_desc = {};
        miss_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        optixCheckError(optixProgramGroupCreate(context, &miss_group_desc, 1, &group_options, 0, 0, &miss_group));
    }

    // init optix pipeline
    pipeline = 0;
    {
        const uint32_t    max_trace_depth  = 0;
        OptixProgramGroup program_groups[] = { raygen_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth          = max_trace_depth;
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        optixCheckError(optixPipelineCreate(
            context,
            &pipeline_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            0,
            0,
            &pipeline));

        OptixStackSizes stack_sizes = {};
        for(auto& prog_group : program_groups)
            optixCheckError(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        optixCheckError(optixUtilComputeStackSizes(
            &stack_sizes,
            max_trace_depth,
            0, // maxCCDepth
            0, // maxDCDEpth
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state, &continuation_stack_size));
        optixCheckError(optixPipelineSetStackSize(
            pipeline,
            direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state,
            continuation_stack_size,
            2 // maxTraversableDepth
            ));
    }

    // init optix shader binding table
    sbt = {};
    {
        RayGenSbtRecord* raygen_sbt_record;
        cudaCheckError(cudaMallocManaged(&raygen_sbt_record, sizeof(RayGenSbtRecord)));
        optixCheckError(optixSbtRecordPackHeader(raygen_group, raygen_sbt_record));
        raygen_sbt_record->data = make_float3(0.462f, 0.725f, 0.f);

        MissSbtRecord* miss_sbt_record;
        cudaCheckError(cudaMallocManaged(&miss_sbt_record, sizeof(MissSbtRecord)));
        optixCheckError(optixSbtRecordPackHeader(miss_group, miss_sbt_record));

        sbt.raygenRecord                = (CUdeviceptr)raygen_sbt_record;
        // TODO sbt.exceptionRecord
        sbt.missRecordBase              = (CUdeviceptr)miss_sbt_record;
        sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        sbt.missRecordCount             = 1;
    }

    // alloc kernel parameter memory (managed)
    cudaCheckError(cudaMallocManaged(&params, sizeof(Params)));

    // load default volume
    if (!volume)
        volume = std::make_shared<voldata::Volume>();
    
    // setup fbo
    const glm::ivec2 res = Context::resolution();
    fbo.resize(dim3(res.x, res.y));
}

RendererOptix::~RendererOptix() {
    cudaCheckError(cudaFree(params));
    cudaCheckError(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    cudaCheckError(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    optixCheckError(optixPipelineDestroy(pipeline));
    optixCheckError(optixProgramGroupDestroy(miss_group));
    optixCheckError(optixProgramGroupDestroy(raygen_group));
    optixCheckError(optixModuleDestroy(module));
    optixCheckError(optixDeviceContextDestroy(context));
}

void RendererOptix::init() {}

void RendererOptix::resize(uint32_t w, uint32_t h) {
    fbo.resize(dim3(w, h));
}

void RendererOptix::commit() {
    // TODO
    /*
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
    */
}

#include "cuda/ptx/common.cuh"

void RendererOptix::trace() {
    // upload params
    params->image = fbo.map_cuda();
    params->resolution = make_float2(fbo.size.x, fbo.size.y);
    params->cam_pos = cast(current_camera()->pos);
    params->cam_dir = cast(current_camera()->dir);
    params->cam_fov = current_camera()->fov_degree;
    const auto& [bb_min, bb_max] = volume->AABB();
    params->vol_bb_min = cast(bb_min);
    params->vol_bb_max = cast(bb_max);

    optixCheckError(optixLaunch(pipeline, 0, (CUdeviceptr)params, sizeof(Params), &sbt, fbo.size.x, fbo.size.y, /*depth=*/1));
    
    cudaDeviceSynchronize();
    cudaCheckError(cudaGetLastError());

    fbo.unmap_cuda();
}

void RendererOptix::draw() {
    fbo.draw(1.f, 1.f);
}
