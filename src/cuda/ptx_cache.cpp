#include "ptx_cache.h"
#include <iostream>
#include <fstream>

#include "host_helpers.h"

// ----------------------------------------------------------
// CachedPtx


PtxModule::PtxModule() : module(0) {
    cuda_init();
}

PtxModule::PtxModule(const fs::path& filepath) : PtxModule() {
    compile(filepath);
}

PtxModule::~PtxModule() {
    if (module) cuCheckError(cuModuleUnload(module));
}

void PtxModule::compile(const fs::path& filepath) {
    std::cout << "Compiling: " << filepath << "..." << std::endl;
    // read source from file
    if (!fs::exists(filepath))
        throw std::runtime_error("ERROR: CUDA source file not found: " + filepath.string());
    std::ifstream ifs(filepath, std::ios::binary);
    const std::string cu_source = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    if (cu_source.empty())
        throw std::runtime_error("ERROR: Trying to compile PTX from empty source!");
    // create program
    nvrtcProgram program = 0;
    nvrtcCheckError(nvrtcCreateProgram(&program, cu_source.c_str(), filepath.filename().c_str(), 0, NULL, NULL));
    // set compiler options // TODO: make options accessable
    std::vector<std::string> options;
    options.push_back(std::getenv("CPPFLAGS"));
    options.push_back("-I" + filepath.parent_path().string());
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
        throw std::runtime_error(std::string("NVRTC Compilation of \"" + filepath.string() + "\" failed: ") + nvrtcGetErrorString(result) + "\n" + log);
    }
    // fetch ptx
    std::string ptx;
    size_t ptx_size = 0;
    nvrtcCheckError(nvrtcGetPTXSize(program, &ptx_size));
    ptx.resize(ptx_size);
    nvrtcCheckError(nvrtcGetPTX(program, &ptx[0]));
    // cleanup
    nvrtcCheckError(nvrtcDestroyProgram(&program));
    // load module
    if (module) cuCheckError(cuModuleUnload(module));
    cuCheckError(cuModuleLoadDataEx(&module, &ptx[0], 0, 0, 0));
}

CUfunction PtxModule::get_kernel(const std::string& name) const {
    CUfunction kernel;
    cuCheckError(cuModuleGetFunction(&kernel, module, name.c_str()));
    return kernel;
}

void PtxModule::launch_kernel(const std::string& name, const dim3 blocks, const dim3 threads, void** args, uint32_t shared_mem_bytes, CUstream stream, void** extra) const {
    cuCheckError(cuLaunchKernel(get_kernel(name), blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, shared_mem_bytes, stream, args, extra));
}

// ----------------------------------------------------------
// PtxCache

PtxCache::PtxCache() {}

PtxCache::PtxCache(const fs::path& folder) {
    for(const fs::path& p : fs::directory_iterator(folder))
        if (p.extension() == ".cu")
            cache[p] = { std::make_shared<PtxModule>(p), fs::last_write_time(p) };
}

bool PtxCache::reload_modified() {
    bool reloaded = false;
    for (auto [path, cache_entry] : cache) {
        try {
            const auto timestamp = fs::last_write_time(path);
            if (timestamp != cache_entry.second) {
                cache_entry.first->compile(path);
                cache_entry.second = timestamp;
                reloaded |= true;
            }
        } catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    return reloaded;
}

std::pair<std::shared_ptr<PtxModule>, fs::file_time_type> PtxCache::lookup(const fs::path& filepath) {
    if (!cache.count(filepath)) cache[filepath] = { std::make_shared<PtxModule>(filepath), fs::last_write_time(filepath) };
    return cache[filepath];
}

std::shared_ptr<PtxModule> PtxCache::get_module(const fs::path& filepath) {
    return lookup(filepath).first;
}

fs::file_time_type PtxCache::get_timestamp(const fs::path& filepath) {
    return lookup(filepath).second;
}

CUfunction PtxCache::get_kernel(const fs::path& filepath, const std::string& name) {
    return get_module(filepath)->get_kernel(name);
}