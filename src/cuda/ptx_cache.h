#pragma once

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
namespace fs = std::filesystem;
#include <cuda.h>
#include <cuda_runtime.h>

class PtxModule {
public:
    PtxModule();
    PtxModule(const fs::path& filepath);
    ~PtxModule();

    PtxModule(const PtxModule&) = delete;
    PtxModule& operator=(const PtxModule&) = delete;

    void compile(const fs::path& filename);

    CUfunction get_kernel(const std::string& name) const;
    void launch_kernel(const std::string& name, const dim3 blocks, const dim3 threads, void** args, uint32_t shared_mem_bytes = 0, CUstream stream = 0, void** extra = 0) const;

    // data
    CUmodule module;
};

class PtxCache {
public:
    PtxCache();
    PtxCache(const fs::path& folder);

    bool reload_modified();

    std::pair<std::shared_ptr<PtxModule>, fs::file_time_type> lookup(const fs::path& filepath);
    std::shared_ptr<PtxModule> get_module(const fs::path& filepath);
    fs::file_time_type get_timestamp(const fs::path& filepath);
    CUfunction get_kernel(const fs::path& filepath, const std::string& name);

    // data
    std::map<fs::path, std::pair<std::shared_ptr<PtxModule>, fs::file_time_type>> cache;
};