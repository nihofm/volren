#pragma once

#include <iostream>
#include <filesystem>
#include <cuda_runtime.h>
//#include <cuda_fp16.h>
#include <pybind11/pybind11.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

__hostdev__ inline float3 cast(const nanovdb::Vec3R& x) { return make_float3(x[0], x[1], x[2]); }
__hostdev__ inline int3 cast(const nanovdb::Coord& x) { return make_int3(x[0], x[1], x[2]); }

class Grid {
    using BufferT = nanovdb::HostBuffer;
public:
    Grid(nanovdb::GridHandle<BufferT>&& handle);
    Grid(const std::filesystem::path& path, const std::string& gridname);

    operator bool() const { return handle.operator bool(); }

    // access to meta data
    __hostdev__ const nanovdb::GridMetaData* metaData() const { return handle.gridMetaData(); }
    __hostdev__ std::string gridName() const { return metaData()->gridName(); }
    __hostdev__ nanovdb::GridType gridType() const { return metaData()->gridType(); }
    __hostdev__ nanovdb::GridClass gridClass() const { return metaData()->gridClass(); }
    __hostdev__ const nanovdb::Map& map() const { return metaData()->map(); }
    __hostdev__ float3 worldBBoxMin() const { return cast(metaData()->worldBBox().min()); }
    __hostdev__ float3 worldBBoxMax() const { return cast(metaData()->worldBBox().max()); }
    __hostdev__ int3 indexBBoxMin() const { return cast(metaData()->indexBBox().min()); }
    __hostdev__ int3 indexBBoxMax() const { return cast(metaData()->indexBBox().max()); }
    __hostdev__ float3 voxelSize() const { return cast(metaData()->voxelSize()); }
    __hostdev__ uint64_t activeVoxelCount() const { return metaData()->activeVoxelCount(); }

    static void init_bindings(pybind11::module_& m);

    // data
    nanovdb::GridHandle<BufferT> handle;
};
