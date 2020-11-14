#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

__hostdev__ float3 cast(const nanovdb::Vec3R& x) { return make_float3(x[0], x[1], x[2]); }
__hostdev__ int3 cast(const nanovdb::Coord& x) { return make_int3(x[0], x[1], x[2]); }

template <typename BufferT = nanovdb::HostBuffer>
class Grid {
public:
    Grid(const nanovdb::GridHandle<BufferT>& handle) : handle(std::move(handle)) {}
    Grid(const std::string& filename, const std::string& gridname) : handle(nanovdb::io::readGrid<BufferT>(filename, gridname)) {}

    operator bool() const { return handle.operator bool(); }

    __hostdev__ const nanovdb::GridMetaData* metaData() const { return handle.gridMetaData(); }

    __hostdev__ std::string gridName() const { return metaData()->gridName(); }
    __hostdev__ nanovdb::GridType gridType() const { return metaData()->gridType(); }
    __hostdev__ nanovdb::GridClass gridClass() const { return metaData()->gridClass(); }

    __hostdev__ const nanovdb::Map& map() const { return metaData()->map(); }

    __hostdev__ float3 worldBBoxMin() const { return cast(metaData()->worldBBox().min()); }
    __hostdev__ float3 worldBBoxMax() const { return cast(metaData()->worldBBox().max()); }

    __hostdev__ int3 indexBBoxMin() const { return cast(metaData()->indexBBox().min()); }
    __hostdev__ int3 indexBBoxMax() const { return cast(metaData()->indexBBox().max()); }

    __hostdev__ double voxelSize() const { return metaData()->voxelSize(); }
    __hostdev__ uint64_t activeVoxelCount() const { return metaData()->activeVoxelCount(); }

    // data
    nanovdb::GridHandle<BufferT> handle;
};

using HostGrid = Grid<nanovdb::HostBuffer>;
using DeviceGrid = Grid<nanovdb::CudaDeviceBuffer>;
