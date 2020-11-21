#include "grid.h"
#include <nanovdb/util/IO.h>
#include <nanovdb/util/OpenToNanoVDB.h>

// ------------------------------------------------------------------------
// Grid

Grid::Grid(nanovdb::GridHandle<BufferT>&& handle) : handle(std::move(handle)) {}

Grid::Grid(const std::filesystem::path& path, const std::string& gridname) {
    if (path.extension() == ".vdb") {
        openvdb::initialize();
        openvdb::io::File file(path.string());
        file.open();
        openvdb::GridBase::Ptr baseGrid = file.readGrid(gridname);
        file.close();
        if (!baseGrid) throw std::runtime_error(std::string("error loading grid: ") + path.string() + ", grid name <" + gridname + "> not found!");
        handle = std::move(nanovdb::openToNanoVDB<nanovdb::HostBuffer>(baseGrid));
    } else if (path.extension() == ".nvdb")
        handle = std::move(nanovdb::io::readGrid<nanovdb::HostBuffer>(path.string(), gridname));
    else
        throw std::runtime_error(std::string("error loading grid: ") + path.string() + ", extension <" + path.extension().string() + "> not supported!");
}

// ------------------------------------------------------------------------
// Python bindings

namespace py = pybind11;

void Grid::init_bindings(pybind11::module_& m) {
    py::class_<Grid>(m, "Grid")
        .def(py::init<const std::string&, const std::string&>())
        .def("gridName", &Grid::gridName)
        .def("worldBBoxMin", &Grid::worldBBoxMin)
        .def("worldBBoxMax", &Grid::worldBBoxMax)
        .def("indexBBoxMin", &Grid::indexBBoxMin)
        .def("indexBBoxMax", &Grid::indexBBoxMax)
        .def("voxelSize", &Grid::voxelSize)
        .def("activeVoxelCount", &Grid::activeVoxelCount);
}
