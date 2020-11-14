#include "grid.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(volpy, m) {
    py::class_<HostGrid>(m, "Grid")
        .def(py::init<const std::string&, const std::string&>())
        .def_property_readonly("gridName", &HostGrid::gridName)
        .def_property_readonly("activeVoxelCount", &HostGrid::activeVoxelCount);
}