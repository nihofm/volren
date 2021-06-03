#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <glm/glm.hpp>

// ------------------------------------------------------------------------
// python bindings

namespace py = pybind11;

template <typename VecT, typename ScalarT>
py::class_<VecT> register_vector_operators(py::class_<VecT>& pyclass) {
    return pyclass
        .def(py::self + py::self)
        .def(py::self + ScalarT())
        .def(ScalarT() + py::self)
        .def(py::self += py::self)
        .def(py::self += ScalarT())
        .def(py::self * py::self)
        .def(py::self * ScalarT())
        .def(ScalarT() * py::self)
        .def(py::self *= py::self)
        .def(py::self *= ScalarT())
        .def(py::self / py::self)
        .def(py::self / ScalarT())
        .def(ScalarT() / py::self)
        .def(py::self /= py::self)
        .def(py::self /= ScalarT())
        .def(-py::self);
}

PYBIND11_MODULE(volpy, m) {

    // ------------------------------------------------------------
    // glm::vec

    register_vector_operators<glm::vec2, float>(
        py::class_<glm::vec2>(m, "vec2")
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<float, float>())
            .def_readwrite("x", &glm::vec2::x)
            .def_readwrite("y", &glm::vec2::y)
            .def("__repr__", [](const glm::vec2& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
            }));

    register_vector_operators<glm::vec3, float>(
        py::class_<glm::vec3>(m, "vec3")
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<float, float, float>())
            .def_readwrite("x", &glm::vec3::x)
            .def_readwrite("y", &glm::vec3::y)
            .def_readwrite("z", &glm::vec3::z)
            .def("__repr__", [](const glm::vec3& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
            }));

    register_vector_operators<glm::vec4, float>(
        py::class_<glm::vec4>(m, "vec4")
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<float, float, float, float>())
            .def_readwrite("x", &glm::vec4::x)
            .def_readwrite("y", &glm::vec4::y)
            .def_readwrite("z", &glm::vec4::z)
            .def_readwrite("w", &glm::vec4::w)
            .def("__repr__", [](const glm::vec4& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + ")";
            }));

    // ------------------------------------------------------------
    // glm::ivec

    register_vector_operators<glm::ivec2, int>(
        py::class_<glm::ivec2>(m, "ivec2")
            .def(py::init<>())
            .def(py::init<int>())
            .def(py::init<int, int>())
            .def_readwrite("x", &glm::ivec2::x)
            .def_readwrite("y", &glm::ivec2::y)
            .def("__repr__", [](const glm::ivec2& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
            }));

    register_vector_operators<glm::ivec3, int>(
        py::class_<glm::ivec3>(m, "ivec3")
            .def(py::init<>())
            .def(py::init<int>())
            .def(py::init<int, int, int>())
            .def_readwrite("x", &glm::ivec3::x)
            .def_readwrite("y", &glm::ivec3::y)
            .def_readwrite("z", &glm::ivec3::z)
            .def("__repr__", [](const glm::ivec3& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
            }));

    register_vector_operators<glm::ivec4, int>(
        py::class_<glm::ivec4>(m, "ivec4")
            .def(py::init<>())
            .def(py::init<int>())
            .def(py::init<int, int, int, int>())
            .def_readwrite("x", &glm::ivec4::x)
            .def_readwrite("y", &glm::ivec4::y)
            .def_readwrite("z", &glm::ivec4::z)
            .def_readwrite("w", &glm::ivec4::w)
            .def("__repr__", [](const glm::ivec4& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + ")";
            }));

    // ------------------------------------------------------------
    // glm::uvec

    register_vector_operators<glm::uvec2, uint32_t>(
        py::class_<glm::uvec2>(m, "uvec2")
            .def(py::init<>())
            .def(py::init<uint32_t>())
            .def(py::init<uint32_t, uint32_t>())
            .def_readwrite("x", &glm::uvec2::x)
            .def_readwrite("y", &glm::uvec2::y)
            .def("__repr__", [](const glm::uvec2& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
            }));

    register_vector_operators<glm::uvec3, uint32_t>(
        py::class_<glm::uvec3>(m, "uvec3")
            .def(py::init<>())
            .def(py::init<uint32_t>())
            .def(py::init<uint32_t, uint32_t, uint32_t>())
            .def_readwrite("x", &glm::uvec3::x)
            .def_readwrite("y", &glm::uvec3::y)
            .def_readwrite("z", &glm::uvec3::z)
            .def("__repr__", [](const glm::uvec3& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
            }));

    register_vector_operators<glm::uvec4, uint32_t>(
        py::class_<glm::uvec4>(m, "uvec4")
            .def(py::init<>())
            .def(py::init<uint32_t>())
            .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>())
            .def_readwrite("x", &glm::uvec4::x)
            .def_readwrite("y", &glm::uvec4::y)
            .def_readwrite("z", &glm::uvec4::z)
            .def_readwrite("w", &glm::uvec4::w)
            .def("__repr__", [](const glm::uvec4& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + ")";
            }));

    // ------------------------------------------------------------
    // TODO automatic collection of classes with bindings?

}
