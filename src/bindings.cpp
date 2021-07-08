#include <glm/glm.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cppgl.h>
#include <voldata.h>

#include "renderer.h"
#include "environment.h"
#include "transferfunc.h"

// ------------------------------------------------------------------------
// python bindings

template <typename VecT, typename ScalarT>
py::class_<VecT> register_vector_operators(py::class_<VecT>& pyclass) {
    return pyclass
        .def(py::self + py::self)
        .def(py::self + ScalarT())
        .def(ScalarT() + py::self)
        .def(py::self += py::self)
        .def(py::self += ScalarT())
        .def(py::self - py::self)
        .def(py::self - ScalarT())
        .def(ScalarT() - py::self)
        .def(py::self -= py::self)
        .def(py::self -= ScalarT())
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

//PYBIND11_MODULE(volpy, m) {
PYBIND11_EMBEDDED_MODULE(volpy, m) {

    // ------------------------------------------------------------
    // voldata::Buf3D bindings

    py::class_<voldata::Buf3D<float>, std::shared_ptr<voldata::Buf3D<float>>>(m, "ImageDataFloat", py::buffer_protocol())
        .def_buffer([](voldata::Buf3D<float>& buf) -> py::buffer_info {
            return py::buffer_info(buf.data.data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    3,
                    { buf.stride.x, buf.stride.y, buf.stride.z },
                    { sizeof(float) * buf.stride.z * buf.stride.y, sizeof(float) * buf.stride.z, sizeof(float) });
        });

    // ------------------------------------------------------------
    // voldata::Volume bindings

    py::class_<voldata::Volume, std::shared_ptr<voldata::Volume>>(m, "Volume")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def(py::init<size_t, size_t, size_t, const uint8_t*>())
        .def(py::init<size_t, size_t, size_t, const float*>())
        .def("clear", &voldata::Volume::clear)
        .def("load_grid", &voldata::Volume::load_grid)
        .def("current_grid", &voldata::Volume::current_grid)
        .def("AABB", &voldata::Volume::AABB)
        .def("minorant_majorant", &voldata::Volume::minorant_majorant)
        .def_readwrite("albedo", &voldata::Volume::albedo)
        .def_readwrite("phase", &voldata::Volume::phase)
        .def_readwrite("density_scale", &voldata::Volume::density_scale)
        .def_readwrite("grid_frame", &voldata::Volume::grid_frame)
        .def("__repr__", &voldata::Volume::to_string, py::arg("indent") = "");

    // ------------------------------------------------------------
    // environment bindings

    py::class_<Environment, std::shared_ptr<Environment>>(m, "Environment")
        .def(py::init<std::string>())
        .def_readwrite("strength", &Environment::strength);

    // ------------------------------------------------------------
    // transferfunc bindings

    py::class_<TransferFunction, std::shared_ptr<TransferFunction>>(m, "TransferFunction")
        .def(py::init<const std::string&>())
        .def(py::init<const std::vector<glm::vec4>&>())
        .def_readwrite("window_left", &TransferFunction::window_left)
        .def_readwrite("window_width", &TransferFunction::window_width);

    // ------------------------------------------------------------
    // renderer bindings

    py::class_<Renderer>(m, "Renderer")
        .def_static("init", &Renderer::init,
            py::arg("w") = uint32_t(1920), py::arg("h") = uint32_t(1080), py::arg("vsync") = false, py::arg("pinned") = false, py::arg("visible") = false)
        .def_static("commit", []() {
            Renderer::commit();
            current_camera()->update();
            Renderer::sample = 0;
            Renderer::fbo->bind();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            Renderer::fbo->unbind();
        })
        .def_static("trace", &Renderer::trace)
        .def_static("draw", []() {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            Renderer::draw();
            Context::swap_buffers();
        })
        .def_static("resolution", []() {
            return Context::resolution();
        })
        .def_static("resize", [](int w, int h) {
            Context::resize(w, h);
        })
        .def_static("render", [](int spp) {
            Renderer::commit();
            current_camera()->update();
            Renderer::sample = 0;
            Renderer::fbo->bind();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            Renderer::fbo->unbind();
            while (Renderer::sample < spp)
                Renderer::trace();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            Renderer::draw();
            Context::swap_buffers();
        })
        .def_static("fbo_num_buffers", []() { return Renderer::fbo->color_textures.size(); })
        .def_static("fbo_data", [](uint32_t i = 0) {
            auto tex = Renderer::fbo->color_textures.at(i);
            auto buf = std::make_shared<voldata::Buf3D<float>>(glm::uvec3(tex->w, tex->h, 3));
            glBindTexture(GL_TEXTURE_2D, tex->id);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, &buf->data[0]);
            glBindTexture(GL_TEXTURE_2D, 0);
            return buf;
        })
        .def_static("save", [](const std::string& filename = "out.png") {
            Context::screenshot(filename);
        })
        .def_readwrite_static("volume", &Renderer::volume)
        .def_readwrite_static("environment", &Renderer::environment)
        .def_readwrite_static("transferfunc", &Renderer::transferfunc)
        .def_readwrite_static("cam_pos", &current_camera()->pos)
        .def_readwrite_static("cam_dir", &current_camera()->dir)
        .def_readwrite_static("cam_fov", &current_camera()->fov_degree)
        .def_readwrite_static("sample", &Renderer::sample)
        .def_readwrite_static("sppx", &Renderer::sppx)
        .def_readwrite_static("bounces", &Renderer::bounces)
        .def_readwrite_static("seed", &Renderer::seed)
        .def_readwrite_static("tonemap_exposure", &Renderer::tonemap_exposure)
        .def_readwrite_static("tonemap_gamma", &Renderer::tonemap_gamma)
        .def_readwrite_static("tonemapping", &Renderer::tonemapping)
        .def_readwrite_static("show_environment", &Renderer::show_environment)
        .def_readwrite_static("vol_crop_min", &Renderer::vol_crop_min)
        .def_readwrite_static("vol_crop_max", &Renderer::vol_crop_max)
        .def_static("shutdown", []() { exit(0); });

    // ------------------------------------------------------------
    // glm vector bindings

    register_vector_operators<glm::vec2, float>(
        py::class_<glm::vec2>(m, "vec2")
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<float, float>())
            .def_readwrite("x", &glm::vec2::x)
            .def_readwrite("y", &glm::vec2::y)
            .def("normalize", [](const glm::vec2& v) { return glm::normalize(v); })
            .def("length", [](const glm::vec2& v) { return glm::length(v); })
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
            .def("normalize", [](const glm::vec3& v) { return glm::normalize(v); })
            .def("length", [](const glm::vec3& v) { return glm::length(v); })
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
            .def("normalize", [](const glm::vec4& v) { return glm::normalize(v); })
            .def("length", [](const glm::vec4& v) { return glm::length(v); })
            .def("__repr__", [](const glm::vec4& v) {
                return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + ")";
            }));

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
}
