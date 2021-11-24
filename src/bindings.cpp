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
#include "renderer_gl.h"
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

template <typename MatT, typename ScalarT>
py::class_<MatT> register_matrix_operators(py::class_<MatT>& pyclass) {
    return pyclass
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self * ScalarT())
        .def(ScalarT() * py::self)
        .def(py::self *= py::self)
        .def(py::self *= ScalarT())
        .def(-py::self);
}

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
        .def_readwrite("grid_frame", &voldata::Volume::grid_frame_counter)
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

    py::class_<RendererOpenGL, std::shared_ptr<RendererOpenGL>>(m, "Renderer")
        .def(py::init<>())
        .def("init", &Renderer::init)
        .def("commit", &Renderer::commit)
        .def("trace", &RendererOpenGL::trace)
        .def("render", [](const std::shared_ptr<RendererOpenGL>& renderer, int spp) {
            current_camera()->update();
            renderer->sample = 0;
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            while (renderer->sample++ < spp)
                renderer->trace();
        })
        .def("draw", [](const std::shared_ptr<RendererOpenGL>& renderer) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderer->draw();
            Context::swap_buffers();
        })
        .def_static("resolution", []() {
            return Context::resolution();
        })
        .def("fbo_data", [](const std::shared_ptr<RendererOpenGL>& renderer) {
            auto tex = renderer->color;
            auto buf = std::make_shared<voldata::Buf3D<float>>(glm::uvec3(tex->w, tex->h, 3));
            glBindTexture(GL_TEXTURE_2D, tex->id);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, &buf->data[0]);
            glBindTexture(GL_TEXTURE_2D, 0);
            return buf;
        })
        .def("save", [](const std::shared_ptr<RendererOpenGL>& renderer, const std::string& filename = "out.png") {
            Context::screenshot(filename);
        })
        // members
        .def_readwrite("volume", &RendererOpenGL::volume)
        .def_readwrite("environment", &RendererOpenGL::environment)
        .def_readwrite("transferfunc", &RendererOpenGL::transferfunc)
        .def_readwrite("sample", &RendererOpenGL::sample)
        .def_readwrite("sppx", &RendererOpenGL::sppx)
        .def_readwrite("bounces", &RendererOpenGL::bounces)
        .def_readwrite("seed", &RendererOpenGL::seed)
        .def_readwrite("tonemap_exposure", &RendererOpenGL::tonemap_exposure)
        .def_readwrite("tonemap_gamma", &RendererOpenGL::tonemap_gamma)
        .def_readwrite("tonemapping", &RendererOpenGL::tonemapping)
        .def_readwrite("show_environment", &RendererOpenGL::show_environment)
        .def_readwrite("vol_clip_min", &RendererOpenGL::vol_clip_min)
        .def_readwrite("vol_clip_max", &RendererOpenGL::vol_clip_max)
        // camera
        .def_readwrite_static("cam_pos", &current_camera()->pos)
        .def_readwrite_static("cam_dir", &current_camera()->dir)
        .def_readwrite_static("cam_up", &current_camera()->up)
        .def_readwrite_static("cam_fov", &current_camera()->fov_degree)
        .def_readwrite_static("cam_near", &current_camera()->near)
        .def_readwrite_static("cam_far", &current_camera()->far)
        .def_readwrite_static("view_matrix", &current_camera()->view)
        .def_readwrite_static("proj_matrix", &current_camera()->proj)
        .def_static("cam_aspect", &current_camera()->aspect_ratio)
        .def_static("shutdown", []() { exit(0); })
        // colmap stuff
        .def_static("colmap_view_trans", []() {
            return glm::vec3(glm::inverse(current_camera()->view)[3]);
        })
        .def_static("colmap_view_rot", []() {
            // return glm::quat_cast(current_camera()->view);
            const glm::mat4 GL_TO_COLMAP = glm::inverse(glm::mat4(1, 0, 0, 0,   0, -1, 0, 0,    0, 0, -1, 0,    0, 0, 0, 1));
            return glm::quat_cast(GL_TO_COLMAP * current_camera()->view);
        })
        .def_static("colmap_focal_length", []() {
            // TODO different f_x and f_y params, or quadratic images only?
            return Context::resolution().y / (2 * tan(0.5 * glm::radians(current_camera()->fov_degree)));
        })
        .def_static("colmap_test_pos", [](const glm::quat& rot, const glm::vec3& trans) {
            glm::mat3 R = glm::mat3_cast(rot);
            glm::mat3 C = (-1.f*glm::transpose(R));
            glm::vec3 cam_pos = C*trans;
            return cam_pos;

            return -glm::transpose(glm::mat3_cast(rot)) * trans;
        })
        .def_static("colmap_test_view", [](const glm::quat& rot, const glm::vec3& trans) {
            const glm::mat4 COLMAP_TO_GL = glm::mat4(1, 0, 0, 0,   0, -1, 0, 0,    0, 0, -1, 0,    0, 0, 0, 1);
            glm::mat4 V = glm::mat4_cast(rot); // R mat
            V[3] = glm::vec4(trans.x, trans.y, trans.z, 1.f);
            return COLMAP_TO_GL * V;

            return glm::mat4_cast(rot) * glm::translate(glm::mat4(1), trans);
            const glm::mat4 RT = glm::transpose(glm::mat4_cast(rot));
            return -RT * glm::translate(glm::mat4(1), trans);
            // return glm::mat4(1, 0, 0, 0,   0, -1, 0, 0,    0, 0, -1, 0,    0, 0, 0, 1) * current_camera()->view;
        })
        .def_static("colmap_test_proj", [](float f_x, float f_y, float c_x, float c_y, float near, float far, uint32_t w, uint32_t h) {
            const glm::mat4 proj = glm::mat4( 
                2*f_x/w,  0.0,    (w - 2*c_x)/w, 0.0,
                0.0,    2*f_y/h, (h - 2*c_y)/h, 0.0,
                0.0, 0.0, (-far - near) / (far - near), -2.0*far*near/(far-near),
                0.0, 0.0, -1.0, 0.0);
            return glm::transpose(proj);
        });

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
                return glm::to_string(v);
            }));

    register_vector_operators<glm::vec3, float>(
        py::class_<glm::vec3>(m, "vec3", py::buffer_protocol())
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<float, float, float>())
            .def_readwrite("x", &glm::vec3::x)
            .def_readwrite("y", &glm::vec3::y)
            .def_readwrite("z", &glm::vec3::z)
            .def("normalize", [](const glm::vec3& v) { return glm::normalize(v); })
            .def("length", [](const glm::vec3& v) { return glm::length(v); })
            .def_buffer([](glm::vec3& m) -> py::buffer_info {
                return py::buffer_info(&m[0],
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        1,
                        { 3 },
                        { sizeof(float) });
            })
            .def("__repr__", [](const glm::vec3& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::vec4, float>(
        py::class_<glm::vec4>(m, "vec4", py::buffer_protocol())
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<float, float, float, float>())
            .def_readwrite("x", &glm::vec4::x)
            .def_readwrite("y", &glm::vec4::y)
            .def_readwrite("z", &glm::vec4::z)
            .def_readwrite("w", &glm::vec4::w)
            .def("normalize", [](const glm::vec4& v) { return glm::normalize(v); })
            .def("length", [](const glm::vec4& v) { return glm::length(v); })
            .def_buffer([](glm::vec4& m) -> py::buffer_info {
                return py::buffer_info(&m[0],
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        1,
                        { 4 },
                        { sizeof(float) });
            })
            .def("__repr__", [](const glm::vec4& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::ivec2, int>(
        py::class_<glm::ivec2>(m, "ivec2")
            .def(py::init<>())
            .def(py::init<int>())
            .def(py::init<int, int>())
            .def_readwrite("x", &glm::ivec2::x)
            .def_readwrite("y", &glm::ivec2::y)
            .def("__repr__", [](const glm::ivec2& v) {
                return glm::to_string(v);
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
                return glm::to_string(v);
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
                return glm::to_string(v);
            }));

    register_vector_operators<glm::uvec2, uint32_t>(
        py::class_<glm::uvec2>(m, "uvec2")
            .def(py::init<>())
            .def(py::init<uint32_t>())
            .def(py::init<uint32_t, uint32_t>())
            .def_readwrite("x", &glm::uvec2::x)
            .def_readwrite("y", &glm::uvec2::y)
            .def("__repr__", [](const glm::uvec2& v) {
                return glm::to_string(v);
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
                return glm::to_string(v);
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
                return glm::to_string(v);
            }));

    // ------------------------------------------------------------
    // glm matrix bindings

    register_matrix_operators<glm::mat3, float>(
        py::class_<glm::mat3>(m, "mat3", py::buffer_protocol())
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<glm::vec3, glm::vec3, glm::vec3>())
            .def("column", [](const std::shared_ptr<glm::mat3>& m, uint32_t i) {
                return m->operator[](i);
            })
            .def("value", [](const std::shared_ptr<glm::mat3>& m, uint32_t i, uint32_t j) {
                return m->operator[](i)[j];
            })
            .def_buffer([](glm::mat3& m) -> py::buffer_info {
                return py::buffer_info(&m[0],
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        2,
                        { 3, 3 },
                        { sizeof(float) * 3, sizeof(float) });
            })
            .def("__repr__", [](const glm::mat3& m) {
                return glm::to_string(m);
            }));

    register_matrix_operators<glm::mat4, float>(
        py::class_<glm::mat4>(m, "mat4", py::buffer_protocol())
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<glm::vec4, glm::vec4, glm::vec4, glm::vec4>())
            .def("column", [](const std::shared_ptr<glm::mat4>& m, uint32_t i) {
                return m->operator[](i);
            })
            .def("value", [](const std::shared_ptr<glm::mat4>& m, uint32_t i, uint32_t j) {
                return m->operator[](i)[j];
            })
            .def_buffer([](glm::mat4& m) -> py::buffer_info {
                return py::buffer_info(&m[0],
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        2,
                        { 4, 4 },
                        { sizeof(float) * 4, sizeof(float) });
            })
            .def("__repr__", [](const glm::mat4& m) {
                return glm::to_string(m);
            }));
    
    // ------------------------------------------------------------
    // glm quaternion bindings

    register_matrix_operators<glm::quat, float>(
        py::class_<glm::quat>(m, "quat", py::buffer_protocol())
            .def(py::init<>())
            .def(py::init<glm::vec3>())
            .def(py::init<glm::mat3>())
            .def(py::init<glm::mat4>())
            .def_readwrite("x", &glm::quat::x)
            .def_readwrite("y", &glm::quat::y)
            .def_readwrite("z", &glm::quat::z)
            .def_readwrite("w", &glm::quat::w)
            .def_buffer([](glm::quat& m) -> py::buffer_info {
                return py::buffer_info(&m[0],
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        1,
                        { 4 },
                        { sizeof(float) });
            })
            .def("__repr__", [](const glm::quat& v) {
                return glm::to_string(v);
            }));
}
