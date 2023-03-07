#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cppgl.h>
#include <voldata.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "renderer.h"
#include "environment.h"
#include "transferfunc.h"

using namespace cppgl;

// ------------------------------------------------------------------------
// python bindings

template <typename VecT, typename ScalarT>
pybind11::class_<VecT> register_vector_operators(pybind11::class_<VecT>& pyclass) {
    return pyclass
        .def(pybind11::self + pybind11::self)
        .def(pybind11::self + ScalarT())
        .def(ScalarT() + pybind11::self)
        .def(pybind11::self += pybind11::self)
        .def(pybind11::self += ScalarT())
        .def(pybind11::self - pybind11::self)
        .def(pybind11::self - ScalarT())
        .def(ScalarT() - pybind11::self)
        .def(pybind11::self -= pybind11::self)
        .def(pybind11::self -= ScalarT())
        .def(pybind11::self * pybind11::self)
        .def(pybind11::self * ScalarT())
        .def(ScalarT() * pybind11::self)
        .def(pybind11::self *= pybind11::self)
        .def(pybind11::self *= ScalarT())
        .def(pybind11::self / pybind11::self)
        .def(pybind11::self / ScalarT())
        .def(ScalarT() / pybind11::self)
        .def(pybind11::self /= pybind11::self)
        .def(pybind11::self /= ScalarT())
        .def(-pybind11::self);
}

template <typename MatT, typename ScalarT>
pybind11::class_<MatT> register_matrix_operators(pybind11::class_<MatT>& pyclass) {
    return pyclass
        .def(pybind11::self + pybind11::self)
        .def(pybind11::self += pybind11::self)
        .def(pybind11::self - pybind11::self)
        .def(pybind11::self -= pybind11::self)
        .def(pybind11::self * pybind11::self)
        .def(pybind11::self * ScalarT())
        .def(ScalarT() * pybind11::self)
        .def(pybind11::self *= pybind11::self)
        .def(pybind11::self *= ScalarT())
        .def(-pybind11::self);
}

PYBIND11_EMBEDDED_MODULE(volpy, m) {

    // ------------------------------------------------------------
    // voldata::Buf3D bindings

    pybind11::class_<voldata::Buf3D<float>, std::shared_ptr<voldata::Buf3D<float>>>(m, "ImageDataFloat", pybind11::buffer_protocol())
        .def_buffer([](voldata::Buf3D<float>& buf) -> pybind11::buffer_info {
            return pybind11::buffer_info(buf.data.data(),
                    sizeof(float),
                    pybind11::format_descriptor<float>::format(),
                    3,
                    { buf.stride.x, buf.stride.y, buf.stride.z },
                    { sizeof(float) * buf.stride.z * buf.stride.y, sizeof(float) * buf.stride.z, sizeof(float) });
        });

    // ------------------------------------------------------------
    // voldata::Volume bindings

    pybind11::class_<voldata::Volume, std::shared_ptr<voldata::Volume>>(m, "Volume")
        .def(pybind11::init<>())
        .def(pybind11::init<std::string>())
        .def(pybind11::init<size_t, size_t, size_t, const uint8_t*>())
        .def(pybind11::init<size_t, size_t, size_t, const float*>())
        .def("load_grid", &voldata::Volume::load_grid)
        .def("clear", &voldata::Volume::clear)
        .def("add_grid_frame", &voldata::Volume::add_grid_frame)
        .def("update_grid_frame", &voldata::Volume::update_grid_frame)
        .def("AABB", &voldata::Volume::AABB) // TODO default argument
        .def_readwrite("grid_frame_counter", &voldata::Volume::grid_frame_counter)
        .def("minorant_majorant", &voldata::Volume::minorant_majorant)
        .def("__repr__", &voldata::Volume::to_string, pybind11::arg("indent") = "");

    // ------------------------------------------------------------
    // environment bindings

    pybind11::class_<Environment, std::shared_ptr<Environment>>(m, "Environment")
        .def(pybind11::init<std::string>())
        .def_readwrite("strength", &Environment::strength);

    // ------------------------------------------------------------
    // transferfunc bindings

    pybind11::class_<TransferFunction, std::shared_ptr<TransferFunction>>(m, "TransferFunction")
        .def(pybind11::init<const std::string&>())
        .def(pybind11::init<const std::vector<glm::vec4>&>())
        .def_readwrite("window_left", &TransferFunction::window_left)
        .def_readwrite("window_width", &TransferFunction::window_width);

    // ------------------------------------------------------------
    // renderer bindings

    pybind11::class_<RendererOpenGL, std::shared_ptr<RendererOpenGL>>(m, "Renderer")
        .def(pybind11::init<>())
        .def("init", &RendererOpenGL::init)
        .def("commit", &RendererOpenGL::commit)
        .def("trace", &RendererOpenGL::trace)
        .def("reset", &RendererOpenGL::reset)
        .def("scale_and_move_to_unit_cube", &RendererOpenGL::scale_and_move_to_unit_cube)
        .def("render", [](const std::shared_ptr<RendererOpenGL>& renderer, int spp) {
            current_camera()->update();
            renderer->sample = 0;
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            while (renderer->sample < spp) {
                renderer->trace();
                Context::swap_buffers(); // keep interactivity
            }
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
            const glm::ivec2 size = Context::resolution();
            std::vector<uint8_t> pixels(size.x * size.y * 3);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, size.x, size.y, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
            const fs::path outfile = fs::path(filename);
            image_store_ldr(outfile, pixels.data(), size.x, size.y, 3, true, true);
            std::cout << outfile << " written." << std::endl;
        })
        .def("save_with_alpha", [](const std::shared_ptr<RendererOpenGL>& renderer, const std::string& filename = "out.png") {
            const glm::ivec2 size = Context::resolution();
            std::vector<uint8_t> pixels(size.x * size.y * 4);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
            const fs::path outfile = fs::path(filename).replace_extension(".png");
            image_store_ldr(outfile, pixels.data(), size.x, size.y, 4, true, true);
            std::cout << outfile << " written." << std::endl;
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
        .def_readwrite("albedo", &RendererOpenGL::albedo)
        .def_readwrite("phase", &RendererOpenGL::phase)
        .def_readwrite("density_scale", &RendererOpenGL::density_scale)
        .def_readwrite("emission_scale", &RendererOpenGL::emission_scale)
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
        // colmap stuff
        .def_static("colmap_view_trans", []() {
            const glm::mat4 GL_TO_COLMAP = glm::inverse(glm::mat4(1, 0, 0, 0,   0, -1, 0, 0,    0, 0, -1, 0,    0, 0, 0, 1));
            return glm::vec3((GL_TO_COLMAP * current_camera()->view)[3]);
        })
        .def_static("colmap_view_rot", []() {
            const glm::mat4 GL_TO_COLMAP = glm::inverse(glm::mat4(1, 0, 0, 0,   0, -1, 0, 0,    0, 0, -1, 0,    0, 0, 0, 1));
            return glm::normalize(glm::toQuat(GL_TO_COLMAP * current_camera()->view));
        })
        .def_static("colmap_focal_length", []() {
            return Context::resolution().y / (2 * tan(0.5 * glm::radians(current_camera()->fov_degree)));
        })
        .def_static("shutdown", []() {
            exit(0);
        });

    // ------------------------------------------------------------
    // glm vector bindings

    register_vector_operators<glm::vec2, float>(
        pybind11::class_<glm::vec2>(m, "vec2")
            .def(pybind11::init<>())
            .def(pybind11::init<float>())
            .def(pybind11::init<float, float>())
            .def_readwrite("x", &glm::vec2::x)
            .def_readwrite("y", &glm::vec2::y)
            .def("normalize", [](const glm::vec2& v) { return glm::normalize(v); })
            .def("length", [](const glm::vec2& v) { return glm::length(v); })
            .def("__repr__", [](const glm::vec2& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::vec3, float>(
        pybind11::class_<glm::vec3>(m, "vec3", pybind11::buffer_protocol())
            .def(pybind11::init<>())
            .def(pybind11::init<float>())
            .def(pybind11::init<float, float, float>())
            .def_readwrite("x", &glm::vec3::x)
            .def_readwrite("y", &glm::vec3::y)
            .def_readwrite("z", &glm::vec3::z)
            .def("normalize", [](const glm::vec3& v) { return glm::normalize(v); })
            .def("length", [](const glm::vec3& v) { return glm::length(v); })
            .def_buffer([](glm::vec3& m) -> pybind11::buffer_info {
                return pybind11::buffer_info(&m[0],
                        sizeof(float),
                        pybind11::format_descriptor<float>::format(),
                        1,
                        { 3 },
                        { sizeof(float) });
            })
            .def("__repr__", [](const glm::vec3& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::vec4, float>(
        pybind11::class_<glm::vec4>(m, "vec4", pybind11::buffer_protocol())
            .def(pybind11::init<>())
            .def(pybind11::init<float>())
            .def(pybind11::init<float, float, float, float>())
            .def_readwrite("x", &glm::vec4::x)
            .def_readwrite("y", &glm::vec4::y)
            .def_readwrite("z", &glm::vec4::z)
            .def_readwrite("w", &glm::vec4::w)
            .def("normalize", [](const glm::vec4& v) { return glm::normalize(v); })
            .def("length", [](const glm::vec4& v) { return glm::length(v); })
            .def_buffer([](glm::vec4& m) -> pybind11::buffer_info {
                return pybind11::buffer_info(&m[0],
                        sizeof(float),
                        pybind11::format_descriptor<float>::format(),
                        1,
                        { 4 },
                        { sizeof(float) });
            })
            .def("__repr__", [](const glm::vec4& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::ivec2, int>(
        pybind11::class_<glm::ivec2>(m, "ivec2")
            .def(pybind11::init<>())
            .def(pybind11::init<int>())
            .def(pybind11::init<int, int>())
            .def_readwrite("x", &glm::ivec2::x)
            .def_readwrite("y", &glm::ivec2::y)
            .def("__repr__", [](const glm::ivec2& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::ivec3, int>(
        pybind11::class_<glm::ivec3>(m, "ivec3")
            .def(pybind11::init<>())
            .def(pybind11::init<int>())
            .def(pybind11::init<int, int, int>())
            .def_readwrite("x", &glm::ivec3::x)
            .def_readwrite("y", &glm::ivec3::y)
            .def_readwrite("z", &glm::ivec3::z)
            .def("__repr__", [](const glm::ivec3& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::ivec4, int>(
        pybind11::class_<glm::ivec4>(m, "ivec4")
            .def(pybind11::init<>())
            .def(pybind11::init<int>())
            .def(pybind11::init<int, int, int, int>())
            .def_readwrite("x", &glm::ivec4::x)
            .def_readwrite("y", &glm::ivec4::y)
            .def_readwrite("z", &glm::ivec4::z)
            .def_readwrite("w", &glm::ivec4::w)
            .def("__repr__", [](const glm::ivec4& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::uvec2, uint32_t>(
        pybind11::class_<glm::uvec2>(m, "uvec2")
            .def(pybind11::init<>())
            .def(pybind11::init<uint32_t>())
            .def(pybind11::init<uint32_t, uint32_t>())
            .def_readwrite("x", &glm::uvec2::x)
            .def_readwrite("y", &glm::uvec2::y)
            .def("__repr__", [](const glm::uvec2& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::uvec3, uint32_t>(
        pybind11::class_<glm::uvec3>(m, "uvec3")
            .def(pybind11::init<>())
            .def(pybind11::init<uint32_t>())
            .def(pybind11::init<uint32_t, uint32_t, uint32_t>())
            .def_readwrite("x", &glm::uvec3::x)
            .def_readwrite("y", &glm::uvec3::y)
            .def_readwrite("z", &glm::uvec3::z)
            .def("__repr__", [](const glm::uvec3& v) {
                return glm::to_string(v);
            }));

    register_vector_operators<glm::uvec4, uint32_t>(
        pybind11::class_<glm::uvec4>(m, "uvec4")
            .def(pybind11::init<>())
            .def(pybind11::init<uint32_t>())
            .def(pybind11::init<uint32_t, uint32_t, uint32_t, uint32_t>())
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
        pybind11::class_<glm::mat3>(m, "mat3", pybind11::buffer_protocol())
            .def(pybind11::init<>())
            .def(pybind11::init<float>())
            .def(pybind11::init<glm::vec3, glm::vec3, glm::vec3>())
            .def("column", [](const std::shared_ptr<glm::mat3>& m, uint32_t i) {
                return m->operator[](i);
            })
            .def("value", [](const std::shared_ptr<glm::mat3>& m, uint32_t i, uint32_t j) {
                return m->operator[](i)[j];
            })
            .def_buffer([](glm::mat3& m) -> pybind11::buffer_info {
                return pybind11::buffer_info(&m[0],
                        sizeof(float),
                        pybind11::format_descriptor<float>::format(),
                        2,
                        { 3, 3 },
                        { sizeof(float) * 3, sizeof(float) });
            })
            .def("__repr__", [](const glm::mat3& m) {
                return glm::to_string(m);
            }));

    register_matrix_operators<glm::mat4, float>(
        pybind11::class_<glm::mat4>(m, "mat4", pybind11::buffer_protocol())
            .def(pybind11::init<>())
            .def(pybind11::init<float>())
            .def(pybind11::init<glm::vec4, glm::vec4, glm::vec4, glm::vec4>())
            .def("column", [](const std::shared_ptr<glm::mat4>& m, uint32_t i) {
                return m->operator[](i);
            })
            .def("value", [](const std::shared_ptr<glm::mat4>& m, uint32_t i, uint32_t j) {
                return m->operator[](i)[j];
            })
            .def_buffer([](glm::mat4& m) -> pybind11::buffer_info {
                return pybind11::buffer_info(&m[0],
                        sizeof(float),
                        pybind11::format_descriptor<float>::format(),
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
        pybind11::class_<glm::quat>(m, "quat", pybind11::buffer_protocol())
            .def(pybind11::init<>())
            .def(pybind11::init<glm::vec3>())
            .def(pybind11::init<glm::mat3>())
            .def(pybind11::init<glm::mat4>())
            .def_readwrite("x", &glm::quat::x)
            .def_readwrite("y", &glm::quat::y)
            .def_readwrite("z", &glm::quat::z)
            .def_readwrite("w", &glm::quat::w)
            .def_buffer([](glm::quat& m) -> pybind11::buffer_info {
                return pybind11::buffer_info(&m[0],
                        sizeof(float),
                        pybind11::format_descriptor<float>::format(),
                        1,
                        { 4 },
                        { sizeof(float) });
            })
            .def("__repr__", [](const glm::quat& v) {
                return glm::to_string(v);
            }));
}
