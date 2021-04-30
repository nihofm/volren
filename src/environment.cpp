#include "environment.h"

// importance map parameters (power of two!)
const uint32_t DIMENSION = 512;
const uint32_t SAMPLES = 64;

EnvironmentImpl::EnvironmentImpl(const std::string& name, const Texture2D& envmap) :
    name(name),
    model(1),
    strength(1),
    envmap(envmap),
    impmap("env_importance", DIMENSION, DIMENSION, GL_R32F, GL_RED, GL_FLOAT)
{
    // build importance map
    static Shader setup_shader = Shader("env_setup", "shader/env_setup.glsl");
    const uint32_t n_samples = (uint32_t)std::sqrt(SAMPLES);
    setup_shader->bind();
    impmap->bind_image(0, GL_WRITE_ONLY, GL_R32F);
    setup_shader->uniform("envmap", envmap, 0);
    setup_shader->uniform("output_size", glm::ivec2(DIMENSION));
    setup_shader->uniform("output_size_samples", glm::ivec2(DIMENSION * n_samples, DIMENSION * n_samples));
    setup_shader->uniform("num_samples", glm::ivec2(n_samples, n_samples));
    setup_shader->uniform("inv_samples", 1.f / (n_samples * n_samples));
    setup_shader->dispatch_compute(DIMENSION, DIMENSION);
    impmap->unbind_image(0);
    setup_shader->unbind();
    impmap->bind(0);
    glGenerateMipmap(GL_TEXTURE_2D);
    impmap->unbind();

    /*
    // TODO test envmap setup
    static Shader test_shader = Shader("env_test", "shader/env_test.glsl");
    Texture2D test_texture = Texture2D("env_test", DIMENSION, DIMENSION, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    test_shader->bind();
    uint32_t tex_unit = 0;
    set_uniforms(test_shader, tex_unit);
    test_shader->uniform("global_seed", 42);
    test_texture->bind_image(0, GL_WRITE_ONLY, GL_RGBA32F);
    test_shader->dispatch_compute(DIMENSION, DIMENSION);
    test_texture->unbind_image(0);
    test_shader->unbind();
    // write results
    envmap->save_jpg("env_envmap.jpg");
    impmap->save_jpg("env_impmap.jpg");
    test_texture->save_jpg("env_testmap.jpg");
    */
}

EnvironmentImpl::~EnvironmentImpl() {}

int EnvironmentImpl::num_mip_levels() const {
    return 1 + floor(log2(DIMENSION));
}

void EnvironmentImpl::set_uniforms(const Shader& shader, uint32_t& texture_unit) const {
    shader->uniform("env_model", model);
    shader->uniform("env_inv_model", glm::inverse(model));
    shader->uniform("env_strength", strength);
    shader->uniform("env_imp_inv_dim", glm::vec2(1.f / DIMENSION));
    shader->uniform("env_imp_base_mip", int(floor(log2(DIMENSION))));
    shader->uniform("env_envmap", envmap, texture_unit++);
    shader->uniform("env_impmap", impmap, texture_unit++);
}
