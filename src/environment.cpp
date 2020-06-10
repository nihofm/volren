#include "environment.h"

EnvironmentImpl::EnvironmentImpl(const Texture2D& texture) : model(1), texture(texture), strength(1) {
    // build cdf
    std::vector<std::vector<float>> conditional;
    std::vector<float> marginal;
    integral = build_cdf_2D(texture, conditional, marginal);
    // upload conditional
    cdf_U = SSBO("TODO_env_conditional");
    cdf_U->resize(conditional[0].size() * conditional.size() * sizeof(float));
    float* gpu = (float*)cdf_U->map(GL_WRITE_ONLY);
    for (size_t y = 0; y < conditional.size(); ++y)
        for (size_t x = 0; x < conditional[y].size(); ++x)
            gpu[y * conditional[y].size() + x] = conditional[y][x];
    cdf_U->unmap();
    // upload marginal
    cdf_V = SSBO("TODO_env_marginal");
    cdf_V->resize(marginal.size() * sizeof(float));
    gpu = (float*) cdf_V->map(GL_WRITE_ONLY);
    for (size_t y = 0; y < marginal.size(); ++y)
        gpu[y] = marginal[y];
    cdf_V->unmap();
}

EnvironmentImpl::~EnvironmentImpl() {}

float EnvironmentImpl::build_cdf_1D(const float* f, uint32_t N, std::vector<float>& cdf) {
    cdf.resize(N + 1);
    // build cdf and save integral
    float integral = 0;
    cdf[0] = 0;
    for (uint32_t i = 1; i < N + 1; ++i)
        cdf[i] = cdf[i - 1] + f[i - 1];
    integral = cdf[N];
    // ensure density
    for (uint32_t i = 1; i < N + 1; ++i)
        cdf[i] = integral != 0.f ? cdf[i] / integral : float(i) / float(N);
    return integral / N; // unit integral
}

float EnvironmentImpl::build_cdf_2D(const Texture2D& tex, std::vector<std::vector<float>>& conditional, std::vector<float>& marginal) {
    conditional.clear(); marginal.clear();
    // download texture data
    std::vector<glm::vec3> buf(tex->w * tex->h);
    glBindTexture(GL_TEXTURE_2D, tex->id);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, buf.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    // compute luma per pixel and counter spherical distortion
    std::vector<float> func(buf.size());
    for (int y = 0; y < tex->h; ++y) {
        const float sin_t = sinf(M_PI * float(y + .5f) / float(tex->h));
        for (int x = 0; x < tex->w; ++x)
            func[y * tex->w + x] = sin_t * glm::dot(glm::vec3(buf[y * tex->w + x]), glm::vec3(0.212671f, 0.715160f, 0.072169f));
    }
    // build conditional distributions
    std::vector<float> conditional_integrals;
    for (int y = 0; y < tex->h; ++y) {
        conditional.push_back(std::vector<float>());
        conditional_integrals.push_back(build_cdf_1D(&func[y * tex->w], tex->w, conditional[y]));
    }
    // build marginal distribution and return overall integral
    return build_cdf_1D(&conditional_integrals[0], tex->h, marginal);
}
