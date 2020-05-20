#version 130
in vec2 tc;
uniform sampler2D tex;
uniform float exposure;
out vec4 out_col;

vec3 hable(in vec3 rgb) {
    const float A = 0.15f;
    const float B = 0.50f;
    const float C = 0.10f;
    const float D = 0.20f;
    const float E = 0.02f;
    const float F = 0.30f;
    return ((rgb * (A * rgb + C * B) + D * E) / (rgb * (A * rgb + B) + D * F)) - E / F;
}
vec3 hable_tonemap(in vec3 rgb, in float exposure) {
    const float W = 11.2f;
    return hable(exposure * rgb) / hable(glm::vec3(W));
}

void main() {
    out_col = texture(tex, tc);
    out_col.rgb = hable_tonemap(out_col.rgb, exposure);
}
