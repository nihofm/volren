#version 430
in vec2 tc;
uniform ivec2 size;
uniform float exposure;
uniform float gamma;
out vec4 out_col;

layout(std430, binding = 0) buffer Buffer {
    vec4 data[];
};

float luma(const vec3 col) { return dot(col, vec3(0.212671f, 0.715160f, 0.072169f)); }

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
    return hable(exposure * rgb) / hable(vec3(W));
}

void main() {
    const ivec2 gid = ivec2(gl_FragCoord.x, gl_FragCoord.y);
    if (any(greaterThanEqual(gid, size))) return;
    const uint idx = gid.y * size.x + gid.x;
    // tonemap
    const vec4 hdr = data[idx];

    const float scale = 1.f;
    out_col = scale * abs(hdr);
    return;

    const vec3 ldr = pow(hable_tonemap(hdr.rgb, exposure), vec3(1.f / gamma));
    out_col = vec4(ldr, hdr.a);
}
