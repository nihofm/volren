#version 430
in vec2 tc;
out vec4 out_col;

uniform sampler2D color;
uniform sampler2D even; 

float luma(const vec3 col) { return dot(col, vec3(0.212671f, 0.715160f, 0.072169f)); }

vec3 heatmap(const float val) {
    const float hue = 251.1 / 360.f; // blue
    const vec3 hsv = vec3(hue + clamp(val, 0.f, 1.f) * -hue, 1, val < 1e-4 ? 0 : 1); // from blue to red
    // map hsv to rgb
    const vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    const vec3 p = abs(fract(vec3(hsv.x) + vec3(K)) * 6.f - vec3(K.w));
    return hsv.z * mix(vec3(K.x), clamp(p - vec3(K.x), vec3(0.f), vec3(1.f)), hsv.y);
}

void main() {
    vec3 col = texture(color, tc).rgb;
    vec3 eve = texture(even, tc).rgb;
    out_col = vec4(abs(col - eve) / max(1e-4, luma(col)), 1);
    out_col.rgb = heatmap(luma(out_col.rgb));
}
