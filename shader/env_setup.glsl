#version 450 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, r32f) uniform image2D impmap;

uniform sampler2D envmap;
uniform ivec2 output_size;
uniform ivec2 output_size_samples;
uniform ivec2 num_samples;
uniform float inv_samples;

// ---------------------------------------------------
// main

float luma(const vec3 col) { return dot(col, vec3(0.212671f, 0.715160f, 0.072169f)); }

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, output_size))) return;
    const ivec2 size_env = textureSize(envmap, 0);

    // compute importance
    float importance = 0.f;
    for (int y = 0; y < num_samples.y; ++y) {
        for (int x = 0; x < num_samples.x; ++x) {
            const vec2 uv = (pixel * num_samples + vec2(x + .5f, y + .5f)) / output_size_samples;
            importance += luma(texture(envmap, uv).rgb);
        }
    }

    // write averaged output
    imageStore(impmap, pixel, vec4(importance * inv_samples));
}
