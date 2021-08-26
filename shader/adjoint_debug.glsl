#version 450 core

#extension GL_NV_shader_atomic_float : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color_adjoint;
layout (binding = 1, rgba32f) uniform image2D color_reference;
layout (binding = 2, rgba32f) uniform image2D color_output;
layout (binding = 3, rgba32f) uniform image2D color_debug;

// ---------------------------------------------------
// helper funcs

float sum(const vec3 x) { return x.x + x.y + x.z; }
float mean(const vec3 x) { return sum(x) * (1.f / 3.f); }
vec3 sanitize(const vec3 x) { return mix(x, vec3(0), isnan(x) || isinf(x)); }
vec3 visualize_grad(const float grad) { return abs(grad) * (sign(grad) > 0.f ? vec3(1, 0, 0) : vec3(0, 0, 1)); }

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 size = imageSize(color_adjoint);
    if (any(greaterThanEqual(pixel, size))) return;

    // ------------------------------------------
    // 4-way debug visualization
    
    vec3 out_col;
    if (pixel.y < size.y / 2) {
        if (pixel.x < size.x / 2) { // bottom left
            const ivec2 pixel_adj = ivec2(pixel.x * 2, pixel.y * 2);
            const vec3 col_adj = imageLoad(color_adjoint, pixel_adj).rgb;
            const vec3 col_ref = imageLoad(color_reference, pixel_adj).rgb;
            const vec3 l2_grad = 2 * (col_adj - col_ref);
            out_col = abs(l2_grad);
        } else {                    // bottom right
            const ivec2 pixel_adj = ivec2((pixel.x - size.x / 2) * 2, pixel.y * 2);
            out_col = imageLoad(color_debug, pixel_adj).rgb;
            // const vec3 col_adj = imageLoad(color_adjoint, pixel_adj).rgb;
            // const vec3 col_ref = imageLoad(color_reference, pixel_adj).rgb;
            // const vec3 l2_grad = 2 * (col_adj - col_ref);
            // out_col = visualize_grad(sum(l2_grad));
        }
    } else {
        if (pixel.x < size.x / 2) { // top left
            const ivec2 pixel_adj = ivec2(pixel.x * 2, (pixel.y - size.y / 2) * 2);
            out_col = imageLoad(color_adjoint, pixel_adj).rgb;
        } else {                    // top right
            const ivec2 pixel_adj = ivec2((pixel.x - size.x / 2) * 2, (pixel.y - size.y / 2) * 2);
            out_col = imageLoad(color_reference, pixel_adj).rgb;
        }
    }
    imageStore(color_output, pixel, vec4(sanitize(out_col), 1));
}
