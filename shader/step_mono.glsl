#version 450 core

layout (local_size_x = 512) in;

layout(std430, binding = 0) buffer ParameterBuffer {
    vec4 parameters[];
};

layout(std430, binding = 1) buffer ParameterBackBuffer {
    vec4 parameters_back[];
};

uniform uint n_parameters;

// ---------------------------------------------------
// main

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= n_parameters) return;

    // load current parameters
    const vec4 x = parameters[idx];

    // ensure monotonic extinction TF
    const float lower = parameters[max(0, idx-1)].a;
    const float upper = parameters[min(idx+1, n_parameters-1)].a;
    const vec4 y = vec4(x.rgb, clamp(x.a, lower, upper));

    // store updated parameters
    parameters_back[idx] = y;
}
