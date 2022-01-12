#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "../buffer.cuh"
#include "../camera.cuh"
#include "../grid.cuh"

extern "C" void call_trace_kernel(BufferCUDA<glm::vec4> fbo, CameraCUDA cam, DenseGridCUDA grid, uint32_t sample);