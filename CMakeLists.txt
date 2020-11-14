project(vol LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.10)

# ---------------------------------------------------------------------
# path management

# output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# ---------------------------------------------------------------------
# options

# OpenVDB
option(WITH_OPENVDB "Enable OpenVDB." OFF)

if (WITH_OPENVDB)
    add_definitions(-DWITH_OPENVDB)
else()
    add_definitions(-UWITH_OPENVDB)
endif()

# pytorch (set CMAKE_PREFIX_PATH=/path/to/libtorch)
option(WITH_TORCH "Enable PyTorch." OFF)

if (WITH_TORCH)
    add_definitions(-DWITH_TORCH)
else()
    add_definitions(-UWITH_TORCH)
endif()

# ---------------------------------------------------------------------
# submodules

add_subdirectory(submodules/cppgl)
add_subdirectory(submodules/pybind11)

include_directories(external)
include_directories(submodules)
include_directories(submodules/cppgl/src)

# ---------------------------------------------------------------------
# compiler flags

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-pthread -fopenmp -march=native)
add_compile_options(-Wall -Wno-unused-parameter -Wno-unused-variable -Wno-int-to-pointer-cast)

# ---------------------------------------------------------------------
# install shaders

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shader DESTINATION bin)

# ---------------------------------------------------------------------
# traverse source tree

include_directories(src)
add_subdirectory(src)
