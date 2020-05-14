project(vol)
cmake_minimum_required(VERSION 3.10)

# ---------------------------------------------------------------------
# path management

# output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# ---------------------------------------------------------------------
# options

# pytorch (set CMAKE_PREFIX_PATH=/path/to/libtorch)
option(WITH_TORCH "Enable PyTorch." OFF)

if (WITH_TORCH)
    add_definitions(-DWITH_TORCH)
else()
    add_definitions(-UWITH_TORCH)
endif()

# ---------------------------------------------------------------------
# compiler options

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-pthread -fopenmp -march=native)
add_compile_options(-Wall -Wno-unused-parameter -Wno-unused-variable -Wno-int-to-pointer-cast)

# ---------------------------------------------------------------------
# install

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shader DESTINATION .)# FILES_MATCHING PATTERN "*.bin")

# ---------------------------------------------------------------------
# traverse source tree

include_directories(src)
add_subdirectory(src)
