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

# OpenVDB
option(WITH_OPENVDB "Enable OpenVDB." OFF)

if (WITH_OPENVDB)
    add_definitions(-DWITH_OPENVDB)
else()
    add_definitions(-UWITH_OPENVDB)
endif()

option(WITH_DCMTK "Enable DICOM toolkit." ON)

if (WITH_DCMTK)
    add_definitions(-DWITH_DCMTK)
else()
    add_definitions(-UWITH_DCMTK)
endif()

# ---------------------------------------------------------------------
# external projects

include(ExternalProject)

# directory setup for external projects
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
include_directories(${EXTERNAL_DIR}/include)
link_directories(${EXTERNAL_DIR}/lib)

# CppGL
ExternalProject_Add(cppgl-ext
    GIT_REPOSITORY  git@git9.cs.fau.de:renderingframeworks/cppgl.git
    #GIT_TAG         v0.2
    GIT_TAG         dev
    GIT_SHALLOW     ON
    INSTALL_DIR     ${EXTERNAL_DIR}
    CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -Wno-dev
)

# OpenVDB
if (WITH_OPENVDB)
    ExternalProject_Add(openvdb-ext
        GIT_REPOSITORY  https://github.com/AcademySoftwareFoundation/openvdb
        GIT_TAG         v7.0.0
        INSTALL_DIR     ${EXTERNAL_DIR}
        CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -Wno-dev -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_CORE_STATIC=OFF
    )
endif()

# DCMTK
if (WITH_DCMTK)
    ExternalProject_Add(dcmtk-ext
        GIT_REPOSITORY  https://github.com/DCMTK/dcmtk
        GIT_TAG         DCMTK-3.6.5
        INSTALL_DIR     ${EXTERNAL_DIR}
        CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -Wno-dev -DBUILD_APPS=OFF -DBUILD_SHARED_LIBS=ON
    )
endif()

# install rules for external projects
#install(DIRECTORY ${EXTERNAL_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${EXTERNAL_DIR}/lib DESTINATION ${CMAKE_INSTALL_PREFIX})
list(APPEND CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/.. $ORIGIN/../lib)

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
