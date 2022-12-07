cmake_minimum_required(VERSION 3.16)

PROJECT(volren
    VERSION 0.2
    DESCRIPTION "OpenGL volume rendering framework."
    LANGUAGES C CXX
)

# ---------------------------------------------------------------------
# options

option(VOLREN_BUILD_CPPGL "" ON)
option(VOLREN_BUILD_VOLDATA "" ON)
option(VOLREN_BUILD_PYBIND "" ON)

if (VOLREN_BUILD_CPPGL AND VOLREN_BUILD_VOLDATA)
    set(VOLDATA_BUILD_GLM OFF CACHE BOOL " " FORCE)
endif()

# ---------------------------------------------------------------------
# dependencies

if (VOLREN_BUILD_CPPGL)
    add_subdirectory(submodules/cppgl)
    include_directories(submodules/cppgl/src)
endif()

if (VOLREN_BUILD_VOLDATA)
    add_subdirectory(submodules/voldata)
    include_directories(submodules/voldata/include)
    include_directories(submodules/voldata/submodules/openvdb/nanovdb)
endif()

if (VOLREN_BUILD_PYBIND)
    add_subdirectory(submodules/pybind11)
    include_directories(submodules/pybind11/include)
endif()

# ---------------------------------------------------------------------
# compiler setup

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------
# directory management

include_directories(src)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------
# traverse source tree

add_subdirectory(src)
