cmake_minimum_required(VERSION 3.16)

PROJECT(volren
    VERSION 0.3
    DESCRIPTION "Simple volume rendering framework."
    LANGUAGES C CXX
)

# ---------------------------------------------------------------------
# options

option(WITH_PYTHON_BINDINGS "Build standalone python module?" ON)

# ---------------------------------------------------------------------
# dependencies

add_subdirectory(submodules/cppgl)
include_directories(submodules/cppgl/src)

set(VOLDATA_BUILD_GLM OFF) # avoid including GLM twice
add_subdirectory(submodules/voldata)
include_directories(submodules/voldata/src)
include_directories(submodules/voldata/submodules)

if (WITH_PYTHON_BINDINGS)
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

# ---------------------------------------------------------------------
# directory management

include_directories(src)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------
# compile C++ version

add_executable(volren src/main.cpp src/environment.cpp src/renderer.cpp src/transferfunc.cpp)
target_link_libraries(volren stdc++ stdc++fs dl cppgl voldata)

# ---------------------------------------------------------------------
# compile python version

if (WITH_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development)
    add_library(volpy SHARED src/bindings.cpp src/environment.cpp src/renderer.cpp src/transferfunc.cpp)
    target_link_libraries(volpy stdc++ stdc++fs dl cppgl voldata ${PYTHON_LIBRARIES} pybind11::module)
	pybind11_extension(volpy)
endif()