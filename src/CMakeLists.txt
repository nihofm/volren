# target name to generate here
set(TARGET vol)

# glob source files
file(GLOB_RECURSE HEADERS "./*.h" "./*.cuh")
file(GLOB_RECURSE SOURCES "./*.cpp" "./*.cu")

# define targets
add_executable(${TARGET} ${HEADERS} ${SOURCES})
set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# install
install(TARGETS ${TARGET} DESTINATION bin)

# ----------------------------------------------------------
# dependencies

target_link_libraries(${TARGET} PRIVATE cppgl voldata stdc++ stdc++fs dl)

# CUDA
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(${TARGET} PRIVATE CUDA::cudart CUDA::cuda_driver)

# NVRTC
add_definitions(-DNVRTC_CUDA_INCLUDE="${CUDAToolkit_INCLUDE_DIRS}")
add_definitions(-DNVRTC_OPTIX_INCLUDE="${CMAKE_CURRENT_SOURCE_DIR}/../submodules/optix/include")
target_link_libraries(${TARGET} PRIVATE CUDA::nvrtc)

# pybind11
target_link_libraries(${TARGET} PRIVATE pybind11::embed)
