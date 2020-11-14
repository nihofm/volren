# target name to generate here
set(TARGET vol)

# glob source files
file(GLOB_RECURSE HEADERS "*.h")
file(GLOB_RECURSE SOURCES "*.cpp" "*.cu")

# define target
add_executable(${TARGET} ${HEADERS} ${SOURCES})
pybind11_add_module(volpy ${SOURCES})

# install
install(TARGETS ${TARGET} DESTINATION bin)
install(DIRECTORY . DESTINATION include/${TARGET} FILES_MATCHING PATTERN "*.h")

# ----------------------------------------------------------
# dependencies

target_link_libraries(${TARGET} cppgl stdc++fs)

# pybind11
find_package(Python3 COMPONENTS Development REQUIRED)
target_include_directories(${TARGET} PUBLIC ${Python3_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
target_link_libraries(${TARGET} ${CUDA_LIBRARIES})

# openvdb (optional)
if (WITH_OPENVDB)
    target_link_libraries(${TARGET} openvdb Half tbb)
ENDIF()

# pytorch (optional)
if (WITH_TORCH)
    find_package(Torch REQUIRED)
    target_include_directories(${TARGET} PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${TORCH_LIBRARIES})
endif()
