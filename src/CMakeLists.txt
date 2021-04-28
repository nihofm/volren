# target name to generate here
set(TARGET vol)
set(TARGET_PY volpy)

# glob source files
file(GLOB_RECURSE HEADERS "*.h")
file(GLOB_RECURSE SOURCES "*.cpp" "*.cu")

# define targets
add_executable(${TARGET} ${HEADERS} ${SOURCES})
set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

#pybind11_add_module(${TARGET_PY} ${SOURCES})
#pybind11_strip(${TARGET_PY})
#pybind11_extension(${TARGET_PY})
#set_target_properties(${TARGET_PY} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# install
install(TARGETS ${TARGET} DESTINATION bin)
install(DIRECTORY . DESTINATION include/${TARGET} FILES_MATCHING PATTERN "*.h")

# ----------------------------------------------------------
# dependencies

target_link_libraries(${TARGET} cppgl voldata stdc++fs)
#target_link_libraries(${TARGET_PY} PRIVATE cppgl voldata stdc++fs)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${TARGET} ${CUDA_LIBRARIES})
#target_link_libraries(${TARGET_PY} PRIVATE ${CUDA_LIBRARIES})

# pybind11
find_package(Python3 COMPONENTS Development REQUIRED)
target_include_directories(${TARGET} PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(${TARGET} ${Python3_LIBRARIES} pybind11::embed)

# openvdb
target_link_libraries(${TARGET} openvdb Half tbb)
#target_link_libraries(${TARGET_PY} PRIVATE openvdb Half tbb)

# pytorch (optional)
if (WITH_TORCH)
    find_package(Torch REQUIRED)
    target_include_directories(${TARGET} PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${TORCH_LIBRARIES})
    #target_link_libraries(${TARGET_PY} PRIVATE ${TORCH_LIBRARIES})
endif()
