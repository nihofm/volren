# target name to generate here
set(TARGET vol)

# glob source files
file(GLOB_RECURSE HEADERS "*.h")
file(GLOB_RECURSE SOURCES "*.cpp")

# define targets
add_executable(${TARGET} ${HEADERS} ${SOURCES})
set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# install
install(TARGETS ${TARGET} DESTINATION bin)

# ----------------------------------------------------------
# dependencies

target_link_libraries(${TARGET} PRIVATE cppgl voldata stdc++fs)

# CUDA
#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})
#target_link_libraries(${TARGET} PRIVATE ${CUDA_LIBRARIES})

# pybind11
target_link_libraries(${TARGET} PRIVATE pybind11::embed)
