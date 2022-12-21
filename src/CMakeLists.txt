file(GLOB SOURCES "*.cpp")

if (VOLREN_BUILD_CUDA)
    file(GLOB SOURCES_CU "*.cu")
endif()

add_executable(volren ${SOURCES} ${SOURCES_CU})
target_link_libraries(volren stdc++ stdc++fs dl cppgl voldata pybind11::embed)
