file(GLOB SOURCES "*.cpp")

add_executable(volren ${SOURCES})
target_link_libraries(volren stdc++ stdc++fs dl cppgl voldata pybind11::embed)
