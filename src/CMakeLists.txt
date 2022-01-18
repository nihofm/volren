file(GLOB SOURCES "*.cpp")

add_executable(volren ${SOURCES})
target_link_libraries(volren cppgl voldata pybind11::embed stdc++ stdc++fs dl)
