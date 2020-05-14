# target name to generate here
set(TARGET cppgl)

# glob source files
file(GLOB_RECURSE HEADERS "*.h")
file(GLOB_RECURSE SOURCES "*.cpp")

# define target
add_library(${TARGET} SHARED ${SOURCES} ${HEADERS})
set_target_properties(${TARGET} PROPERTIES VERSION 0.1)

# install
install(TARGETS ${TARGET} DESTINATION .)

# ----------------------------------------------------------
# dependencies

# opengl
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
target_include_directories(${TARGET} PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(${TARGET} ${OPENGL_LIBRARIES})

# glew
find_package(GLEW REQUIRED)
target_include_directories(${TARGET} PUBLIC ${GLEW_INCLUDE_DIR})
target_link_libraries(${TARGET} ${GLEW_LIBRARIES})

# glfw3
find_package(glfw3 REQUIRED)
target_link_libraries(${TARGET} glfw)

# std::filesystem
target_link_libraries(${TARGET} stdc++fs)
