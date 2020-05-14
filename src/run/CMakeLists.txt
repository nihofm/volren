# target name to generate here
set(TARGET run)

# glob source files
file(GLOB_RECURSE HEADERS "*.h")
file(GLOB_RECURSE SOURCES "*.cpp")

# define target
add_executable(${TARGET} ${SOURCES} ${HEADERS})
set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# install
install(TARGETS ${TARGET} DESTINATION .)

# ----------------------------------------------------------
# dependencies

# threads
find_package(Threads REQUIRED)
target_link_libraries(${TARGET} ${CMAKE_THREAD_LIBS_INIT})

# openmp
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
include_directories(${TARGET} ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(${TARGET} ${OpenMP_CXX_LIBRARIES})

# opengl
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
target_include_directories(${TARGET} PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(${TARGET} ${OPENGL_LIBRARIES})

# glew
find_package(GLEW REQUIRED)
target_include_directories(${TARGET} PUBLIC ${GLEW_INCLUDE_DIR})
target_link_libraries(${TARGET} ${GLEW_LIBRARIES})

# std::filesystem
target_link_libraries(${TARGET} stdc++fs)

# pytorch (optional)
if (WITH_TORCH)
    find_package(Torch REQUIRED)
    target_include_directories(${TARGET} PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${TORCH_LIBRARIES})
endif()

# built libs
target_link_libraries(${TARGET} cppgl)
