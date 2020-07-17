# target name to generate here
set(TARGET vol)

# glob source files
file(GLOB_RECURSE HEADERS "*.h")
file(GLOB_RECURSE SOURCES "*.cpp" "*.cu")

# define target
add_executable(${TARGET} ${HEADERS} ${SOURCES})

# install
install(TARGETS ${TARGET} DESTINATION bin)

# ----------------------------------------------------------
# dependencies

# threads
find_package(Threads REQUIRED)
target_link_libraries(${TARGET} ${CMAKE_THREAD_LIBS_INIT})

# opengl
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
target_include_directories(${TARGET} PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(${TARGET} ${OPENGL_LIBRARIES})

# external libs and dependencies
add_dependencies(${TARGET} cppgl-ext)
find_library(GLEW GLEW PATHS ${EXTERNAL_DIR}/lib NO_DEFAULT_PATH)
find_library(GLFW glfw PATHS ${EXTERNAL_DIR}/lib NO_DEFAULT_PATH)
find_library(ASSIMP assimp PATHS ${EXTERNAL_DIR}/lib NO_DEFAULT_PATH)
target_link_libraries(${TARGET} ${GLEW} ${GLFW} ${ASSIMP} stdc++fs cppgl)

# CUDA
find_package(CUDA REQUIRED)
target_link_libraries(${TARGET} ${CUDA_LIBRARIES})

# pytorch (optional)
if (WITH_TORCH)
    find_package(Torch REQUIRED)
    target_include_directories(${TARGET} PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${TORCH_LIBRARIES})
endif()

# openvdb (optional)
if (WITH_OPENVDB)
    add_dependencies(${TARGET} openvdb-ext)
    target_link_libraries(${TARGET} openvdb Half tbb)
ENDIF()

# dcmtk (optional)
if (WITH_DCMTK)
    add_dependencies(${TARGET} dcmtk-ext)
    target_link_libraries(${TARGET} dcmdata dcmimage dcmimgle oflog ofstd)
ENDIF()
