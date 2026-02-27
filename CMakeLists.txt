cmake_minimum_required(VERSION 3.8)
project(cuda_cpp_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

set(SOURCES
    src/main.cpp
    src/sim.cu
)

add_executable(cuda_cpp_project ${SOURCES})

target_include_directories(cuda_cpp_project PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuda_cpp_project PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
    glfw
)