cmake_minimum_required(VERSION 3.8)
project(cuda_cpp_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

include_directories(include)

set(SOURCES
    src/main.cu
    src/utils.cu
)

add_executable(cuda_cpp_project ${SOURCES})