cmake_minimum_required(VERSION 3.20)
project(Project CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

# CPU version
add_executable(Raytracer-CPU
               CPU/src/main.cpp
               CPU/src/vec3.cpp
               CPU/src/color.cpp
               CPU/src/color.cpp
               CPU/src/ray.cpp
               CPU/src/hittable.cpp
               CPU/src/sphere.cpp
               CPU/src/hittable_list.cpp
               CPU/src/interval.cpp
               CPU/src/camera.cpp
               CPU/src/material.cpp
               CPU/include/util.h
)

target_include_directories(Raytracer-CPU PRIVATE ${PROJECT_SOURCE_DIR}/CPU/include)

# GPU version
add_executable(Raytracer-GPU
               GPU/src/main.cu
               GPU/include/util.cuh
)

target_include_directories(Raytracer-GPU PRIVATE ${PROJECT_SOURCE_DIR}/GPU/include)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -Wno-deprecated-gpu-targets")

set_target_properties(Raytracer-GPU PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)