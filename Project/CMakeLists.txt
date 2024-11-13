cmake_minimum_required(VERSION 3.20)
project(Project CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(Raytracer-CPU
               CPU/main.cpp
               CPU/vec3.cpp
               CPU/color.cpp
               CPU/color.cpp
               CPU/ray.cpp
               CPU/hittable.cpp
               CPU/sphere.cpp
               CPU/hittable_list.cpp
               CPU/util.h
               CPU/interval.cpp
)

add_executable(Raytracer-GPU
               GPU/main.cu
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -Wno-deprecated-gpu-targets")

set_target_properties(Raytracer-GPU PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)