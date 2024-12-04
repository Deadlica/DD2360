cmake_minimum_required(VERSION 3.20)
project(Assignment_III CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(exercise1 ex_1/histogram.cu)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -Wno-deprecated-gpu-targets")

set_target_properties(exercise1 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)