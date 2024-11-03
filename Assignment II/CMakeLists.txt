cmake_minimum_required(VERSION 3.20)
project(Assignment_I CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(exercise1 ex_1/vector_add.cu)
add_executable(exercise2 ex_2/matrix_mult.cu)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -Wno-deprecated-gpu-targets")

set_target_properties(exercise1 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(exercise2 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)