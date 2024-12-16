cmake_minimum_required(VERSION 3.20)
project(Bonus_Assignment CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_ARCHITECTURES 50)

# Exercise 1
add_executable(exercise1 ex_1/main.cu)
set_target_properties(exercise1 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# Exercise 2
add_executable(exercise2 ex_2/main.cu)
set_target_properties(exercise2 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)