cmake_minimum_required(VERSION 3.20)
#===================== Ugly Fix ======================
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
#=====================================================
project(Assignment_IV CUDA CXX)

set(ARCH "sm_52")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 50)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=${ARCH} -Wno-deprecated-gpu-targets")

find_package(CUDAToolkit REQUIRED)

# Exercise 1
add_executable(exercise1 ex_1/warps.cu)
set_target_properties(exercise1 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# Exercise 2
add_executable(exercise2 ex_2/stream_add.cu)
set_target_properties(exercise2 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# Exercise 3
add_executable(exercise3 ex_3/heat_equation.cu)
set_target_properties(exercise3 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(exercise3 cublas cusparse)