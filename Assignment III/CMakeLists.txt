cmake_minimum_required(VERSION 3.20)
project(Assignment_III CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_ARCHITECTURES 50)

# Exercise 1
add_executable(exercise1 ex_1/histogram.cu)
set_target_properties(exercise1 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# Exercise 2
include_directories(${CMAKE_SOURCE_DIR}/ex_2/include)

file(GLOB_RECURSE EX2_SOURCES
     "ex_2/src/*.cu"
     "ex_2/src/*.cpp"
)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/ex_2/bin)

add_executable(exercise2 ${EX2_SOURCES})

set_target_properties(exercise2 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/ex_2/bin
)

target_compile_options(exercise2 PRIVATE
                       $<$<COMPILE_LANGUAGE:CXX>:-O3 -g -Wall>
                       $<$<COMPILE_LANGUAGE:CUDA>:-O3 -g -Xcompiler=-Wall>
)