cmake_minimum_required(VERSION 3.20)
project(Project CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_52 -Wno-deprecated-gpu-targets")

# CPU version
add_executable(Raytracer-CPU
               CPU/src/main.cpp
               CPU/src/vec3.cpp
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
               GPU/src/vec3.cu
               GPU/src/color.cu
               GPU/src/ray.cu
               GPU/src/hittable.cu
               GPU/src/hittable_list.cu
               GPU/src/sphere.cu
               GPU/src/interval.cu
               GPU/src/camera.cu
               GPU/src/material.cu
               GPU/include/util.cuh
               GPU/include/precision.cuh
)

target_include_directories(Raytracer-GPU PRIVATE ${PROJECT_SOURCE_DIR}/GPU/include)
set_target_properties(Raytracer-GPU PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# GPU Unified Memory version
add_executable(Raytracer-GPU-Unified
               GPU/src/main.cu
               GPU/src/vec3.cu
               GPU/src/color.cu
               GPU/src/ray.cu
               GPU/src/hittable.cu
               GPU/src/hittable_list.cu
               GPU/src/sphere.cu
               GPU/src/interval.cu
               GPU/src/camera.cu
               GPU/src/material.cu
               GPU/include/util.cuh
               GPU/include/precision.cuh
)

target_include_directories(Raytracer-GPU-Unified PRIVATE ${PROJECT_SOURCE_DIR}/GPU/include)
target_compile_definitions(Raytracer-GPU-Unified PRIVATE UNIFIED)
set_target_properties(Raytracer-GPU-Unified PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# CPU version with SFML
add_executable(Raytracer-CPU-SFML
               CPU/src/main.cpp
               CPU/src/vec3.cpp
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

target_include_directories(Raytracer-CPU-SFML PRIVATE ${PROJECT_SOURCE_DIR}/CPU/include)
target_compile_definitions(Raytracer-CPU-SFML PRIVATE SFML)
find_package(SFML 2.5 REQUIRED graphics window system)
target_link_libraries(Raytracer-CPU-SFML PRIVATE sfml-graphics sfml-window sfml-system)


# GPU version with SFML
add_executable(Raytracer-GPU-SFML
               GPU/src/main.cu
               GPU/src/vec3.cu
               GPU/src/color.cu
               GPU/src/ray.cu
               GPU/src/hittable.cu
               GPU/src/hittable_list.cu
               GPU/src/sphere.cu
               GPU/src/interval.cu
               GPU/src/camera.cu
               GPU/src/material.cu
               GPU/include/util.cuh
               GPU/include/precision.cuh
)

target_include_directories(Raytracer-GPU-SFML PRIVATE ${PROJECT_SOURCE_DIR}/GPU/include)
target_compile_definitions(Raytracer-GPU-SFML PRIVATE SFML)
find_package(SFML 2.5 REQUIRED graphics window system)
target_link_libraries(Raytracer-GPU-SFML PRIVATE sfml-graphics sfml-window sfml-system)
set_target_properties(Raytracer-GPU-SFML PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# GPU Unified Memory version with SFML
add_executable(Raytracer-GPU-SFML-Unified
               GPU/src/main.cu
               GPU/src/vec3.cu
               GPU/src/color.cu
               GPU/src/ray.cu
               GPU/src/hittable.cu
               GPU/src/hittable_list.cu
               GPU/src/sphere.cu
               GPU/src/interval.cu
               GPU/src/camera.cu
               GPU/src/material.cu
               GPU/include/util.cuh
               GPU/include/precision.cuh
)

target_include_directories(Raytracer-GPU-SFML-Unified PRIVATE ${PROJECT_SOURCE_DIR}/GPU/include)
target_compile_definitions(Raytracer-GPU-SFML-Unified PRIVATE SFML UNIFIED)
find_package(SFML 2.5 REQUIRED graphics window system)
target_link_libraries(Raytracer-GPU-SFML-Unified PRIVATE sfml-graphics sfml-window sfml-system)
set_target_properties(Raytracer-GPU-SFML-Unified PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)