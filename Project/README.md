<h1 align="center">Raytracing on CPU vs GPU</h1>

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/Deadlica/DD2360/refs/heads/main/Project/images/README_logo.png" 
         alt="Image of raytraced spheres" 
         width="711" 
         height="400">
</div>

<br />

![pull](https://img.shields.io/github/issues-pr/deadlica/DD2360)
![issues](https://img.shields.io/github/issues/deadlica/DD2360)
![coverage](https://img.shields.io/codecov/c/github/deadlica/DD2360)
![language](https://img.shields.io/github/languages/top/deadlica/DD2360)

## Introduction
___
This repository provides two implementations of a ray tracing application: one using the CPU and another written in `CUDA` to leverage the GPU. The CPU version is based on the ray tracing application presented by Peter Shirley in his book [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html), while the GPU version is inspired by an [Nvidia blog post](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/).

The CPU implementation closely follows Peter Shirley's version with minor adjustments, whereas the GPU implementation differs slightly more from the blog post. The Nvidia blog introduces significant changes that diverge from Shirley's approach, while our GPU version aims to mimic the CPU implementation as closely as possible. As a result, our GPU version is more aligned with Shirley's original CPU-based design.

Both versions of the program render an image in the PPM format. Additionally, both versions can be built with SFML to directly display the image once rendering is complete.

## Dependencies
___
The following technologies have been tested and are required to run the project:
- g++ 11.4.0, or newer
- nvcc 11.15.119, or newer
- GNU Make 4.3, or newer
- CMake 3.22.1, or newer
- (Optional) SFML 2.5.1, or newer
- (Optional) Doxygen 1.11.0, or newer
- (Optional) Python 3.10.12, or newer
- Nvidia GPU

`Note 1: Older versions might work, but they have not been tested.`

## Documentation
___
Extensive documentation can be found [here](https://deadlica.github.io/DD2360/).

If you would like to generate new documentation pages make sure to install doxygen.
```bash
# For Debian
sudo apt-get install doxygen
```
```bash
# For Arch
sudo pacman -S doxygen
```
and thereafter simply run either of the following commands:
```bash
# Generates documentation with make
make docs

# Generates documentation without make
doxygen Doxygen/CPU/Doxyfile && doxygen Doxygen/GPU/Doxyfile && cp Doxygen/index.html docs
```
to generate the html pages.

<a id="build-run"></a>
## Building and Running
___
This project supports two build systems: `Make` and `CMake`.

`Note 2: All commands are to be run from the project root directory.`
`Note 3: You might need to edit the compute architecture in the "makefile" or "CMakeLists.txt". By default, it is set to sm_52.`
<a id="make"></a>
### Make
To build the project, use one of the following commands:
```bash
# Build all versions
make all

# Build the CPU version
make raytracer_cpu

# Build the GPU version
make raytracer_gpu

# Build the GPU unified memory version
make raytracer_gpu_unified

# Build the CPU version with SFML
make raytracer_cpu_sfml

# Build the GPU version with SFML
make raytracer_gpu_sfml

# Build the GPU unified memory version with SFML
make raytracer_gpu_sfml_unified
```

When running the program through `make` commands, the CPU and GPU versions will output the rendered images to `cpu_image.ppm` and `gpu_image.ppm`, respectively. Use the following commands to run them:
```bash
# Run the CPU version
make run_cpu

# Run the GPU version
make run_gpu

# Run the GPU unified memory version
make run_gpu_unified

# Run the CPU version with SFML
make run_cpu_sfml

# Run the GPU version with SFML
make run_gpu_sfml

# Run the GPU unified memory version with SFML
make run_gpu_sfml_unified
```

If you prefer to run the programs manually, you can either let the CPU version output to `cpu_image.ppm` and the GPU version to `gpu_image.ppm`, or redirect the output to a file of your choice.
```bash
# raytracer_cpu can be replaced with raytracer_cpu_sfml
# Run the CPU version and output to cpu_image.ppm
build/raytracer_cpu

# Run the CPU version and output to a custom file (e.g, image.ppm)
build/raytracer_cpu > image.ppm

# raytracer_gpu can be replaced with raytracer_gpu_sfml
# Run the GPU version and output to gpu_image.ppm
build/raytracer_gpu

# Run the GPU version and output to a custom file (e.g, image.ppm)
build/raytracer_gpu > image.ppm

# raytracer_gpu_unified can be replaced with raytracer_gpu_sfml_unified
# Run the GPU unified memory version and output to gpu_image.ppm
build/raytracer_gpu_unified

# Run the GPU unified memory version and output to a custom file (e.g, image.ppm)
build/raytracer_gpu_unified > image.ppm
```

To clean up the build, use the following command:
```bash
# Remove executables and the build directory
make clean
```

<a id="cmake"></a>
### CMake
To build the project, use the following commands:
```bash
# Build "build" directory
cmake -B build
```

To build specific versions, use any of the following commands:
```bash
# Build all versions
cmake --build build

# Build the CPU version
cmake --build build --target Raytracer-CPU

# Build the GPU version
cmake --build build --target Raytracer-GPU

# Build the GPU unified memory version
cmake --build build --target Raytracer-GPU-Unified

# Build the CPU version with SFML
cmake --build build --target Raytracer-CPU-SFML

# Build the GPU version SFML
cmake --build build --target Raytracer-GPU-SFML

# Build the GPU unified memory version with SFML
cmake --build build --target Raytracer-GPU-SFML-Unified
```

To run either version, use any of the following commands:
```bash
# Raytracer-CPU can be replaced with Raytracer-CPU-SFML
# Run the CPU version and output to cpu_image.ppm
build/Raytracer-CPU

# Run the CPU version and output to a custom file (e.g, image.ppm)
build/Raytracer-CPU > image.ppm

# Raytracer-GPU can be replaced with Raytracer-GPU-SFML
# Run the GPU version and output to gpu_image.ppm
build/Raytracer-GPU

# Run the GPU version and output to a custom file (e.g, image.ppm)
build/Raytracer-GPU > image.ppm

# Raytracer-GPU-Unified can be replaced with Raytracer-GPU-SFML-Unified
# Run the GPU unified memory version and output to gpu_image.ppm
build/Raytracer-GPU-Unified

# Run the GPU unified memory version and output to a custom file (e.g, image.ppm)
build/Raytracer-GPU-Unified > image.ppm

```

#### Optimized builds
You can choose between `Debug` or `Release` configurations when building the project.

##### On Windows:
```bash
# Build "build" directory
cmake -B build

# Build all versions as Debug
cmake --build build --config Debug

# Build the CPU version as Debug
cmake --build build --config Debug --target Raytracer-CPU

# Build the GPU version as Debug
cmake --build build --config Debug --target Raytracer-GPU

# Build the GPU unified memory version as Debug
cmake --build build --config Debug --target Raytracer-GPU-Unified

# Build the CPU version with SFML as Debug
cmake --build build --config Debug --target Raytracer-CPU-SFML

# Build the GPU version with SFML as Debug
cmake --build build --config Debug --target Raytracer-GPU-SFML

# Build the GPU unified memory version with SFML as Debug
cmake --build build --config Debug --target Raytracer-GPU-SFML-Unified

# Build all versions as Release
cmake --build build --config Release

# Build the CPU version as Release
cmake --build build --config Release --target Raytracer-CPU

# Build the GPU version as Release
cmake --build build --config Release --target Raytracer-GPU

# Build the GPU unified memory version as Release
cmake --build build --config Release --target Raytracer-GPU-Unified

# Build the CPU version with SFML as Release
cmake --build build --config Release --target Raytracer-CPU-SFML

# Build the GPU version with SFML as Release
cmake --build build --config Release --target Raytracer-GPU-SFML

# Build the GPU unified memory version with SFML as Release
cmake --build build --config Release --target Raytracer-GPU-SFML-Unified
```

##### On Linux / macOS:
```bash
# Build all versions as Debug
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug

# Build the CPU version as Debug
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --target Raytracer-CPU

# Build the GPU version as Debug
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --target Raytracer-GPU

# Build the CPU version with SFML as Debug
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --target Raytracer-CPU-SFML

# Build the GPU version with SFML as Debug
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --target Raytracer-GPU-SFML

# Build the GPU unified memory version as Debug
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --target Raytracer-GPU-Unified

# Build the GPU unified memory version with SFML as Debug
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --target Raytracer-GPU-SFML-Unified

# Build all versions as Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release

# Build the CPU version as Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --target Raytracer-CPU

# Build the GPU version as Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --target Raytracer-GPU

# Build the CPU version with SFML as Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --target Raytracer-CPU-SFML

# Build the GPU version with SMFL as Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --target Raytracer-GPU-SFML

# Build the GPU unified memory version as Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --target Raytracer-GPU-Unified

# Build the GPU unified memory version with SFML as Release
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --target Raytracer-GPU-SFML-Unified
```

##### Note about executables
The executables are located in the `build/Debug` or `build/Release` directories, depending on the selected configuration. You can run them as before:
```bash
# Example for running the CPU version in Debug mode
build/Debug/Raytracer-CPU
```

## Program parameters
You can modify the parameters for the rendered image, such as resolution or camera position, by editing their values in the source code.

### CPU parameters
In `CPU/src/main.cpp`, locate the following code section:
```cpp
...
cam.aspect_ratio      = 16.0 / 9.0;
cam.image_width       = width;
cam.samples_per_pixel = 10;
cam.max_depth         = 50;
  
cam.vfov              = 20;
cam.lookfrom          = point3(13, 2, 3);
cam.lookat            = point3(0, 0, 0);
cam.vup               = vec3(0, 1, 0);
  
cam.defocus_angle     = 0.6;
cam.focus_dist        = 10.0;
...
```

You can adjust these values as needed. After making changes, rebuild the CPU version executable using either [Make](#make) or [CMake](#cmake)

### GPU parameters
Similarly, in `GPU/src/main.cu`, find this section:
```cpp
...
rec.aspect_ratio      = datatype(16.0) / datatype(9.0);
rec.image_width       = width;
rec.samples_per_pixel = 10;

rec.vfov              = 20;
rec.lookfrom          = point3(13, 2, 3);
rec.lookat            = point3( 0, 0, 0);
rec.vup               = vec3  ( 0, 1, 0);
  
rec.defocus_angle     = 0.6;
rec.focus_dist        = 10.0;
...
```

After modifying these values, rebuild the GPU version executable using either [Make](#make) or [CMake](#cmake).

Once rebuilt, run the program as described in the [Building and Running](#build-run) section.

## Benchmarking Plots
A Python script is provided to efficiently generate plots while benchmarking the different ray tracing implementations. To begin, set up a Python virtual environment:
```bash
# Create the venv
python3 -m venv venv

# Activate it (windows)
venv\Scripts\activate

# Activate it (Linux/MacOS)
source venv/bin/activate
```

Next, you can use Make to run a predefined test configuration with the following command:

```bash
make benchmark
```

To run the script manually, use the following command to view the available CLI arguments:

```bash
python3 scripts/benchmark.py --help
```

### Example Configuration

```bash
python3 scripts/benchmark.py -cgm -c -r 10 -w 500 1000 1500 2000
```

This command compares the non-unified and unified memory versions, running 10 samples per render with image widths of 500, 1000, 1500, and 2000. Additionally, it cleans up the build files upon completion.