# Compiler
CXX = g++
CXXFLAGS = -O3 -std=c++11

# Directories
CPU_SRCDIR = CPU/src
CPU_INCDIR = CPU/include
GPU_SRCDIR = GPU/src
GPU_INCDIR = GPU/include
BUILDDIR = build
TARGET_CPU = raytracer_cpu
TARGET_GPU = raytracer_gpu

# Source files
CPU_SOURCES = $(wildcard $(CPU_SRCDIR)/*.cpp)
CPU_OBJECTS = $(patsubst $(CPU_SRCDIR)/%.cpp, $(BUILDDIR)/cpu_%.o, $(CPU_SOURCES))

GPU_SOURCES = $(wildcard $(GPU_SRCDIR)/*.cu)
GPU_OBJECTS = $(patsubst $(GPU_SRCDIR)/%.cu, $(BUILDDIR)/gpu_%.o, $(GPU_SOURCES))

# Default target
all: $(TARGET_CPU) $(TARGET_GPU)

# CPU Build target
$(TARGET_CPU): $(CPU_OBJECTS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# GPU Build target
$(TARGET_GPU): $(GPU_OBJECTS) | $(BUILDDIR)
	nvcc -O3 -arch=sm_50 -o $@ $^

# Build CPU objects
$(BUILDDIR)/cpu_%.o: $(CPU_SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(CPU_INCDIR) -c $< -o $@

# Build GPU objects
$(BUILDDIR)/gpu_%.o: $(GPU_SRCDIR)/%.cu | $(BUILDDIR)
	nvcc -O3 -arch=sm_50 -I$(GPU_INCDIR) -c $< -o $@

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Run CPU version
run_cpu: $(TARGET_CPU)
	./$(TARGET_CPU) > image_cpu.ppm

# Run GPU version
run_gpu: $(TARGET_GPU)
	./$(TARGET_GPU) > image_gpu.ppm

# Clean build
clean:
	rm -rf $(BUILDDIR) $(TARGET_CPU) $(TARGET_GPU)

.PHONY: all run_cpu run_gpu clean