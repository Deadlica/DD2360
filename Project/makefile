# Compiler
CXX = g++
CXXFLAGS = -O3 -std=c++11

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -arch=sm_50 -Wno-deprecated-gpu-targets --compiler-options="-fPIC" -Xcompiler -rdynamic

# Directories
CPU_SRCDIR   	= CPU/src
CPU_INCDIR   	= CPU/include
GPU_SRCDIR   	= GPU/src
GPU_INCDIR   	= GPU/include
BUILDDIR     	= build
DOCSDIR      	= docs
TARGET_CPU   	= raytracer_cpu
TARGET_GPU   	= raytracer_gpu
TARGET_CPU_SFML = raytracer_cpu_sfml
TARGET_GPU_SFML = raytracer_gpu_sfml
CPU_DOXYFILE 	= Doxygen/CPU/Doxyfile
GPU_DOXYFILE 	= Doxygen/GPU/Doxyfile
CUSTOM_INDEX 	= Doxygen/index.html

# SFML dependencies
SFML_LIBS = -lsfml-graphics -lsfml-window -lsfml-system

# Source files
CPU_SOURCES = $(wildcard $(CPU_SRCDIR)/*.cpp)
CPU_OBJECTS = $(patsubst $(CPU_SRCDIR)/%.cpp, $(BUILDDIR)/cpu_%.o, $(CPU_SOURCES))

GPU_SOURCES = $(wildcard $(GPU_SRCDIR)/*.cu)
GPU_OBJECTS = $(patsubst $(GPU_SRCDIR)/%.cu, $(BUILDDIR)/gpu_%.o, $(GPU_SOURCES))

# SFML Object files
CPU_OBJECTS_SFML = $(patsubst $(CPU_SRCDIR)/%.cpp, $(BUILDDIR)/cpu_%_sfml.o, $(CPU_SOURCES))
GPU_OBJECTS_SFML = $(patsubst $(GPU_SRCDIR)/%.cu, $(BUILDDIR)/gpu_%_sfml.o, $(GPU_SOURCES))

# Default target
all: $(TARGET_CPU) $(TARGET_GPU) $(TARGET_CPU_SFML) $(TARGET_GPU_SFML)

# CPU Build target
$(TARGET_CPU): $(CPU_OBJECTS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $(BUILDDIR)/$@ $^

# GPU Build target
$(TARGET_GPU): $(GPU_OBJECTS) | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) --disable-warnings -dlink $(GPU_OBJECTS) -o $(BUILDDIR)/gpu_dlink.o
	$(NVCC) $(NVCCFLAGS) -o $(BUILDDIR)/$@ $(GPU_OBJECTS) $(BUILDDIR)/gpu_dlink.o

# CPU SFML Build target
$(TARGET_CPU_SFML): $(CPU_OBJECTS_SFML) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $(BUILDDIR)/$@ $^ $(SFML_LIBS)

# GPU SFML Build target
$(TARGET_GPU_SFML): $(GPU_OBJECTS_SFML) | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) --disable-warnings -dlink $(GPU_OBJECTS_SFML) -o $(BUILDDIR)/gpu_sfml_dlink.o
	$(NVCC) $(NVCCFLAGS) -o $(BUILDDIR)/$@ $(GPU_OBJECTS_SFML) $(BUILDDIR)/gpu_sfml_dlink.o $(SFML_LIBS)

# Build CPU objects
$(BUILDDIR)/cpu_%.o: $(CPU_SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(CPU_INCDIR) -c $< -o $@

# Build GPU objects
$(BUILDDIR)/gpu_%.o: $(GPU_SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -I$(GPU_INCDIR) -dc $< -o $@

# Build CPU SFML objects
$(BUILDDIR)/cpu_%_sfml.o: $(CPU_SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(CPU_INCDIR) -DSFML -c $< -o $@

# Build GPU SFML objects
$(BUILDDIR)/gpu_%_sfml.o: $(GPU_SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -I$(GPU_INCDIR) -DSFML -dc $< -o $@

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Run CPU version
run_cpu: $(TARGET_CPU)
	$(BUILDDIR)/$(TARGET_CPU) > image_cpu.ppm

# Run GPU version
run_gpu: $(TARGET_GPU)
	$(BUILDDIR)/$(TARGET_GPU) > image_gpu.ppm

# Run CPU SFML version
run_cpu_sfml: $(TARGET_CPU_SFML)
	$(BUILDDIR)/$(TARGET_CPU_SFML)

# Run GPU SFML version
run_gpu_sfml: $(TARGET_GPU_SFML)
	$(BUILDDIR)/$(TARGET_GPU_SFML)

# Clean build
clean:
	rm -rf $(BUILDDIR) $(TARGET_CPU) $(TARGET_GPU) $(TARGET_CPU_SFML) $(TARGET_GPU_SFML) $(DOCSDIR)

# Generate docs
docs:
	doxygen $(CPU_DOXYFILE)
	doxygen $(GPU_DOXYFILE)
	cp $(CUSTOM_INDEX) $(DOCSDIR)

.PHONY: all run_cpu run_gpu run_cpu_sfml run_gpu_sfml clean docs