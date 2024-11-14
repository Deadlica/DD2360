# Compiler
CXX = g++
CXXFLAGS = -O3 -std=c++11

# Directories
SRCDIR = CPU
BUILDDIR = build
TARGET = raytracer

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SOURCES))

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build objects
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Run
run: all
	./raytracer > image.ppm

# Clean build
clean:
	rm -rf $(BUILDDIR) $(TARGET)

.PHONY: all run clean