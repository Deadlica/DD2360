#ifndef GPU_COLOR_H
#define GPU_COLOR_H

// Project
#include <vec3.cuh>

// std
#include <iostream>

using color = vec3;

__host__ inline datatype linear_to_gamma(datatype linear_component) {
    if (linear_component > datatype(0.0)) {
        return std::sqrt(linear_component);
    }

    return datatype(0.0);
}

__host__ void write_color(std::ostream& out, const color& pixel_color);

#endif //GPU_COLOR_H
