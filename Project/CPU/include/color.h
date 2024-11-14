#ifndef CPU_COLOR_H
#define CPU_COLOR_H

// Project
#include <vec3.h>

// std
#include <iostream>

using color = vec3;

inline datatype linear_to_gamma(datatype linear_component) {
    if (linear_component > 0) {
        return std::sqrt(linear_component);
    }

    return 0;
}

void write_color(std::ostream& out, const color& pixel_color);

#endif //CPU_COLOR_H
