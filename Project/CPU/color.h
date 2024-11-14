#ifndef PROJECT_COLOR_H
#define PROJECT_COLOR_H

#include "vec3.h"

#include <iostream>

using color = vec3;

inline datatype linear_to_gamma(datatype linear_component) {
    if (linear_component > 0) {
        return std::sqrt(linear_component);
    }

    return 0;
}

void write_color(std::ostream& out, const color& pixel_color);

#endif //PROJECT_COLOR_H
