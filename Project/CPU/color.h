#ifndef PROJECT_COLOR_H
#define PROJECT_COLOR_H

#include "vec3.h"

#include <iostream>

using color = vec3;

void write_color(std::ostream& out, const color& pixel_color);

#endif //PROJECT_COLOR_H
