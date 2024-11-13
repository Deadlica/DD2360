#ifndef PROJECT_UTIL_H
#define PROJECT_UTIL_H

#include <cmath>
#include <limits>

// Constants

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

#endif //PROJECT_UTIL_H
