#ifndef PROJECT_UTIL_H
#define PROJECT_UTIL_H

#include <cmath>
#include <limits>
#include <random>

#define datatype double

// Constants

constexpr datatype infinity = std::numeric_limits<datatype>::infinity();
constexpr datatype pi = 3.1415926535897932385;
constexpr datatype eps = 1e-160;

static std::random_device rd;
static std::mt19937 gen;

// Utility Functions

inline datatype degrees_to_radians(datatype degrees) {
    return degrees * pi / 180.0;
}

inline datatype random_real_number() {
    std::uniform_real_distribution<datatype> urd(0.0, 1.0);
    return urd(gen);
}

inline datatype random_real_number(datatype min, datatype max) {
    return min + (max - min) * random_real_number();
}

#endif //PROJECT_UTIL_H
