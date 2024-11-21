#ifndef CPU_UTIL_H
#define CPU_UTIL_H

// std
#include <cmath>
#include <limits>
#include <random>

#define datatype double

// Constants

/**
 * Constant representing infinity.
 * This is set to the maximum value representable by the datatype.
 */
constexpr datatype infinity = std::numeric_limits<datatype>::infinity();

/**
 * Constant representing the value of pi.
 */
constexpr datatype pi = 3.1415926535897932385;

/**
 * Small epsilon value used for numerical comparisons.
 * It is used for precision tolerance in floating-point operations.
 */
constexpr datatype eps = 1e-160;

static std::random_device rd;
static std::mt19937 gen;

// Utility Functions

/**
 * Converts an angle from degrees to radians.
 *
 * @param degrees The angle in degrees.
 * @return The angle in radians.
 */
inline datatype degrees_to_radians(datatype degrees) {
    return degrees * pi / 180.0;
}

/**
 * Generates a random floating-point number between 0.0 and 1.0.
 *
 * @return A random real number between 0.0 and 1.0.
 */
inline datatype random_real_number() {
    std::uniform_real_distribution<datatype> urd(0.0, 1.0);
    return urd(gen);
}

/**
 * Generates a random floating-point number between the specified range.
 *
 * @param min The lower bound of the random number range.
 * @param max The upper bound of the random number range.
 * @return A random real number between min and max.
 */
inline datatype random_real_number(datatype min, datatype max) {
    return min + (max - min) * random_real_number();
}

#endif //CPU_UTIL_H
