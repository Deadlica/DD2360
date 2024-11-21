#ifndef CPU_COLOR_H
#define CPU_COLOR_H

// Project
#include <vec3.h>

// std
#include <iostream>

using color = vec3; /**< Type alias for vec3 representing a RGB color. */

/**
 * @brief Converts a linear color value to gamma space.
 *
 * @param linear_component The linear color value to convert.
 * @return The gamma-corrected color value.
 */
inline datatype linear_to_gamma(datatype linear_component) {
    if (linear_component > 0) {
        return std::sqrt(linear_component);
    }

    return 0;
}

/**
 * @brief Writes a color value to an output stream.
 *
 * This function writes a color value to the provided output stream. The color is expected
 * to be in the format of an RGB value, and the function handles writing the
 * components to the output stream.
 *
 * @param out The output stream to write to (e.g., std::cout).
 * @param pixel_color The color value to write.
 */
void write_color(std::ostream& out, const color& pixel_color);

#endif //CPU_COLOR_H
