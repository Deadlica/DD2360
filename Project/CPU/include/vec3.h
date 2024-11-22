#ifndef CPU_VEC3_H
#define CPU_VEC3_H

// Project
#include <util.h>

// std
#include <cmath>
#include <iostream>

/**
 * @class vec3
 * @brief A class that represents a 3D vector with common operations.
 *
 * This class provides basic vector operations like addition, subtraction,
 * dot product, cross product, and more for 3D vectors.
 */
class vec3 {
public:
    datatype e[3];

    /**
     * @brief Default constructor, initializes the vector to (0, 0, 0).
     */
    vec3();

    /**
     * @brief Constructs a vec3 with specified values.
     * @param e0 The x value of the vector.
     * @param e1 The y value of the vector.
     * @param e2 The z value of the vector.
     */
    vec3(datatype e0, datatype e1, datatype e2);

    /**
     * @brief Returns the x value of the vector.
     * @return The x value of the vector.
     */
    datatype x() const;

    /**
     * @brief Returns the y value of the vector.
     * @return The y value of the vector.
     */
    datatype y() const;

    /**
     * @brief Returns the z value of the vector.
     * @return The z value of the vector.
     */
    datatype z() const;

    /**
     * @brief Negates the vector.
     * @return A new vector that is the negation of the current vector.
     */
    vec3 operator-() const;

    /**
     * @brief Accessor for vector values.
     * @param i The index of the value to access (0 for x, 1 for y, 2 for z).
     * @return The value at index i.
     */
    datatype operator[](int i) const;

    /**
     * @brief Mutator for vector values.
     * @param i The index of the value to modify.
     * @return A reference to the value at index i.
     */
    datatype& operator[](int i);

    /**
     * @brief Adds another vector to the current vector.
     * @param rhs The vector to add.
     * @return A reference to the resulting vector after addition.
     */
    vec3& operator+=(const vec3& rhs);

    /**
     * @brief Scales the current vector by a scalar value.
     * @param rhs The scalar value to multiply with.
     * @return A reference to the resulting vector after scaling.
     */
    vec3& operator*=(datatype rhs);

    /**
     * @brief Scales the current vector by division of a scalar value.
     * @param rhs The scalar value to divide by.
     * @return A reference to the resulting vector after division.
     */
    vec3& operator/=(datatype rhs);

    /**
     * @brief Calculates the length of the vector.
     * @return The length of the vector.
     */
    datatype length() const;

    /**
     * @brief Calculates the squared length of the vector.
     * @return The squared length of the vector.
     */
    datatype length_squared() const;

    /**
     * @brief Checks if the vector is near zero (within a small epsilon).
     * @return True if the vector's length is near zero, false otherwise.
     */
    bool near_zero() const;

    /**
     * @brief Generates a random 3D vector with values in the range [0, 1).
     * @return A random 3D vector.
     */
    static vec3 random();

    /**
     * @brief Generates a random 3D vector with values in the specified range.
     * @param min The minimum value for each value.
     * @param max The maximum value for each value.
     * @return A random 3D vector with values in the specified range.
     */
    static vec3 random(datatype min, datatype max);
};

using point3 = vec3; /**< Type alias for vec3 representing a point in 3D space. */

/**
 * @brief Outputs the values of the vector to a stream.
 * @param out The output stream.
 * @param v The vector to output.
 * @return The output stream with the vector values.
 */
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << " " << v.e[1] << " " << v.e[2];
}

/**
 * @brief Adds two vectors value-wise.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return A new vector that is the result of the addition.
 */
inline vec3 operator+(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] + rhs.e[0], lhs.e[1] + rhs.e[1], lhs.e[2] + rhs.e[2]);
}

/**
 * @brief Subtracts the values of one vector from another.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return A new vector that is the result of the subtraction.
 */
inline vec3 operator-(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] - rhs.e[0], lhs.e[1] - rhs.e[1], lhs.e[2] - rhs.e[2]);
}

/**
 * @brief Multiplies two vectors value-wise.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return A new vector that is the result of the multiplication.
 */
inline vec3 operator*(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] * rhs.e[0], lhs.e[1] * rhs.e[1], lhs.e[2] * rhs.e[2]);
}

/**
 * @brief Multiplies a scalar with a vector.
 * @param lhs The scalar value.
 * @param rhs The vector.
 * @return A new vector that is the result of the scalar multiplication.
 */
inline vec3 operator*(datatype lhs, const vec3& rhs) {
    return vec3(lhs * rhs.e[0], lhs * rhs.e[1], lhs * rhs.e[2]);
}

/**
 * @brief Multiplies a vector with a scalar.
 * @param lhs The vector.
 * @param rhs The scalar value.
 * @return A new vector that is the result of the scalar multiplication.
 */
inline vec3 operator*(const vec3& lhs, datatype rhs) {
    return rhs * lhs;
}

/**
 * @brief Divides a vector by a scalar value.
 * @param lhs The vector.
 * @param rhs The scalar value.
 * @return A new vector that is the result of the division.
 */
inline vec3 operator/(const vec3& lhs, datatype rhs) {
    return (1 / rhs) * lhs;
}

/**
 * @brief Computes the dot product of two vectors.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return The dot product of the two vectors.
 */
inline datatype dot(const vec3& lhs, const vec3& rhs) {
    return lhs.e[0] * rhs.e[0]
           + lhs.e[1] * rhs.e[1]
           + lhs.e[2] * rhs.e[2];
}

/**
 * @brief Computes the cross product of two vectors.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return A new vector that is the result of the cross product.
 */
inline vec3 cross(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[1] * rhs.e[2] - lhs.e[2] * rhs.e[1],
                lhs.e[2] * rhs.e[0] - lhs.e[0] * rhs.e[2],
                lhs.e[0] * rhs.e[1] - lhs.e[1] * rhs.e[0]);
}

/**
 * @brief Returns a unit vector in the same direction as the input vector.
 * @param v The vector to normalize.
 * @return A new unit vector.
 */
inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

/**
 * @brief Generates a random point in the unit disk (2D).
 * @return A random point in the unit disk.
 */
inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_real_number(-1, 1), random_real_number(-1, 1), 0);
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

/**
 * @brief Generates a random unit vector (a vector on the unit sphere).
 * @return A random unit vector.
 */
inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3::random(-1, 1);
        auto lensq = p.length_squared();
        if (eps < lensq && lensq <= 1) {
            return p / std::sqrt(lensq);
        }
    }
}

/**
 * @brief Generates a random vector on the hemisphere with respect to a normal vector.
 * @param normal The normal vector for the hemisphere.
 * @return A random vector on the hemisphere.
 */
inline vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    }
    return -on_unit_sphere;
}

/**
 * Reflects a vector across a normal.
 *
 * @param v The input vector to be reflected.
 * @param n The normal vector of the surface.
 * @return The reflected vector.
 */
inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

/**
 * Refracts a vector through a surface with a given normal, using Snell' law.
 *
 * @param uv The incident vector (incoming ray).
 * @param n The normal vector of the surface.
 * @param etai_over_etat The ratio of the refractive indices (incident to transmitted).
 * @return The refracted vector, or the result of total internal reflection if applicable.
 */
inline vec3 refract(const vec3& uv, const vec3& n, datatype etai_over_etat) {
    auto cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif //CPU_VEC3_H