#ifndef GPU_VEC3_CUH
#define GPU_VEC3_CUH

// Project
#include <precision.cuh>

// std
#include <cmath>
#include <iostream>
#include <curand_kernel.h>

/**
 * @class vec3
 * @brief A 3D vector class for floating-point operations.
 *
 * Represents a 3D vector with x, y, z values and provides utility functions for vector operations.
 */
class vec3 {
public:
    datatype e[3];

    /**
     * @brief Default constructor.
     * Initializes the vector to (0, 0, 0).
     */
    __host__ __device__ vec3();

    /**
     * @brief Parameterized constructor.
     * @param e0 The x-value.
     * @param e1 The y-value.
     * @param e2 The z-value.
     */
    __host__ __device__ vec3(datatype e0, datatype e1, datatype e2);

    // Accessor methods
    /**
     * @brief Get the x-value of the vector.
     * @return The x-value.
     */
    __host__ __device__ datatype x() const;

    /**
     * @brief Get the y-value of the vector.
     * @return The y-value.
     */
    __host__ __device__ datatype y() const;

    /**
     * @brief Get the z-value of the vector.
     * @return The z-value.
     */
    __host__ __device__ datatype z() const;

    /**
     * @brief Get the r-value of the vector.
     * @return The r-value.
     */
    __host__ __device__ datatype r() const;

    /**
     * @brief Get the g-value of the vector.
     * @return The g-value.
     */
    __host__ __device__ datatype g() const;

    /**
     * @brief Get the b-value of the vector.
     * @return The b-value.
     */
    __host__ __device__ datatype b() const;

    __host__ __device__ const vec3& operator+() const;

    /**
     * @brief Negates the vector.
     * @return A new vector that is the negation of the current vector.
     */
    __host__ __device__ vec3 operator-() const;

    /**
     * @brief Accessor for vector values.
     * @param i The index of the value to access (0 for x, 1 for y, 2 for z).
     * @return The value at index i.
     */
    __host__ __device__ datatype operator[](int i) const;

    /**
     * @brief Mutator for vector values.
     * @param i The index of the value to modify.
     * @return A reference to the value at index i.
     */
    __host__ __device__ datatype& operator[](int i);

    /**
     * @brief Adds another vector to the current vector.
     * @param rhs The vector to add.
     * @return A reference to the resulting vector after addition.
     */
    __host__ __device__ vec3& operator+=(const vec3& rhs);

    /**
     * @brief Subtracts another vector to the current vector.
     * @param rhs The vector to subtract.
     * @return A reference to the resulting vector after subtraction.
     */
    __host__ __device__ vec3& operator-=(const vec3& rhs);


    /**
     * @brief Multiplies another vector to the current vector.
     * @param rhs The vector to with multiply.
     * @return A reference to the resulting vector after multiplication.
     */
    __host__ __device__ vec3& operator*=(const vec3& rhs);


    /**
     * @brief Divides another vector to the current vector.
     * @param rhs The vector to divide with.
     * @return A reference to the resulting vector after division.
     */
    __host__ __device__ vec3& operator/=(const vec3& rhs);

    /**
     * @brief Scales the current vector by a scalar value.
     * @param rhs The scalar value to multiply with.
     * @return A reference to the resulting vector after scaling.
     */
    __host__ __device__ vec3& operator*=(datatype rhs);

    /**
     * @brief Scales the current vector by division of a scalar value.
     * @param rhs The scalar value to divide by.
     * @return A reference to the resulting vector after division.
     */
    __host__ __device__ vec3& operator/=(datatype rhs);

    /**
     * @brief Calculates the length of the vector.
     * @return The length of the vector.
     */
    __host__ __device__ datatype length() const;

    /**
     * @brief Calculates the squared length of the vector.
     * @return The squared length of the vector.
     */
    __host__ __device__ datatype length_squared() const;

    /**
     * @brief Check if the vector is close to zero in all dimensions.
     * @return True if the vector is near zero, otherwise false.
     */
    __host__ __device__ bool     near_zero() const;

    /**
     * @brief Normalize the vector to make it a unit vector.
     * Modifies the current vector instance.
     */
    __host__ __device__ void     make_unit_vector();
};

using point3 = vec3; /**< Type alias for vec3 representing a point in 3D space. */

/**
 * @brief Overloaded stream input operator for vec3.
 * @param in The input stream.
 * @param v The vector to populate.
 * @return The modified input stream.
 */
inline std::istream& operator>>(std::istream& in, vec3& v) {
    return in >> v.e[0] >> v.e[1] >> v.e[2];
}

/**
 * @brief Overloaded stream output operator for vec3.
 * @param out The output stream.
 * @param v The vector to display.
 * @return The modified output stream.
 */
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << " " << v.e[1] << " " << v.e[2];
}

/**
 * @brief Overloaded addition operator for two vectors.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return A new vector that is the sum of lhs and rhs.
 */
__host__ __device__ inline vec3 operator+(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] + rhs.e[0], lhs.e[1] + rhs.e[1], lhs.e[2] + rhs.e[2]);
}

/**
 * @brief Overloaded subtraction operator for two vectors.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return A new vector that is the difference of lhs and rhs.
 */
__host__ __device__ inline vec3 operator-(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] - rhs.e[0], lhs.e[1] - rhs.e[1], lhs.e[2] - rhs.e[2]);
}

/**
 * @brief Overloaded multiplication operator for two vectors.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return A new vector that is the product of lhs and rhs.
 */
__host__ __device__ inline vec3 operator*(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] * rhs.e[0], lhs.e[1] * rhs.e[1], lhs.e[2] * rhs.e[2]);
}

/**
 * @brief Overloaded division operator for two vectors.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return A new vector that is the quotient of lhs and rhs.
 */
__host__ __device__ inline vec3 operator/(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] / rhs.e[0], lhs.e[1] / rhs.e[1], lhs.e[2] / rhs.e[2]);
}

/**
 * @brief Overloaded scalar multiplication operator.
 * @param lhs The scalar value.
 * @param rhs The vector.
 * @return A new vector scaled by lhs.
 */
__host__ __device__ inline vec3 operator*(datatype lhs, const vec3& rhs) {
    return vec3(lhs * rhs.e[0], lhs * rhs.e[1], lhs * rhs.e[2]);
}

/**
 * @brief Overloaded scalar division operator.
 * @param lhs The vector.
 * @param rhs The scalar value.
 * @return A new vector scaled by 1/rhs.
 */
__host__ __device__ inline vec3 operator/(const vec3& lhs, datatype rhs) {
    return (datatype(1) / rhs) * lhs;
}

/**
 * @brief Overloaded scalar multiplication operator.
 * @param lhs The vector.
 * @param rhs The scalar value.
 * @return A new vector scaled by rhs.
 */
__host__ __device__ inline vec3 operator*(const vec3& lhs, datatype rhs) {
    return rhs * lhs;
}

/**
 * @brief Compute the dot product of two vectors.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return The dot product of lhs and rhs.
 */
__host__ __device__ inline datatype dot(const vec3& lhs, const vec3& rhs) {
    return lhs.e[0] * rhs.e[0]
           + lhs.e[1] * rhs.e[1]
           + lhs.e[2] * rhs.e[2];
}

/**
 * @brief Compute the cross product of two vectors.
 * @param lhs The left-hand side vector.
 * @param rhs The right-hand side vector.
 * @return The cross product of lhs and rhs as a new vector.
 */
__host__ __device__ inline vec3 cross(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[1] * rhs.e[2] - lhs.e[2] * rhs.e[1],
                lhs.e[2] * rhs.e[0] - lhs.e[0] * rhs.e[2],
                lhs.e[0] * rhs.e[1] - lhs.e[1] * rhs.e[0]);
}

/**
 * @brief Compute the unit vector of a given vector.
 * @param v The input vector.
 * @return A new vector with the same direction as v but with a length of 1.
 */
__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

/**
 * @brief Generate a random point in a unit disk.
 * @param local_rand_state A pointer to the CUDA random state.
 * @return A random vector inside the unit disk.
 */
__device__ inline vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = datatype(2.0) * vec3(curand_uniform(local_rand_state),
                                 curand_uniform(local_rand_state),0)
                          - vec3(1, 1, 0);
    } while (dot(p, p) >= datatype(1.0));
    return p;
}

/**
 * @brief Reflect a vector around a normal.
 * @param v The incident vector.
 * @param n The normal vector.
 * @return The reflected vector.
 */
__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - datatype(2) * dot(v, n) * n;
}

/**
 * @brief Refract a vector based on the Snell's law.
 * @param uv The incident unit vector.
 * @param n The normal vector.
 * @param etai_over_etat The ratio of refractive indices.
 * @return The refracted vector.
 */
__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, datatype etai_over_etat) {
    datatype dot_uv = dot(-uv, n);
    datatype cos_theta = dot_uv < datatype(1.0) ? dot_uv : datatype(1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    datatype fabs_val = datatype(1.0) - r_out_perp.length_squared();
    fabs_val = fabs_val < 0 ? fabs_val * datatype(-1.0) : fabs_val;
    vec3 r_out_parallel = -std::sqrt(fabs_val) * n;
    return r_out_perp + r_out_parallel;
}

#endif //GPU_VEC3_CUH