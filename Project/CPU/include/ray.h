#ifndef CPU_RAY_H
#define CPU_RAY_H

// Project
#include <vec3.h>

/**
 * @class ray
 * @brief Represents a ray in 3D space.
 *
 * The `ray` class represents a 3D ray, defined by an origin point and a direction vector.
 * It provides methods to retrieve the ray's origin, direction, and a point along the ray
 * at a specific parameter value.
 */
class ray {
public:

    /**
     * @brief Default constructor that initializes a ray with zero origin and direction.
     */
    ray();

    /**
     * @brief Constructs a ray with a given origin and direction.
     *
     * @param origin The starting point of the ray.
     * @param direction The direction vector of the ray.
     */
    ray(const point3& origin, const vec3& direction);

    /**
     * @brief Returns the origin of the ray.
     *
     * @return A constant reference to the origin point of the ray.
     */
    const point3& origin() const;

    /**
     * @brief Returns the direction of the ray.
     *
     * @return A constant reference to the direction vector of the ray.
     */
    const vec3& direction() const;

    /**
     * @brief Computes the point along the ray at a given parameter t.
     *
     * @param t The parameter along the ray.
     * @return The point located at the distance `t` along the ray.
     */
    point3 at(datatype t) const;

private:
    point3 _origin;
    vec3 _direction;
};


#endif //CPU_RAY_H
