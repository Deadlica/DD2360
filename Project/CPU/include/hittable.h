#ifndef CPU_HITTABLE_H
#define CPU_HITTABLE_H

// Project
#include <ray.h>
#include <interval.h>

// std
#include <memory>

class material;

using material_ptr = std::shared_ptr<material>;

/**
 * @brief Stores information about a ray intersection.
 *
 * This structure contains details about the intersection point, the surface normal at the intersection,
 * the material of the intersected object, and other relevant data. It is used by the `hittable` objects
 * to provide information about the hit when a ray intersects the object.
 */
class hit_record {
public:
    point3 p;
    vec3 normal;
    material_ptr mat;
    datatype t;
    bool front_face;

    /**
     * @brief Sets the face normal based on the direction of the incoming ray.
     *
     * @param r The incoming ray that intersects the object.
     * @param outward_normal The outward normal vector of the object at the intersection.
     */
    void set_face_normal(const ray& r, const vec3& outward_normal);
};

/**
 * @class hittable
 * @brief Abstract base class for objects that can be intersected by rays.
 *
 * This class defines the interface for all objects that can be hit by a ray.
 */
class hittable {
public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~hittable() = default;

    /**
     * @brief Tests if a ray intersects the object.
     *
     * @param r The ray to test for intersection.
     * @param ray_t The interval within which to check for intersections.
     * @param rec A `hit_record` object that will be filled with intersection data if a hit occurs.
     * @return `true` if the ray hits the object, `false` otherwise.
     */
    virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};


#endif //CPU_HITTABLE_H
