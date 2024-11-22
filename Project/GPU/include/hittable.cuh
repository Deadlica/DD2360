#ifndef GPU_HITTABLE_CUH
#define GPU_HITTABLE_CUH

// Project
#include <ray.cuh>
#include <interval.cuh>

class material;

/**
 * @brief Stores information about a ray intersection.
 *
 * This structure contains details about the intersection point, the surface normal at the intersection,
 * the material of the intersected object, and other relevant data. It is used by the `hittable` objects
 * to provide information about the hit when a ray intersects the object.
 */
struct hit_record {
    point3 p;
    vec3 normal;
    datatype t;
    bool front_face;
    material* mat;

    /**
     * @brief Sets the face normal based on the direction of the incoming ray.
     *
     * @param r The incoming ray that intersects the object.
     * @param outward_normal The outward normal vector of the object at the intersection.
     */
    __device__ void set_face_normal(const ray& r, const vec3& outward_normal);
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
     * @brief Tests if a ray intersects the object.
     *
     * @param r The ray to test for intersection.
     * @param ray_t The interval within which to check for intersections.
     * @param rec A `hit_record` object that will be filled with intersection data if a hit occurs.
     * @return `true` if the ray hits the object, `false` otherwise.
     */
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};

#endif //GPU_HITTABLE_CUH
