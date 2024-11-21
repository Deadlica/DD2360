#ifndef CPU_SPHERE_H
#define CPU_SPHERE_H

// Project
#include <hittable.h>
#include <vec3.h>

/**
 * @class sphere
 * @brief Represents a sphere object in 3D space.
 *
 * A sphere is defined by its center position, radius, and material.
 * It overrides the `hit` function from the `hittable` class to check if a ray intersects with the sphere.
 */
class sphere : public hittable {
public:
    /**
     * @brief Constructs a sphere with a specified center, radius, and material.
     *
     * @param center The center of the sphere.
     * @param radius The radius of the sphere.
     * @param mat The material of the sphere.
     */
    sphere(const point3& center, datatype radius, material_ptr mat);

    /**
     * @brief Tests if a ray intersects with the sphere.
     *
     * @param r The ray to test for intersection.
     * @param ray_t The interval representing the range of possible intersection distances along the ray.
     * @param rec The hit record that stores information about the intersection.
     * @return `true` if the ray intersects the sphere within the specified range, otherwise `false`.
     */
    bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

private:
    point3 _center;
    datatype _radius;
    material_ptr _mat;
};


#endif //CPU_SPHERE_H
