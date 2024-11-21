#ifndef CPU_HITTABLE_LIST_H
#define CPU_HITTABLE_LIST_H

// Project
#include <hittable.h>

// std
#include <vector>

using hittable_ptr = std::shared_ptr<hittable>;

/**
 * @class hittable_list
 * @brief A collection of `hittable` objects that can be intersected by rays.
 */
class hittable_list : public hittable {
public:
    std::vector<hittable_ptr> objects;

    /**
     * @brief Default constructor for an empty hittable list.
     */
    hittable_list();
    /**
     * @brief Constructs a hittable list with a single object.
     *
     * @param object The `hittable` object to add to the list.
     */
    hittable_list(hittable_ptr object);

    /**
     * @brief Clears all objects from the hittable list.
     *
     * This method removes all objects from the list, effectively resetting it.
     */
    void clear();

    /**
     * @brief Adds an object to the hittable list.
     *
     * @param object The `hittable` object to add.
     */
    void add(hittable_ptr object);

    /**
     * @brief Checks if any object in the list is hit by the given ray.
     *
     * This method tests all objects in the list for intersection with the given ray. If a hit is detected,
     * it populates the `hit_record` with relevant intersection details.
     *
     * @param r The ray to test for intersection.
     * @param ray_t The interval within which to check for intersections.
     * @param rec A `hit_record` object to store the intersection data if a hit occurs.
     * @return `true` if any object in the list is hit by the ray, `false` otherwise.
     */
    bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
};


#endif //CPU_HITTABLE_LIST_H
