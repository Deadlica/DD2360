#ifndef GPU_HITTABLE_LIST_CUH
#define GPU_HITTABLE_LIST_CUH

// Project
#include <hittable.cuh>

/**
 * @class hittable_list
 * @brief A collection of `hittable` objects that can be intersected by rays.
 */
class hittable_list : public hittable {
public:

    /**
     * @brief Default constructor for an empty hittable list.
     */
    __device__ hittable_list();

    /**
     * @brief Constructs a hittable_list from an array of hittable pointers and the number of elements.
     * @param list A pointer to an array of `hittable` pointers representing the list of objects to be included in the hittable list.
     * @param n The number of elements in the list.
     */
    __device__ hittable_list(hittable** list, int n);

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
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

    hittable** list;
    int list_size;
};

#endif //GPU_HITTABLE_LIST_CUH
