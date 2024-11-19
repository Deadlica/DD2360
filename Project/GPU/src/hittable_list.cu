// Project
#include <hittable_list.cuh>

__device__ hittable_list::hittable_list(): list(nullptr), list_size(0) {}

__device__ hittable_list::hittable_list(hittable** list, int n): list(list), list_size(n) {}

__device__ bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    datatype closest_so_far = ray_t.max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}
