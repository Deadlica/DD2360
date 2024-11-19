#ifndef GPU_HITTABLE_LIST_CUH
#define GPU_HITTABLE_LIST_CUH

// Project
#include <hittable.cuh>

class hittable_list : public hittable {
public:
    __device__ hittable_list();
    __device__ hittable_list(hittable** list, int n);

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

    hittable** list{};
    int list_size{};
};

#endif //GPU_HITTABLE_LIST_CUH
