#ifndef GPU_HITTABLE_CUH
#define GPU_HITTABLE_CUH

// Project
#include <ray.cuh>
#include <interval.cuh>

struct hit_record {
    point3 p;
    vec3 normal;
    datatype t;
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal);
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};

#endif //GPU_HITTABLE_CUH
