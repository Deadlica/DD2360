#ifndef GPU_SPHERE_CUH
#define GPU_SPHERE_CUH

// Project
#include <hittable.cuh>

class sphere : public hittable {
public:
    __device__ sphere();
    __device__ sphere(const point3& center, datatype radius, material* mat);

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

    material* mat;

private:
    point3 _center;
    datatype _radius;
};

#endif //GPU_SPHERE_CUH
