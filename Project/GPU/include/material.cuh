#ifndef GPU_MATERIAL_CUH
#define GPU_MATERIAL_CUH

#include <hittable.cuh>
#include <color.cuh>

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                                    ray& scattered, curandState* local_rand_state) const;
};

class lambertian : public material {
public:
    __device__ lambertian(const color& albedo);

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                            ray& scattered, curandState* local_rand_state) const override;

private:
    color _albedo;
};

class metal : public material {
public:
    __device__ metal(const color& albedo, datatype fuzz);

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                            ray& scattered, curandState* local_rand_state) const override;

private:
    color _albedo;
    datatype _fuzz;
};

#endif //GPU_MATERIAL_CUH
