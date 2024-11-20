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

class dielectric : public material {
public:
    __device__ dielectric(datatype refraction_index);

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                            ray& scattered, curandState* local_rand_state) const override;

private:
    datatype _refraction_index;

    __device__ static datatype reflectance(datatype cosine, datatype refraction_index);
};

__device__ bool refract(const vec3& v, const vec3& n, datatype ni_over_nt, vec3& refracted);

#endif //GPU_MATERIAL_CUH
