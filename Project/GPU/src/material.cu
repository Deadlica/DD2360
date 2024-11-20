#include <material.cuh>

__device__ bool material::scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                                  ray& scattered, curandState* local_rand_state) const {
    return false;
}

__device__ lambertian::lambertian(const color& albedo): _albedo(albedo) {}

__device__ bool lambertian::scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                                    ray& scattered, curandState* local_rand_state) const {
    auto scatter_direction = rec.normal + random_in_unit_sphere(local_rand_state);

    if (scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, scatter_direction);
    attenuation = _albedo;
    return true;
}

__device__ metal::metal(const color& albedo, datatype fuzz)
: _albedo(albedo), _fuzz(fuzz < 1 ? fuzz : 1) {}

__device__ bool metal::scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                        ray& scattered, curandState* local_rand_state) const {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (_fuzz * random_in_unit_sphere(local_rand_state));
    scattered = ray(rec.p, reflected);
    attenuation = _albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}
