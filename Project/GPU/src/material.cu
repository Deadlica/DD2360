// Project
#include <material.cuh>

__device__ bool material::scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                                  ray& scattered, curandState* local_rand_state) const {
    return false;
}

__device__ lambertian::lambertian(const color& albedo): _albedo(albedo) {}

__device__ bool lambertian::scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                                    ray& scattered, curandState* local_rand_state) const {
    vec3 scatter_direction = rec.normal + random_in_unit_sphere(local_rand_state);

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

__device__ dielectric::dielectric(datatype refraction_index): _refraction_index(refraction_index) {}

__device__ bool dielectric::scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                                    ray& scattered, curandState* local_rand_state) const {
    attenuation = color(1.0, 1.0, 1.0);
    datatype ri = rec.front_face ? (datatype(1.0) / _refraction_index) : _refraction_index;

    vec3 unit_direction = unit_vector(r_in.direction());
    datatype dot_val = dot(-unit_direction, rec.normal);
    datatype cos_theta = dot_val < datatype(1.0) ? dot_val : datatype(1.0);
    datatype sin_theta = std::sqrt(datatype(1.0) - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > datatype(1.0);
    vec3 direction;

    if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(local_rand_state)) {
        direction = reflect(unit_direction, rec.normal);
    }
    else {
        direction = refract(unit_direction, rec.normal, ri);
    }

    scattered = ray(rec.p, direction);
    return true;
}

__device__ datatype dielectric::reflectance(datatype cosine, datatype refraction_index) {
    datatype r0 = (datatype(1.0) - refraction_index) / (datatype(1.0) + refraction_index);
    r0 = r0 * r0;
    return r0 + (datatype(1.0) - r0) * std::pow(datatype(1.0) - cosine, datatype(5.0));
}