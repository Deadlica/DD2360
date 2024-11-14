#include "material.h"

bool material::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const {
    return false;
}

lambertian::lambertian(const color& albedo): _albedo(albedo) {}

bool lambertian::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const {
    auto scatter_direction = rec.normal + random_unit_vector();

    if (scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, scatter_direction);
    attenuation = _albedo;
    return true;
}

metal::metal(const color& albedo, datatype fuzz): _albedo(albedo), _fuzz(fuzz) {}

bool metal::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (_fuzz * random_unit_vector());
    scattered = ray(rec.p, reflected);
    attenuation = _albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}
