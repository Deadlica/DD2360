#ifndef GPU_MATERIAL_CUH
#define GPU_MATERIAL_CUH

// Project
#include <hittable.cuh>
#include <color.cuh>

/**
 * @class material
 * @brief Base class for materials that interact with light in ray tracing.
 *
 * The `material` class provides a common interface for different materials in the scene.
 * It allows for scattering behavior, where incoming light is scattered into a new ray
 * direction based on the material's properties.
 */
class material {
public:

    /**
     * @brief Scatters the incoming ray based on the material's properties.
     *
     * @param r_in The incoming ray.
     * @param rec The hit record containing intersection details.
     * @param attenuation The color that represents the intensity of scattered light.
     * @param scattered The resulting scattered ray.
     * @param local_rand_state A pointer to the CUDA random state.
     * @return `true` if scattering occurs, `false` otherwise.
     */
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                                    ray& scattered, curandState* local_rand_state) const;
};

/**
 * @class lambertian
 * @brief Lambertian (diffuse) material that scatters light uniformly in all directions.
 *
 * The `lambertian` material reflects light diffusely with a fixed albedo color.
 */
class lambertian : public material {
public:

    /**
     * @brief Constructs a Lambertian material with a given albedo color.
     *
     * @param albedo The color of the material (used for diffuse reflection).
     */
    __device__ lambertian(const color& albedo);

    /**
     * @brief Scatters an incoming ray according to Lambertian behavior.
     *
     * @param r_in The incoming ray.
     * @param rec The hit record containing intersection details.
     * @param attenuation The color representing the scattered light intensity.
     * @param scattered The resulting scattered ray.
     * @param local_rand_state A pointer to the CUDA random state.
     * @return `true`, since Lambertian materials always scatter light.
     */
    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                            ray& scattered, curandState* local_rand_state) const override;

private:
    color _albedo;
};

/**
 * @class metal
 * @brief Metal material that reflects light with a shiny surface.
 *
 * The `metal` material reflects light with a metallic sheen and some degree of fuzziness.
 */
class metal : public material {
public:

    /**
     * @brief Constructs a Metal material with a given albedo and fuzz factor.
     *
     * @param albedo The color of the metal surface.
     * @param fuzz The degree of fuzziness of the metal surface.
     */
    __device__ metal(const color& albedo, datatype fuzz);

    /**
     * @brief Scatters an incoming ray based on metallic behavior.
     *
     * @param r_in The incoming ray.
     * @param rec The hit record containing intersection details.
     * @param attenuation The color representing the reflected light intensity.
     * @param scattered The resulting scattered ray.
     * @param local_rand_state A pointer to the CUDA random state.
     * @return `true` if the ray is reflected.
     */
    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                            ray& scattered, curandState* local_rand_state) const override;

private:
    color _albedo;
    datatype _fuzz;
};

/**
 * @class dielectric
 * @brief Dielectric material that refracts light.
 *
 * The `dielectric` material simulates transparent objects like glass or water, which refract light.
 */
class dielectric : public material {
public:

    /**
     * @brief Constructs a Dielectric material with a given refraction index.
     *
     * @param refraction_index The refractive index of the material.
     */
    __device__ dielectric(datatype refraction_index);

    /**
     * @brief Scatters an incoming ray based on dielectric behavior.
     *
     * @param r_in The incoming ray.
     * @param rec The hit record containing intersection details.
     * @param attenuation The color representing the refracted/reflected light intensity.
     * @param scattered The resulting scattered ray.
     * @param local_rand_state A pointer to the CUDA random state.
     * @return `true` if scattering occurs.
     */
    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
                            ray& scattered, curandState* local_rand_state) const override;

private:
    datatype _refraction_index;

    /**
     * @brief Computes the reflectance based on the angle of incidence and refractive index (Schlick's approximation).
     *
     * @param cosine The cosine of the angle of incidence.
     * @param refraction_index The refractive index of the material.
     * @return The reflectance value, which determines how much light is reflected versus refracted.
     */
    __device__ static datatype reflectance(datatype cosine, datatype refraction_index);
};

#endif //GPU_MATERIAL_CUH
