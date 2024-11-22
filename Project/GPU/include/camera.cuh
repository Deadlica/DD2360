#ifndef GPU_CAMERA_CUH
#define GPU_CAMERA_CUH

// Project
#include <hittable.cuh>
#include <color.cuh>

// std
#include <curand_kernel.h>

/**
 * @brief A structure to hold the camera configuration for rendering.
 *
 * This struct contains the essential parameters for camera setup,
 * including aspect ratio, field of view, focus settings, and camera position.
 */
struct cam_record {
    datatype aspect_ratio      = 1.0;
    int      image_width       = 100;
    int      samples_per_pixel = 10;

    datatype vfov              = 90;
    point3   lookfrom          = point3(0, 0, 0);
    point3   lookat            = point3(0, 0, -1);
    vec3     vup               = vec3(0, 1, 0);

    datatype defocus_angle     = 0;
    datatype focus_dist        = 10;
};

/**
 * @class camera
 * @brief Models a camera for ray tracing.
 *
 * The `camera` class defines a virtual camera with properties like aspect ratio, field of view (vfov),
 * and focus distance. It is responsible for generating rays for each pixel in the rendered image.
 */
class camera {
public:
    datatype aspect_ratio      = 1.0;
    int      image_width       = 100;
    int      samples_per_pixel = 10;

    datatype vfov              = 90;
    point3   lookfrom          = point3(0, 0, 0);
    point3   lookat            = point3(0, 0, -1);
    vec3     vup               = vec3(0, 1, 0);

    datatype defocus_angle     = 0;
    datatype focus_dist        = 10;

    /**
     * @brief Initializes the camera's internal state for ray tracing.
     *
     * This method sets up the camera's parameters.
     */
    __device__ void initialize();

    /**
     * @brief Generates a ray for a given pixel (i, j).
     *
     * @param x The x-coordinate of the pixel.
     * @param y The y-coordinate of the pixel.
     * @param local_rand_state A pointer to the CUDA random state.
     * @return The generated ray that passes through the given pixel.
     */
    __device__ ray get_ray(datatype x, datatype y, curandState* local_rand_state) const;

    /**
     * @brief Computes the color of a ray as it interacts with the world.
     *
     * @param r The ray to trace.
     * @param world The scene containing the objects to be hit by rays.
     * @param local_rand_state A pointer to the CUDA random state.
     * @return The computed color for the ray after all interactions.
     */
    __device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state) const;

private:
    int    _image_height;
    point3 _center;
    point3 _pixel00_loc;
    vec3   _pixel_delta_x;
    vec3   _pixel_delta_y;
    vec3   _x;
    vec3   _y;
    vec3   _z;
    vec3   _defocus_disk_x;
    vec3   _defocus_disk_y;

    /**
     * @brief Samples a point on a defocus disk.
     *
     * @param local_rand_state A pointer to the CUDA random state.
     * @return A random sample point on the defocus disk.
     */
    __device__ point3 defocus_disk_sample(curandState* local_rand_state) const;
};

#endif //GPU_CAMERA_CUH
