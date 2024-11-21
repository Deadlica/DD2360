#ifndef CPU_CAMERA_H
#define CPU_CAMERA_H

// Project
#include <hittable.h>
#include <color.h>

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
    int      max_depth         = 10;

    datatype vfov     = 90;
    point3   lookfrom = point3(0, 0, 0);
    point3   lookat   = point3(0, 0, -1);
    vec3     vup      = vec3(0, 1, 0);

    datatype defocus_angle = 0;
    datatype focus_dist    = 10;

    /**
     * @brief Renders the scene using the camera's properties.
     *
     * This function initiates the ray tracing process, rendering the scene to an image.
     * It shoots rays from the camera and computes colors for each pixel using the world
     * (hittable objects) and other camera settings.
     *
     * @param world The scene containing the objects that will be rendered.
     */
    void render(const hittable& world);

private:
    int      _image_height;
    datatype _pixel_samples_scale;
    point3   _center;
    point3   _pixel00_loc;
    vec3     _pixel_delta_x;
    vec3     _pixel_delta_y;
    vec3     _x;
    vec3     _y;
    vec3     _z;
    vec3     _defocus_disk_x;
    vec3     _defocus_disk_y;

    /**
     * @brief Initializes the camera's internal state for ray tracing.
     *
     * This method sets up the camera's parameters.
     */
    void initialize();

    /**
     * @brief Generates a ray for a given pixel (i, j).
     *
     * @param i The x-coordinate of the pixel.
     * @param j The y-coordinate of the pixel.
     * @return The generated ray that passes through the given pixel.
     */
    ray get_ray(int i, int j) const;

    /**
     * @brief Samples a point in the unit square.
     *
     * @return A random sample point in the unit square.
     */
    vec3 sample_square() const;

    /**
     * @brief Samples a point on a disk with a given radius.
     *
     * @param radius The radius of the disk to sample from.
     * @return A random sample point on the disk.
     */
    vec3 sample_disk(datatype radius) const;

    /**
     * @brief Samples a point on a defocus disk.
     *
     * @return A random sample point on the defocus disk.
     */
    point3 defocus_disk_sample() const;

    /**
     * @brief Computes the color of a ray as it interacts with the world.
     *
     * @param r The ray to trace.
     * @param depth The current recursion depth for ray tracing.
     * @param world The scene containing the objects to be hit by rays.
     * @return The computed color for the ray after all interactions.
     */
    color ray_color(const ray& r, int depth, const hittable& world) const;
};


#endif //CPU_CAMERA_H
