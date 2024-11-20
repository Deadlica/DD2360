#ifndef GPU_CAMERA_CUH
#define GPU_CAMERA_CUH

// Project
#include <hittable.cuh>
#include <color.cuh>

// std
#include <curand_kernel.h>

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

    __device__ void initialize();
    __device__ ray get_ray(datatype x, datatype y, curandState* local_rand_state) const;
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

    __device__ point3 defocus_disk_sample(curandState* local_rand_state) const;
};

#endif //GPU_CAMERA_CUH
