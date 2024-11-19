#ifndef GPU_CAMERA_CUH
#define GPU_CAMERA_CUH

// Project
#include <hittable.cuh>
#include <color.cuh>

// std
#include <curand_kernel.h>


class camera {
public:
    datatype aspect_ratio      = 1.0;
    int      image_width       = 100;
    int      samples_per_pixel = 10;

    __device__ void initialize();
    __device__ ray get_ray(datatype x, datatype y) const;
    __device__ color ray_color(const ray& r, hittable** world) const;

private:
    int    _image_height;
    point3 _center;
    point3 _pixel00_loc;
    vec3   _pixel_delta_x;
    vec3   _pixel_delta_y;
};

#endif //GPU_CAMERA_CUH
