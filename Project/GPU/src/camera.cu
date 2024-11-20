// Project
#include <camera.cuh>
#include <material.cuh>

__device__ void camera::initialize() {
    _image_height = int(image_width / aspect_ratio);
    _image_height = _image_height < 1 ? 1 : _image_height;

    _center = point3(0, 0, 0);

    // Camera
    datatype focal_length = 1.0;
    datatype viewport_height = 2.0;
    datatype viewport_width = viewport_height * (datatype(image_width) / _image_height);
    point3 camera_center = point3(0, 0, 0);

    // Viewports
    auto viewport_x = vec3(viewport_width, 0, 0);
    auto viewport_y = vec3(0, -viewport_height, 0);

    // Pixel distance
    _pixel_delta_x = viewport_x / image_width;
    _pixel_delta_y = viewport_y / _image_height;

    // Start pos
    auto viewport_upper_left = camera_center
                               - vec3(0, 0, focal_length) - viewport_x / 2 - viewport_y / 2;
    _pixel00_loc = viewport_upper_left + 0.5 * (_pixel_delta_x + _pixel_delta_y);
}

__device__ ray camera::get_ray(datatype x, datatype y, curandState* local_rand_state) const {
    return ray(_center, _pixel00_loc + x * _pixel_delta_x + y * _pixel_delta_y);
}

__device__ color camera::ray_color(const ray& r, hittable** world, curandState* local_rand_state) const {
    ray curr_ray = r;
    color curr_attenuation = color(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(curr_ray, interval(datatype(0.001), infinity), rec)) {
            ray scattered;
            color attenuation;
            if(rec.mat->scatter(r, rec, attenuation, scattered, local_rand_state)) {
                curr_attenuation *= attenuation;
                curr_ray = scattered;
            }
            else {
                return color(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(curr_ray.direction());
            datatype a = datatype(0.5) * (unit_direction.y() + datatype(1.0));
            vec3 c = (datatype(1.0) - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
            return curr_attenuation * c;
        }
    }

    return color(0.0, 0.0, 0.0);
}