// Project
#include <camera.cuh>
#include <material.cuh>

__device__ void camera::initialize() {
    _image_height = int(image_width / aspect_ratio);
    _image_height = _image_height < 1 ? 1 : _image_height;

    _center = lookfrom;

    // Camera
    datatype theta        = degrees_to_radians(vfov);
    datatype h            = std::tan(theta / 2);
    datatype viewport_height = datatype(2.0) * h * focus_dist;
    datatype viewport_width = viewport_height * (datatype(image_width) / _image_height);
    point3 camera_center = point3(0, 0, 0);

    // Camera position
    _z = unit_vector(lookfrom - lookat);
    _x = unit_vector(cross(vup, _z));
    _y = cross(_z, _x);

    // Viewports
    auto viewport_x = viewport_width * _x;
    auto viewport_y = viewport_height * -_y;

    // Pixel distance
    _pixel_delta_x = viewport_x / image_width;
    _pixel_delta_y = viewport_y / _image_height;

    // Start pos
    vec3 viewport_upper_left = _center - (focus_dist * _z) - viewport_x / 2 - viewport_y / 2;
    _pixel00_loc = viewport_upper_left + 0.5 * (_pixel_delta_x + _pixel_delta_y);

    // Camera defocus disk
    auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
    _defocus_disk_x = _x * defocus_radius;
    _defocus_disk_y = _y * defocus_radius;
}

__device__ ray camera::get_ray(datatype x, datatype y, curandState* local_rand_state) const {
    //return ray(_center, _pixel00_loc + x * _pixel_delta_x + y * _pixel_delta_y - _center);
    vec3 pixel_sample    = _pixel00_loc + x * _pixel_delta_x + y * _pixel_delta_y;
    point3 ray_origin    = defocus_angle <= 0 ? _center : defocus_disk_sample(local_rand_state);
    vec3   ray_direction = pixel_sample - ray_origin;
    return ray(ray_origin, ray_direction);
}

__device__ color camera::ray_color(const ray& r, hittable** world, curandState* local_rand_state) const {
    ray curr_ray = r;
    color curr_attenuation = color(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(curr_ray, interval(datatype(0.001), infinity), rec)) {
            ray scattered;
            color attenuation;
            if(rec.mat->scatter(curr_ray, rec, attenuation, scattered, local_rand_state)) {
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

__device__ point3 camera::defocus_disk_sample(curandState* local_rand_state) const {
    vec3 p = random_in_unit_disk(local_rand_state);
    return _center + (p[0] * _defocus_disk_x) + (p[1] * _defocus_disk_y);
}