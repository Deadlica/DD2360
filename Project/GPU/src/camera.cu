// Project
#include <camera.cuh>

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

__device__ ray camera::get_ray(datatype x, datatype y) const {
    return ray(_center, _pixel00_loc + x * _pixel_delta_x + y * _pixel_delta_y);
}

__device__ color camera::ray_color(const ray& r, hittable** world) const {
    hit_record rec;
    if((*world)->hit(r, interval(0, infinity), rec)) {
        return datatype(0.5) * vec3(rec.normal.x() + datatype(1.0),
                                    rec.normal.y() + datatype(1.0),
                                    rec.normal.z() + datatype(1.0));
    }
    vec3 unit_direction = unit_vector(r.direction());
    datatype a = datatype(0.5) * (unit_direction.y() + datatype(1.0));
    return (datatype(1.0) - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}