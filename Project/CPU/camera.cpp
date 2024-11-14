#include "camera.h"

void camera::render(const hittable& world) {
    initialize();

    std::cout << "P3\n" << image_width << " " << _image_height << "\n255\n";

    for (int j = 0; j < _image_height; j++) {
        std::clog << "\rScanlines remaining: " << (_image_height - j) << " " << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto pixel_center = _pixel00_loc + (i * _pixel_delta_x) + (j * _pixel_delta_y);
            auto ray_direction = pixel_center - _center;
            ray r(_center, ray_direction);

            color pixel_color = ray_color(r, world);
            write_color(std::cout, pixel_color);
        }
    }
    std::clog << "\rDone.                 \n";
}

void camera::initialize() {
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

color camera::ray_color(const ray& r, const hittable& world) const {
    hit_record rec;

    if (world.hit(r, interval(0, infinity), rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}
