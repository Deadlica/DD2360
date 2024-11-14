#include "camera.h"
#include "material.h"

void camera::render(const hittable& world) {
    initialize();

    std::cout << "P3\n" << image_width << " " << _image_height << "\n255\n";

    for (int j = 0; j < _image_height; j++) {
        std::clog << "\rScanlines remaining: " << (_image_height - j) << " " << std::flush;
        for (int i = 0; i < image_width; i++) {
            color pixel_color(0, 0, 0);
            for (int sample = 0; sample < samples_per_pixel; sample++) {
                ray r = get_ray(i, j);
                pixel_color += ray_color(r, max_depth, world);
            }
            write_color(std::cout, _pixel_samples_scale * pixel_color);
        }
    }
    std::clog << "\rDone.                 \n";
}

void camera::initialize() {
    _image_height = int(image_width / aspect_ratio);
    _image_height = _image_height < 1 ? 1 : _image_height;

    _pixel_samples_scale = 1.0 / samples_per_pixel;

    _center = lookfrom;

    // Camera
    datatype focal_length = (lookfrom - lookat).length();
    datatype theta = degrees_to_radians(vfov);
    datatype h = std::tan(theta / 2);
    datatype viewport_height = 2 * h * focal_length;
    datatype viewport_width = viewport_height * (datatype(image_width) / _image_height);
    point3 camera_center = point3(0, 0, 0);

    // Camera position
    _w = unit_vector(lookfrom - lookat);
    _u = unit_vector(cross(vup, _w));
    _v = cross(_w, _u);

    // Viewports
    auto viewport_x = viewport_width * _u;
    auto viewport_y = viewport_height * -_v;

    // Pixel distance
    _pixel_delta_x = viewport_x / image_width;
    _pixel_delta_y = viewport_y / _image_height;

    // Start pos
    auto viewport_upper_left = _center - (focal_length * _w) - viewport_x / 2 - viewport_y / 2;
    _pixel00_loc = viewport_upper_left + 0.5 * (_pixel_delta_x + _pixel_delta_y);
}

ray camera::get_ray(int i, int j) const {
    auto offset = sample_square();
    auto pixel_sample = _pixel00_loc
                      + ((i + offset.x()) * _pixel_delta_x)
                      + ((j + offset.y()) * _pixel_delta_y);

    auto ray_origin = _center;
    auto ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}

vec3 camera::sample_square() const {
    return vec3(random_real_number() - 0.5, random_real_number() - 0.5, 0);
}

vec3 camera::sample_disk(datatype radius) const {
    //return radius * random_in_unit_disk();
    return vec3();
}

color camera::ray_color(const ray& r, int depth, const hittable& world) const {
    if (depth <= 0) {
        return color(0, 0, 0);
    }

    hit_record rec;

    if (world.hit(r, interval(0.001, infinity), rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat->scatter(r, rec, attenuation, scattered)) {
            return attenuation * ray_color(scattered, depth - 1, world);
        }
        return color(0, 0, 0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}
