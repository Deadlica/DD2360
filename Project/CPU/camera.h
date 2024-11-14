#ifndef PROJECT_CAMERA_H
#define PROJECT_CAMERA_H

#include "hittable.h"
#include "color.h"

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

    void render(const hittable& world);

private:
    int      _image_height;
    datatype _pixel_samples_scale;
    point3   _center;
    point3   _pixel00_loc;
    vec3     _pixel_delta_x;
    vec3     _pixel_delta_y;
    vec3     _u;
    vec3     _v;
    vec3     _w;

    void initialize();
    ray get_ray(int i, int j) const;
    vec3 sample_square() const;
    vec3 sample_disk(datatype radius) const;
    color ray_color(const ray& r, int depth, const hittable& world) const;
};


#endif //PROJECT_CAMERA_H
