#ifndef PROJECT_CAMERA_H
#define PROJECT_CAMERA_H

#include "hittable.h"
#include "color.h"

class camera {
public:
    datatype aspect_ratio = 1.0;
    int      image_width  = 100;

    void render(const hittable& world);

private:
    int    _image_height;
    point3 _center;
    point3 _pixel00_loc;
    vec3   _pixel_delta_x;
    vec3   _pixel_delta_y;

    void initialize();
    color ray_color(const ray& r, const hittable& world) const;
};


#endif //PROJECT_CAMERA_H
