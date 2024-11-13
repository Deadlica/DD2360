#include "sphere.h"

sphere::sphere(const point3& center, double radius): _center(center), _radius(std::fmax(0, radius)) {}

bool sphere::hit(const ray& r, interval ray_t, hit_record& rec) const {
    vec3 oc = _center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - _radius * _radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0) {
        return false;
    }

    auto sqrtd = std::sqrt(discriminant);

    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (h + sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            return false;
        }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - _center) / _radius;
    rec.set_face_normal(r, outward_normal);

    return true;
}
