// Project
#include <ray.h>

ray::ray() {}

ray::ray(const point3& origin, const vec3& direction): _origin(origin), _direction(direction) {}

const point3& ray::origin() const {
    return _origin;
}

const vec3& ray::direction() const {
    return _direction;
}

point3 ray::at(datatype t) const {
    return _origin + t * _direction;
}
