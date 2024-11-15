// Project
#include <ray.cuh>

__device__ ray::ray() {}

__device__ ray::ray(const point3& origin, const vec3& direction): _origin(origin), _direction(direction) {}

__device__ const point3& ray::origin() const {
    return _origin;
}

__device__ const vec3& ray::direction() const {
    return _direction;
}

__device__ point3 ray::at(datatype t) const {
    return _origin + t * _direction;
}
