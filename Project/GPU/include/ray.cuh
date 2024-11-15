#ifndef GPU_RAY_CUH
#define GPU_RAY_CUH

// Project
#include <vec3.cuh>

class ray {
public:
    __device__ ray();
    __device__ ray(const point3& origin, const vec3& direction);

    __device__ const point3& origin() const;
    __device__ const vec3& direction() const;

    __device__ point3 at(datatype t) const;

private:
    point3 _origin;
    vec3 _direction;
};


#endif //GPU_RAY_CUH
