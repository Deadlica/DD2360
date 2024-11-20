#ifndef GPU_INTERVAL_CUH
#define GPU_INTERVAL_CUH

// Project
#include <util.cuh>
#include <vec3.cuh>

class interval {
public:
    datatype min, max;

    __host__ __device__ interval();
    __host__ __device__ interval(datatype min, datatype max);

    __device__ datatype size() const;
    __device__ bool contains(datatype x) const;
    __device__ bool surrounds(datatype x) const;
    __host__ __device__ datatype clamp(datatype x) const;

    static const interval empty, universe;
};

#endif //GPU_INTERVAL_CUH
