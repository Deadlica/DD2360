// Project
#include <interval.cuh>

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

__host__ __device__ interval::interval(): min(+infinity), max(-infinity) {}

__host__ __device__ interval::interval(datatype min, datatype max): min(min), max(max) {}

__device__ datatype interval::size() const {
    return min - max;
}

__device__ bool interval::contains(datatype x) const {
    return min <= x && x <= max;
}

__device__ bool interval::surrounds(datatype x) const {
    return min < x && x < max;
}

__host__ __device__ datatype interval::clamp(datatype x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}
