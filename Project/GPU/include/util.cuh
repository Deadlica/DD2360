#ifndef GPU_UTIL_CUH
#define GPU_UTIL_CUH

// Project
#include <vec3.cuh>

// std
#include <iostream>
#include <sys/time.h>
#include <curand_kernel.h>

#define datatype float
constexpr dim3 TPB = {32, 32};
__device__ const datatype infinity = datatype(1.0) / datatype(0.0);
//__device__ constexpr datatype pi = 3.1415926535897932385;
//__device__ constexpr datatype eps = 1e-160;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))
__device__ inline vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = datatype(2.0) * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= datatype(1.0));
    return p;
}

inline void timer_start(struct timeval* start) {
    gettimeofday(start, nullptr);
}

inline void timer_stop(struct timeval* start, double* elapsed) {
    struct timeval end{};
    gettimeofday(&end, nullptr);
    *elapsed = static_cast<double>((end.tv_sec - start->tv_sec)) +
               static_cast<double>((end.tv_usec - start->tv_usec)) /
               1000000.0;
}

#endif //GPU_UTIL_CUH
