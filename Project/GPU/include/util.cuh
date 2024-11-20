#ifndef GPU_UTIL_CUH
#define GPU_UTIL_CUH

// Project
#include <vec3.cuh>

// std
#include <iostream>
#include <sys/time.h>
#include <curand_kernel.h>
#include <float.h>

#ifndef datatype
#define datatype float
#endif
#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif
#if   defined(datatype) && datatype == float
#define infinity FLT_MAX
#elif defined(datatype) && datatype == double
#define infinity DBL_MAX
#endif

constexpr dim3 TPB = {32, 32};

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

__host__ __device__ inline datatype degrees_to_radians(datatype degrees) {
    return degrees * M_PI / datatype(180.0);
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
