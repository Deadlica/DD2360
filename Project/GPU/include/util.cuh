#ifndef GPU_UTIL_CUH
#define GPU_UTIL_CUH

// Project
#include <vec3.cuh>

// std
#include <iostream>
#include <sys/time.h>
#include <curand_kernel.h>
#include <float.h>

constexpr dim3 TPB = {32, 1};

/**
 * @brief Macro to check CUDA errors and terminate the program on failure.
 * @param val The result of a CUDA API call to check.
 */
#ifdef DEBUG
    #define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
    /**
     * @brief Checks the result of a CUDA API call and prints an error message if the call failed.
     * @param result The result of the CUDA API call.
     * @param func The name of the function where the error occurred.
     * @param file The file name where the error occurred.
     * @param line The line number where the error occurred.
     */
    inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                      file << ":" << line << " '" << func << "' \n";
            cudaDeviceReset();
            exit(99);
        }
    }
#else
    #define checkCudaErrors(val) (val) // No-op in release builds
#endif

/**
 * @brief Macro to generate a random `vec3` using a CUDA random state.
 */
#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))
/**
 * @brief Generates a random point inside a unit sphere.
 * @param local_rand_state A pointer to the CUDA random state.
 * @return A random `vec3` inside the unit sphere.
 */
__device__ inline vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = datatype(2.0) * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= datatype(1.0));
    return p;
}

/**
 * @brief Converts degrees to radians.
 * @param degrees The angle in degrees.
 * @return The angle in radians.
 */
__host__ __device__ inline datatype degrees_to_radians(datatype degrees) {
    return degrees * M_PI / datatype(180.0);
}

/**
 * @brief Starts a timer using `gettimeofday`.
 * @param start A pointer to a `timeval` structure where the start time will be stored.
 */
inline void timer_start(struct timeval* start) {
    gettimeofday(start, nullptr);
}

/**
 * @brief Stops the timer and calculates the elapsed time in seconds.
 * @param start A pointer to the `timeval` structure where the start time is stored.
 * @param elapsed A pointer to a `double` where the elapsed time will be stored.
 */
inline void timer_stop(struct timeval* start, double* elapsed) {
    struct timeval end{};
    gettimeofday(&end, nullptr);
    *elapsed = static_cast<double>((end.tv_sec - start->tv_sec)) +
               static_cast<double>((end.tv_usec - start->tv_usec)) /
               1000000.0;
}

#endif //GPU_UTIL_CUH
