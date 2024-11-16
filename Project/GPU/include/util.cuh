#ifndef GPU_UTIL_CUH
#define GPU_UTIL_CUH

// std
#include <iostream>
#include <sys/time.h>

#define datatype float
constexpr dim3 TPB = {32, 32};
//__device__ constexpr datatype infinity = std::numeric_limits<datatype>::infinity();
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
