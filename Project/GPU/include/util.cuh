#ifndef GPU_UTIL_H
#define GPU_UTIL_H

// std
#include <iostream>

#define datatype float
constexpr dim3 TPB = {32, 32};

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#endif //GPU_UTIL_H
