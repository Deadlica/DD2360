#include <stdio.h>
#include <random>
#include <sys/time.h>

using uint = unsigned int;

static constexpr uint NUM_BINS   = 4096;
static constexpr uint TPB        = 256;
static constexpr uint SATURATION = 127;


__global__ void histogram_kernel(uint* input, uint* bins,
                                 uint inputLength, uint numBins) {


    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int jump = blockDim.x * gridDim.x;

    while (tid < inputLength) {
        uint value = input[tid];
        if (value < numBins) {
            atomicAdd(&bins[value], 1);
        }
        tid += jump;
    }

}

__global__ void convert_kernel(uint *bins, uint numBins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Saturate bins
    if (tid < numBins) {
        uint value = bins[tid];
        if (value > SATURATION) {
            bins[tid] = SATURATION;
        }
    }
}

void timer_start(struct timeval *start) {
    gettimeofday(start, nullptr);
}

void timer_stop(struct timeval *start, double* elapsed) {
    struct timeval end{};
    gettimeofday(&end, nullptr);
    *elapsed = static_cast<double>((end.tv_sec - start->tv_sec)) +
               static_cast<double>((end.tv_usec - start->tv_usec)) /
               1000000.0;
}

int main(int argc, char **argv) {
    struct timeval start{};
    double timeHistogram;
    double timeConvert;
    int inputLength;
    uint* hostInput;
    uint* hostBins;
    uint* resultRef;
    uint* deviceInput;
    uint* deviceBins;

    // Read input length from args
    if (argc > 1) {
        inputLength = std::stoi(argv[1]);
    } else {
        printf("Usage: %s <input_length>\n", argv[0]);
        return 0;
    }
    printf("The input length is %d\n", inputLength);

    size_t inputSize = inputLength * sizeof(uint);
    size_t binSize   = NUM_BINS * sizeof(uint);

    // Allocate host memory
    hostInput = (uint*) malloc(inputSize);
    hostBins  = (uint*) malloc(binSize);
    resultRef = (uint*) malloc(binSize);

    // Initialize input with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, NUM_BINS - 1);
    for (int i = 0; i < inputLength; i++) {
        hostInput[i] = dis(gen);
    }

    // Create reference result in CPU
    memset(resultRef, 0, binSize);
    for (int i = 0; i < inputLength; i++) {
        if (hostInput[i] < NUM_BINS) {
            resultRef[hostInput[i]]++;
            if (resultRef[hostInput[i]] > SATURATION) {
                resultRef[hostInput[i]] = SATURATION;
            }
        }
    }

    // Allocate GPU memory
    cudaMalloc((void**) &deviceInput, inputSize);
    cudaMalloc((void**) &deviceBins, binSize);

    // Copy memory to GPU
    cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice);
    cudaMemset(deviceBins, 0, binSize);

    // Initialize grid and block dimensions for histogram kernel
    dim3 db(TPB);
    dim3 dg((inputLength + db.x - 1) / db.x);
#ifdef PLOT
    printf("Thread block = (%d, %d, %d)\n", db.x, db.y, db.z);
    printf("Thread grid   = (%d, %d, %d)\n", dg.x, dg.y, dg.z);
#endif

    // Launch histogram kernel
    timer_start(&start);
    histogram_kernel<<<dg, db, binSize>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    timer_stop(&start, &timeHistogram);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Histogram kernel launch failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // Initialize grid and block dimensions for convert kernel
    dg.x = (NUM_BINS + db.x - 1) / db.x;

    // Launch convert kernel
    timer_start(&start);
    convert_kernel<<<dg, db>>>(deviceBins, NUM_BINS);
    timer_stop(&start, &timeHistogram);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Convert kernel launch failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // Copy results back to CPU
    cudaMemcpy(hostBins, deviceBins, binSize, cudaMemcpyDeviceToHost);

    // Verify results
    bool correct = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if (hostBins[i] != resultRef[i]) {
            printf("Mismatch at bin %d: %u (GPU) != %u (CPU)\n",
                   i, hostBins[i], resultRef[i]);
            correct = false;
            break;
        }
#ifdef PLOT
        printf("%d\n", hostBins[i]);
#endif
    }
    if (correct) {
        printf("Results match!\n");
        printf("Histogram kernel execution time: %f s\n", timeHistogram);
        printf("Convert kernel execution time: %f s\n", timeConvert);
    }

    // Free GPU memory
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    // Free CPU memory
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}