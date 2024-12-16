#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <random>

constexpr uint TPB     = 256;
constexpr uint STREAMS = 4;

#define DataType double

__global__ void vecAdd(DataType* in1, DataType* in2, DataType* out, int len) {
    //@@ Insert code to implement vector addition here
    const uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= len) return;
    out[thread_index] = in1[thread_index] + in2[thread_index];
}

//@@ Insert code to implement timer start
void timer_start(struct timeval *start) {
    gettimeofday(start, nullptr);
}

//@@ Insert code to implement timer stop
void timer_stop(struct timeval *start, double* elapsed) {
    struct timeval end{};
    gettimeofday(&end, nullptr);
    *elapsed = static_cast<double>((end.tv_sec - start->tv_sec)) +
               static_cast<double>((end.tv_usec - start->tv_usec)) /
               1000000.0;
}


int main(int argc, char **argv) {
    struct timeval start{};
    double timeHostToDevice;
    double timeKernel;
    double timeDeviceToHost;
    int segmentLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    //@@ Insert code below to read in inputLength from args
    if (argc != 2) {
        printf("No segment length was provided\n");
        return 0;
    }
    try {
        segmentLength = std::stoi(argv[1]);
    } catch (std::invalid_argument &e) {
        printf("Invalid segment length provided, got: %s\n", argv[1]);
        return 0;
    }

    printf("The segment length is %d\n", segmentLength);
    size_t allocationSize = STREAMS * segmentLength * sizeof(DataType);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (DataType*) malloc(allocationSize);
    hostInput2 = (DataType*) malloc(allocationSize);
    hostOutput = (DataType*) malloc(allocationSize);
    resultRef = (DataType*) malloc(allocationSize);

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<DataType> urd(0, STREAMS * segmentLength + 1);
    for (int i = 0; i < STREAMS * segmentLength; i++) {
        hostInput1[i] = urd(gen);
        hostInput2[i] = urd(gen);
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput1, allocationSize);
    cudaMalloc(&deviceInput2, allocationSize);
    cudaMalloc(&deviceOutput, allocationSize);

    //@@ Initialize streams
    uint streamBytes = segmentLength * sizeof(DataType);
    cudaStream_t stream[STREAMS];
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 db(TPB);
    dim3 dg((segmentLength + db.x - 1) / db.x);

    timer_start(&start);
    //@@ Insert code to below to Copy memory to the GPU here
    for (int i = 0; i < STREAMS; i++) {
        uint offset = i * segmentLength;
        cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes,
                        cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes,
                        cudaMemcpyHostToDevice, stream[i]);
    }
    timer_stop(&start, &timeHostToDevice);

    timer_start(&start);
    //@@ Launch the GPU Kernel here
    for (int i = 0; i < STREAMS; i++) {
        uint offset = i * segmentLength;
        vecAdd<<<dg, db, 0, stream[i]>>>(&deviceInput1[offset],
                                         &deviceInput2[offset],
                                         &deviceOutput[offset],
                                         segmentLength);
    }
    timer_stop(&start, &timeKernel);

    timer_start(&start);
    //@@ Copy the GPU memory back to the CPU here
    for (int i = 0; i < STREAMS; i++) {
        uint offset = i * segmentLength;
        cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset],
                        streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }
    timer_stop(&start, &timeDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < STREAMS * segmentLength; i++) {
        if (hostOutput[i] != resultRef[i]) {
            printf("hostOutput[%d] != resultRef[%d]", i, i);
        }
    }

    //@@ Destroy streams
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    //@@ Print the time elapsed here
    printf("Host to Device transfer time: %f s\n", timeHostToDevice);
    printf("Kernel execution time: %f s\n", timeKernel);
    printf("Device to Host transfer time: %f s\n", timeDeviceToHost);

    return 0;
}