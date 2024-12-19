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

int main(int argc, char **argv) {
    int       inputLength;
    float     ms;
    DataType* hostInput1;
    DataType* hostInput2;
    DataType* hostOutput;
    DataType* resultRef;
    DataType* deviceInput1;
    DataType* deviceInput2;
    DataType* deviceOutput;
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    //@@ Insert code below to read in inputLength from args
    if (argc != 2) {
        printf("No input length was provided\n");
        return 0;
    }
    try {
        inputLength = std::stoi(argv[1]);
    } catch (std::invalid_argument &e) {
        printf("Invalid input length provided, got: %s\n", argv[1]);
        return 0;
    }

    printf("The input length is %d\n", inputLength);
    size_t allocationSize = inputLength * sizeof(DataType);

    //@@ Insert code below to allocate Host memory for input and output
    cudaMallocHost(&hostInput1, allocationSize);
    cudaMallocHost(&hostInput2, allocationSize);
    cudaMallocHost(&hostOutput, allocationSize);
    //hostInput1 = (DataType*) malloc(allocationSize);
    //hostInput2 = (DataType*) malloc(allocationSize);
    //hostOutput = (DataType*) malloc(allocationSize);
    resultRef = (DataType*) malloc(allocationSize);

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<DataType> urd(0, inputLength + 1);
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = urd(gen);
        hostInput2[i] = urd(gen);
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput1, allocationSize);
    cudaMalloc(&deviceInput2, allocationSize);
    cudaMalloc(&deviceOutput, allocationSize);

    //@@ Initialize streams
    cudaStream_t stream[STREAMS];
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    const uint baseStreamSize = inputLength / STREAMS;
    const uint lastStreamSize = baseStreamSize + (inputLength % STREAMS);

    cudaEventRecord(startEvent, 0);
    //@@ Insert code to below to Copy memory to the GPU here
    for (int i = 0; i < STREAMS; i++) {
        uint offset = i * baseStreamSize;
        uint currentStreamSize = (i == STREAMS - 1) ? lastStreamSize : baseStreamSize;
        uint currentStreamBytes = currentStreamSize * sizeof(DataType);
        cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], currentStreamBytes,
                        cudaMemcpyDefault, stream[i]);
        cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], currentStreamBytes,
                        cudaMemcpyDefault, stream[i]);
    }

    //@@ Launch the GPU Kernel here
    for (int i = 0; i < STREAMS; i++) {
        uint offset = i * baseStreamSize;
        uint currentStreamSize = (i == STREAMS - 1) ? lastStreamSize : baseStreamSize;

        dim3 db(TPB);
        dim3 dg((currentStreamSize + db.x - 1) / db.x);

        vecAdd<<<dg, db, 0, stream[i]>>>(&deviceInput1[offset],
                                         &deviceInput2[offset],
                                         &deviceOutput[offset],
                                         currentStreamSize);
    }

    //@@ Copy the GPU memory back to the CPU here
    for (int i = 0; i < STREAMS; i++) {
        uint offset = i * baseStreamSize;
        uint currentStreamSize = (i == STREAMS - 1) ? lastStreamSize : baseStreamSize;
        uint currentStreamBytes = currentStreamSize * sizeof(DataType);
        cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset],
                        currentStreamBytes, cudaMemcpyDefault, stream[i]);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < inputLength; i++) {
        if (hostOutput[i] != resultRef[i]) {
            printf("hostOutput[%d] != resultRef[%d]", i, i);
        }
    }

    //@@ Destroy streams
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    //@@ Destroy events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    free(resultRef);

    //@@ Print the time elapsed here
    printf("Execution time: %f s\n", ms / 1000);

    return 0;
}