#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <stdexcept>
#include <random>
#include <curand_kernel.h>

constexpr uint TPB = 16;

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType* A, DataType* B, DataType* C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
    //@@ Insert code to implement matrix multiplication here
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    DataType sum = 0;
    if( col >= numBColumns || row >= numARows) return;
    for(int i = 0; i < numAColumns; i++) {
        sum += A[row * numAColumns + i] * B[i * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
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
    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
    if (argc != 4) {
        printf("Missing arguments\n");
        return 0;
    }
    try {
        numARows = std::stoi(argv[1]);
        numAColumns = std::stoi(argv[2]);
        numBRows = numAColumns;
        numBColumns = std::stoi(argv[3]);
        numCRows = numARows;
        numCColumns = numBColumns;
    } catch (std::invalid_argument &e) {
        printf("Bad input\n");
        return 0;
    }

    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    size_t allocationSizeA = numARows * numAColumns * sizeof(DataType);
    size_t allocationSizeB = numBRows * numBColumns * sizeof(DataType);
    size_t allocationSizeC = numCRows * numCColumns * sizeof(DataType);

    //@@ Insert code below to allocate Host memory for input and output
    hostA = (DataType*) malloc(allocationSizeA);
    hostB = (DataType*) malloc(allocationSizeB);
    hostC = (DataType*) malloc(allocationSizeC);
    resultRef = (DataType*) malloc(allocationSizeC);


    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 100);
    for (int i = 0; i < numARows * numAColumns; i++) hostA[i] = dis(gen);
    for (int i = 0; i < numBRows * numBColumns; i++) hostB[i] = dis(gen);
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            DataType temp = 0.0;
            for (int k = 0; k < numAColumns; k++) {
                temp += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            }
            resultRef[i * numBColumns + j] = temp;
        }
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceA, allocationSizeA);
    cudaMalloc(&deviceB, allocationSizeB);
    cudaMalloc(&deviceC, allocationSizeC);

    //@@ Insert code to below to Copy memory to the GPU here
    timer_start(&start);
    cudaMemcpy(deviceA, hostA, allocationSizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, allocationSizeB, cudaMemcpyHostToDevice);
    timer_stop(&start, &timeHostToDevice);

    //@@ Initialize the grid and block dimensions here
    dim3 db = dim3(TPB, TPB);
    dim3 dg((numCColumns + TPB - 1) / TPB, (numCRows + TPB - 1) / TPB);

    //@@ Launch the GPU Kernel here
    timer_start(&start);
    gemm<<<dg, db>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();
    timer_stop(&start, &timeKernel);


    //@@ Copy the GPU memory back to the CPU here
    timer_start(&start);
    cudaMemcpy(hostC, deviceC, allocationSizeC, cudaMemcpyDeviceToHost);
    timer_stop(&start, &timeDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < numCRows * numCColumns; i++) {
        if (hostC[i] - resultRef[i] > 1e-6) {
            printf("hostC != resultRef\n");
            break;
        }
    }

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    //@@ Free the CPU memory here
    free(hostA);
    free(hostB);
    free(hostC);

    //@@ Print the time elapsed here
    printf("Host to Device transfer time: %f s\n", timeHostToDevice);
    printf("Kernel execution time: %f s\n", timeKernel);
    printf("Device to Host transfer time: %f s\n", timeDeviceToHost);

    return 0;
}
