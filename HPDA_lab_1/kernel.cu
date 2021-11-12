
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime> 
#include <iostream>
#include <chrono>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

__global__ void vecSub(int* a, int* b, int* c, int n)
{
    // Get our global thread ID
    // blockIdx - block position in grid (0 to gridDim -1)
    // threadIdx - thread index inside block
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // check for out of bounds
    if (id < n) {
        c[id] = a[id] - b[id];
    }
}

int main()
{
    int n;
    cout << "Enter size of array: ";
    cin >> n;

    // GPU
    printf("GPU implementation:\n");

    int* h_a;
    int* h_b;
    int* h_c;

    int* d_a;
    int* d_b;
    int* d_c;

    size_t bytes = n * sizeof(int);

    // host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // init arrays
    srand((unsigned)time(0));
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // copy host arrays to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    auto gpu_t1 = high_resolution_clock::now();
    vecSub << <gridSize, blockSize >> > (d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto gpu_t2 = high_resolution_clock::now();

    auto gpu_exe_time_ms = duration_cast<milliseconds>(gpu_t2 - gpu_t1);
    //duration<double, std::milli> gpu_exe_time_ms = gpu_t2 - gpu_t1;

    printf("Execution time: %d ms\n", gpu_exe_time_ms.count());

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, h_a[i], i, h_b[i], i, h_c[i]);
    }

    for (int i = n - 1; i > n - 6; i--) {
        printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, h_a[i], i, h_b[i], i, h_c[i]);
    }

    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] - h_b[i]) {
            printf("Error: c[%d] = %d but a[%d] = %d, b[%d] = %d\n", i, h_c[i], i, h_a[i], i, h_b[i]);
        }
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // CPU
    printf("\nCPU implementation:\n");

    int* cpu_a;
    int* cpu_b;
    int* cpu_c;

    cpu_a = (int*)malloc(bytes);
    cpu_b = (int*)malloc(bytes);
    cpu_c = (int*)malloc(bytes);

    memcpy(cpu_a, h_a, bytes);
    memcpy(cpu_b, h_b, bytes);

    auto cpu_t1 = high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        cpu_c[i] = cpu_a[i] - cpu_b[i];
    }
    auto cpu_t2 = high_resolution_clock::now();

    //duration<double, std::milli> cpu_exe_time_ms = cpu_t2 - cpu_t1;
    auto cpu_exe_time_ms = duration_cast<milliseconds>(cpu_t2 - cpu_t1);

    printf("Execution time: %d ms\n", cpu_exe_time_ms.count());

    for (int i = 0; i < 5; i++) {
        printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, h_a[i], i, h_b[i], i, h_c[i]);
    }

    for (int i = n - 1; i > n - 6; i--) {
        printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, h_a[i], i, h_b[i], i, h_c[i]);
    }

    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] - h_b[i]) {
            printf("Error: c[%d] = %d but a[%d] = %d, b[%d] = %d\n", i, h_c[i], i, h_a[i], i, h_b[i]);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
