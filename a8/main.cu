#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
              
#define N 1000
#define I 100000

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// GPU - CUDA kernel A
__global__ void device_kernel_A(int *A)
{
    int sum = 0;
    for(int n = 0; n < N; n++){
        sum += A[N*n + threadIdx.x];
    }
}

// GPU - CUDA kernel B
__global__ void device_kernel_B(int *B)
{
    int sum = 0;
    for(int n = 0; n < N; n++){
        sum += B[N*threadIdx.x + n];
    }
}

double walltime(void)
{
    static struct timeval t;
    gettimeofday(&t, NULL);
    return (t.tv_sec + 1e-6 * t.tv_usec);
}

int main(int argc, char **argv)
{
    // Walltime variables
    double timeStart;
    double timeA = 0;
    double timeB = 0;

    // Host variables
    int a[N*N];
    int b[N*N];
    bool testA = true;
    bool testB = true;

    // Set some random values
    for(int y = 0; y < N; y++){
        for(int x = 0; x < N; x++){
            a[y*N + x] = x + y;
            b[y*N + x] = x + y;
        }
    }

    //********************************* GPU work start *********************************
    
    // Variables
    int *A = NULL;
    int *B = NULL;

    // Set up device memory
    cudaErrorCheck(cudaMalloc((void**)&A, N*N * sizeof(int)));
    cudaErrorCheck(cudaMalloc((void**)&B, N*N * sizeof(int)));
    
    // Copy data from host to device
    cudaErrorCheck(cudaMemcpy(A, a, N*N * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(B, b, N*N * sizeof(int), cudaMemcpyHostToDevice));

    if(testA){
        // GPU computation A
        for(int i = 0; i < I; i++){
            timeStart = walltime();
            device_kernel_A<<<1,N>>>(A);
            timeA += walltime() - timeStart;
        }
    }

    if(testB){
        // GPU computation B
        for(int i = 0; i < I; i++){
            timeStart = walltime();
            device_kernel_B<<<1,N>>>(B);
            timeB += walltime() - timeStart;
        }
    }

    // Free the device memory
    cudaErrorCheck(cudaFree(A));
    cudaErrorCheck(cudaFree(B));

    //********************************* GPU work stop *********************************

    printf("\n");
    printf("Walltime A: %7.3f ms \n", timeA * 1e3);
    printf("Walltime B: %7.3f ms \n", timeB * 1e3);
    printf("\n");

    return 0;
};
