#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000


__device__ void add(float*out, float* a, float*b, int i);
__global__ void vector_add(float *out, float *a, float *b, int n);

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

__device__ void add(float*out, float* a, float*b, int i) {
    float *d_T;
    cudaMalloc((void**)&d_T, sizeof(float));
    d_T[0] = a[i] + b[i];    
    out[i] = d_T[0];
    cudaFree(d_T);
}

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        add(out, a, b, tid);
        // out[tid] = a[tid] + b[tid];
    }
}


int main(){
    float *a, *b, *out; 
    float *d_a, *d_b, *d_out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }


    // Allocate device memory 
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);


    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);

}

