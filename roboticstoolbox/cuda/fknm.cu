#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ int _inv(double *m, double *invOut);
__device__ void mult(double *A, double *B, double *C);
__device__ void copy(double *A, double *B);
__device__ void _eye(double *data);




/* 
 * Params
 *  T: double(N, 4, 4) the final transform matrix of all points (shared)
 *  tool: double(N, 4, 4) the tool transform matrix of all points (shared)
 *  nlinks_pt: long(N,): the number of links associated with each (shared)
 *  link_A: double(max_nlinks, 4, 4) the transformation matrix of all joints
 *  link_axes: long(max_nlinks, ): axes of all links
 *  link_isjoint: long(max_nlinks, ): 1/0 whether links are joints
 *  N: (int) number of points
 *  njoints: (int) number of joints
 *  out: (N, 6, njoints)
 */
__global__ void _jacob0(double *T,
                        double *tool, 
                        double *etool, 
                        double *link_A, 
                        long *nlinks_pt,
                        long *link_axes,
                        long *link_isjoint, 
                        int N, 
                        int max_nlinks, 
                        int njoints, 
                        double *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double *T_i;
    double *tool_i;
    double *U;
    double *temp;
    double *etool_i;
    double *invU;
    double *link_iA;
    U = (double*) malloc(sizeof(double) * 16);
    invU = (double*) malloc(sizeof(double) * 16);
    temp = (double*) malloc(sizeof(double) * 16);
    int j = 0;

    tool_i = &tool[tid * 16];
    etool_i = &etool[tid * 16];

    _eye(U);
    T_i = &T[tid * 16];

    if (tid >= N) {
        free(U);
        free(invU);
        free(temp);
        return;
    }

    long nlinks = nlinks_pt[tid];
    double *link_A_tid = &link_A[tid * max_nlinks * 4 * 4];
    // printf("Hello from tid %d nlinks %ld\n", tid, nlinks);
    for (int i = 0; i < nlinks; i++) {
        // printf("Hello from tid %d link_i %d link_axis %ld isjoint %ld \n", tid, i, link_axes[i], link_isjoint[i]);

        if (link_isjoint[i] == 1) {
            link_iA = &link_A_tid[i * 16];
            mult(U, link_iA, temp);
            copy(temp, U);

            if (i == nlinks - 1) {
                mult(U, etool_i, temp);
                copy(temp, U);
                mult(U, tool_i, temp);
                copy(temp , U);   
            }

            _inv(U, invU);
            mult(invU, T_i, temp);

            double *out_tid = &out[tid * 6 * njoints];

            if (link_axes[i] == 0) {
                out_tid[0 * njoints + j] = U[0 * 4 + 2] * temp[1 * 4 + 3] - U[0 * 4 + 1] * temp[2 * 4 + 3];
                out_tid[1 * njoints + j] = U[1 * 4 + 2] * temp[1 * 4 + 3] - U[1 * 4 + 1] * temp[2 * 4 + 3];
                out_tid[2 * njoints + j] = U[2 * 4 + 2] * temp[1 * 4 + 3] - U[2 * 4 + 1] * temp[2 * 4 + 3];
                out_tid[3 * njoints + j] = U[0 * 4 + 2];
                out_tid[4 * njoints + j] = U[1 * 4 + 2];
                out_tid[5 * njoints + j] = U[2 * 4 + 2];
            }
            else if (link_axes[i] == 1)
            {
                out_tid[0 * njoints + j] = U[0 * 4 + 0] * temp[2 * 4 + 3] - U[0 * 4 + 2] * temp[0 * 4 + 3];
                out_tid[1 * njoints + j] = U[1 * 4 + 0] * temp[2 * 4 + 3] - U[1 * 4 + 2] * temp[0 * 4 + 3];
                out_tid[2 * njoints + j] = U[2 * 4 + 0] * temp[2 * 4 + 3] - U[2 * 4 + 2] * temp[0 * 4 + 3];
                out_tid[3 * njoints + j] = U[0 * 4 + 1];
                out_tid[4 * njoints + j] = U[1 * 4 + 1];
                out_tid[5 * njoints + j] = U[2 * 4 + 1];
            }
            else if (link_axes[i] == 2)
            {
                out_tid[0 * njoints + j] = U[0 * 4 + 1] * temp[0 * 4 + 3] - U[0 * 4 + 0] * temp[1 * 4 + 3];
                out_tid[1 * njoints + j] = U[1 * 4 + 1] * temp[0 * 4 + 3] - U[1 * 4 + 0] * temp[1 * 4 + 3];
                out_tid[2 * njoints + j] = U[2 * 4 + 1] * temp[0 * 4 + 3] - U[2 * 4 + 0] * temp[1 * 4 + 3];
                out_tid[3 * njoints + j] = U[0 * 4 + 2];
                out_tid[4 * njoints + j] = U[1 * 4 + 2];
                out_tid[5 * njoints + j] = U[2 * 4 + 2];
            }
            else if (link_axes[i] == 3)
            {
                out_tid[0 * njoints + j] = U[0 * 4 + 0];
                out_tid[1 * njoints + j] = U[1 * 4 + 0];
                out_tid[2 * njoints + j] = U[2 * 4 + 0];
                out_tid[3 * njoints + j] = 0.0;
                out_tid[4 * njoints + j] = 0.0;
                out_tid[5 * njoints + j] = 0.0;
            }
            else if (link_axes[i] == 4)
            {
                out_tid[0 * njoints + j] = U[0 * 4 + 1];
                out_tid[1 * njoints + j] = U[1 * 4 + 1];
                out_tid[2 * njoints + j] = U[2 * 4 + 1];
                out_tid[3 * njoints + j] = 0.0;
                out_tid[4 * njoints + j] = 0.0;
                out_tid[5 * njoints + j] = 0.0;
            }
            else if (link_axes[i] == 5)
            {
                out_tid[0 * njoints + j] = U[0 * 4 + 2];
                out_tid[1 * njoints + j] = U[1 * 4 + 2];
                out_tid[2 * njoints + j] = U[2 * 4 + 2];
                out_tid[3 * njoints + j] = 0.0;
                out_tid[4 * njoints + j] = 0.0;
                out_tid[5 * njoints + j] = 0.0;
            }
            j++;
        } 
        else 
        {
            link_iA = &link_A_tid[i * 16];    
            mult(U, link_iA, temp);
            copy(temp, U);
        }
    }

    free(U);
    free(invU);
    free(temp);
}


__device__ void _eye(double *data)
{
    data[0] = 1;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
    data[4] = 0;
    data[5] = 1;
    data[6] = 0;
    data[7] = 0;
    data[8] = 0;
    data[9] = 0;
    data[10] = 1;
    data[11] = 0;
    data[12] = 0;
    data[13] = 0;
    data[14] = 0;
    data[15] = 1;
}

__device__ void copy(double *A, double *B)
{
    // copy A into B
    B[0] = A[0];
    B[1] = A[1];
    B[2] = A[2];
    B[3] = A[3];
    B[4] = A[4];
    B[5] = A[5];
    B[6] = A[6];
    B[7] = A[7];
    B[8] = A[8];
    B[9] = A[9];
    B[10] = A[10];
    B[11] = A[11];
    B[12] = A[12];
    B[13] = A[13];
    B[14] = A[14];
    B[15] = A[15];
}

__device__ void mult(double *A, double *B, double *C)
{
    const int N = 4;
    int i, j, k;
    double num;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            num = 0;
            for (k = 0; k < N; k++)
            {
                num += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = num;
        }
    }
}

__device__ int _inv(double *m, double *invOut)
{
    double *inv = (double*) malloc(sizeof(double) * 16);
    double det;
    int i;

    inv[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[14] -
             m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] -
             m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] +
             m[4] * m[11] * m[14] +
             m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] +
             m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] -
             m[4] * m[11] * m[13] -
             m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] +
              m[4] * m[10] * m[13] +
              m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] +
             m[1] * m[11] * m[14] +
             m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[14] -
             m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] +
             m[0] * m[11] * m[13] +
             m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] -
              m[0] * m[10] * m[13] -
              m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[14] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] +
             m[0] * m[7] * m[14] +
             m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[13] -
              m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] +
              m[0] * m[6] * m[13] +
              m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0) {
        free(inv);
        return 0;
    }

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    free(inv);
    return 1;
}



extern "C"{

/* 
 * Params
 *  T: double(N, 4, 4) the final transform matrix of all points (shared)
 *  tool: double(N, 4, 4) the tool transform matrix of all points (shared)
 *  nlinks_pt: long(N,): the number of links associated with each (shared)
 *  link_A: double(N, max_nlinks, 4, 4) the transformation matrix of all joints
 *  link_axes: long(max_nlinks, ): axes of all links
 *  link_isjoint: long(max_nlinks, ): 1/0 whether links are joints
 *  N: (int) number of points
 *  max_nlinks: (int) max number of links on the path
 *  njoints: (int) number of joints
 *  out: (N, 6, njoints)
 */
void jacob0(double *T, 
            double *tool,
            double *etool,
            double *link_A, 
            long *nlinks_pt,
            long *link_axes,
            long *link_isjoint, 
            int N, 
            int max_nlinks, 
            int njoints, 
            double *out)
{
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    // printf("Block size %d N %d gid size %d\n", block_size, N, grid_size);

    double *d_T, *d_tool, *d_etool, *d_link_A;
    long *d_link_axes, *d_link_isjoint, *d_nlinks_pt;
    double *d_out;

    cudaMalloc((void**)&d_T, sizeof(double) * N * 16);
    cudaMalloc((void**)&d_tool, sizeof(double) * N * 16);
    cudaMalloc((void**)&d_etool, sizeof(double) * N * 16);
    cudaMalloc((void**)&d_link_A, sizeof(double) * N * max_nlinks * 16);
    cudaMalloc((void**)&d_nlinks_pt, sizeof(long) * N);
    cudaMalloc((void**)&d_link_axes, sizeof(long) * max_nlinks);
    cudaMalloc((void**)&d_link_isjoint, sizeof(long) * max_nlinks);
    cudaMalloc((void**)&d_out, sizeof(double) * N * 6 * njoints);


    // Transfer data from host to device memory
    cudaMemcpy(d_T, T, sizeof(double) * N * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tool, tool, sizeof(double) * N * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_etool, etool, sizeof(double) * N * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_link_A, link_A, sizeof(double) * N * max_nlinks * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nlinks_pt, nlinks_pt, sizeof(long) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_link_axes, link_axes, sizeof(long) * max_nlinks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_link_isjoint, link_isjoint, sizeof(long) * max_nlinks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, sizeof(double) * N * 6 * njoints, cudaMemcpyHostToDevice);


    _jacob0<<<grid_size,block_size>>>(d_T, 
                                      d_tool,
                                      d_etool,
                                      d_link_A, 
                                      d_nlinks_pt,
                                      d_link_axes,
                                      d_link_isjoint,
                                      N,
                                      max_nlinks,
                                      njoints,
                                      d_out);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    // if (cudaerr != cudaSuccess)
    //     printf("kernel launch failed with error \"%s\".\n",
    //            cudaGetErrorString(cudaerr));

    // memset(out, 1, N * 6 * njoints);
    // out[0] = 1;
    cudaMemcpy(out, d_out, sizeof(double) * N * 6 * njoints, cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_T);
    cudaFree(d_tool);
    cudaFree(d_nlinks_pt);
    cudaFree(d_etool);
    cudaFree(d_link_A);
    cudaFree(d_link_axes);
    cudaFree(d_link_isjoint);
    cudaFree(d_out);
}


}//extern "C"