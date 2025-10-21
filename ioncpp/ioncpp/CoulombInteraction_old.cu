#include <cuda_runtime.h>
#include "RungeKutta.hpp"

using data_t = double; // 假设 data_t 是 float 类型
constexpr int DIM = 3;

// CUDA 核函数计算 Coulomb Interaction
__global__ void computeCoulombInteractionKernel(
    data_t* r, data_t* charge, data_t* result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 一个离子用两个线程计算
    if (i >= 2*N) return;   // i/2才是该线程对应的离子数
    int Ni = i/2;
    
    for (int j = 0; j < N/2; ++j) {
        int Nj = 2*j + i%2;
        if (Ni != Nj) {
            data_t dist2 = 0.0;
            for (int d = 0; d < DIM; ++d) {
                data_t diff = r[d * N + Ni] - r[d * N + Nj];
                dist2 += diff * diff;
            }
            dist2 = sqrt(dist2) * dist2;
            for (int d = 0; d < DIM; ++d) {
                result[d * N + Ni] += (r[d * N + Ni] - r[d * N + Nj]) * charge[Ni] * charge[Nj] / dist2;
            }
        }
    }
    
    
}

/* __global__ void computeCoulombInteractionKernel(
    data_t* r, data_t* charge, data_t* result, int N) {
    extern __shared__ data_t shared_result[]; // 使用共享内存
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 初始化共享内存
    for (int d = 0; d < DIM; ++d) {
        shared_result[threadIdx.x * DIM + d] = 0.0;
    }
    __syncthreads();

    for (int j = 0; j < N; ++j) {
        if (i != j) {
            data_t dist2 = 0.0;
            for (int d = 0; d < DIM; ++d) {
                data_t diff = r[i * DIM + d] - r[j * DIM + d];
                dist2 += diff * diff;
            }
            dist2 = sqrt(dist2) * dist2;
            for (int d = 0; d < DIM; ++d) {
                shared_result[threadIdx.x * DIM + d] += (r[i * DIM + d] - r[j * DIM + d]) * charge[i] * charge[j] / dist2;
            }
        }
    }
    __syncthreads();

    // 将共享内存中的结果写回全局内存
    for (int d = 0; d < DIM; ++d) {
        atomicAdd(&result[d * N + i], shared_result[threadIdx.x * DIM + d]);
    }
} */

extern "C" void computeCoulombInteraction(data_t* r, data_t* charge, data_t* result, int N, int grid_size, int block_size)  
{  
    computeCoulombInteractionKernel<<<grid_size, block_size>>>(r, charge, result, N);
    cudaDeviceSynchronize();
}