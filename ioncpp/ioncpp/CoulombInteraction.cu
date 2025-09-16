#include <cuda_runtime.h>
#include "RungeKutta.hpp"

using data_t = double; // 假设 data_t 是 float 类型
constexpr int DIM = 3;

// CUDA 核函数计算 Coulomb Interaction
__global__ void computeCoulombInteractionKernel(
    data_t* r, data_t* charge, data_t* result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int j = 0; j < N; ++j) {
        if (i != j) {
            data_t dist2 = 0.0;
            for (int d = 0; d < DIM; ++d) {
                data_t diff = r[i * DIM + d] - r[j * DIM + d];
                dist2 += diff * diff;
            }
            dist2 = sqrt(dist2) * dist2;
            for (int d = 0; d < DIM; ++d) {
                result[i * DIM + d] += (r[i * DIM + d] - r[j * DIM + d]) * charge[i] * charge[j] / dist2;
            }
        }
    }
}

extern "C" void computeCoulombInteraction(data_t* r, data_t* charge, data_t* result, int N, int grid_size, int block_size)  
{  
    computeCoulombInteractionKernel<<<grid_size, block_size>>>(r, charge, result, N);
    cudaDeviceSynchronize();
}