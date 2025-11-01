#include <cuda_runtime.h>
#include "RungeKutta.hpp"

using data_t = double; 
constexpr int DIM = 3;


__global__ void computeCoulombInteractionKernel(
    data_t* r, data_t* charge, data_t* result, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    data_t force[DIM] = {0.0};
    data_t pos_i[DIM];
    data_t charge_i = charge[i];
    
    for (int d = 0; d < DIM; ++d) {
        pos_i[d] = r[d * N + i];
    }

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        data_t dist2 = 0.0;
        data_t diff[DIM];

        for (int d = 0; d < DIM; ++d) {
            diff[d] = pos_i[d] - r[d * N + j];
            dist2 += diff[d] * diff[d];
        }
        
        data_t inv_dist = rsqrt(dist2);
        data_t inv_dist3 = inv_dist * inv_dist * inv_dist;
        
        data_t factor = charge_i * charge[j] * inv_dist3;
        
        for (int d = 0; d < DIM; ++d) {
            force[d] += diff[d] * factor;
        }
    }
    
    for (int d = 0; d < DIM; ++d) {
        result[d * N + i] = force[d];
    }
}

extern "C" void computeCoulombInteraction(data_t* r, data_t* charge, data_t* result, int N, int grid_size, int block_size)  
{  
    computeCoulombInteractionKernel<<<grid_size, block_size>>>(r, charge, result, N);
    cudaDeviceSynchronize();
}