#include <cuda_runtime.h>
#include "RungeKutta.hpp"

using data_t = double; 
constexpr int DIM = 3;

__global__ void TransposeKernel(data_t* src, data_t* dst, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows && j < cols) {
        // src: [rows][cols] 布局 -> src[i * cols + j]
        // dst: [cols][rows] 布局 -> dst[j * rows + i]
        dst[j * rows + i] = src[i * cols + j];
    }
}

extern "C" void transposeGPU(data_t* src, data_t* dst, int rows, int cols) {
    dim3 blockSize(16, 16);
    dim3 gridSize((rows + 15) / 16, (cols + 15) / 16);
    
    TransposeKernel<<<gridSize, blockSize>>>(src, dst, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void computeCoulombInteractionKernel(
    data_t* r, data_t* charge, data_t* result, int N) {

    extern __shared__ data_t shared_r[]; // Shared memory for particle positions

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;

    data_t force[DIM] = {0.0};
    data_t pos_i[DIM];
    data_t charge_i = charge[i];

    // Load the position of the current particle into registers
    for (int d = 0; d < DIM; ++d) {
        pos_i[d] = r[d * N + i];
    }

    // Iterate over tiles of particles
    for (int tile_start = 0; tile_start < N; tile_start += blockDim.x) {
        // Load a tile of particle positions into shared memory
        int j = tile_start + tid;
        for (int d = 0; d < DIM; ++d) {
            if (j < N) {
                shared_r[d * blockDim.x + tid] = r[d * N + j];
            } else {
                shared_r[d * blockDim.x + tid] = 0.0; // Padding for out-of-bounds threads
            }
        }
        __syncthreads();

        // Compute interactions within the tile
        for (int j_in_tile = 0; j_in_tile < blockDim.x; ++j_in_tile) {
            int j = tile_start + j_in_tile;
            if (j >= N || i == j) continue;

            data_t dist2 = 0.0;
            data_t diff[DIM];

            for (int d = 0; d < DIM; ++d) {
                diff[d] = pos_i[d] - shared_r[d * blockDim.x + j_in_tile];
                dist2 += diff[d] * diff[d];
            }

            data_t inv_dist = rsqrt(dist2);
            data_t inv_dist3 = inv_dist * inv_dist * inv_dist;
            data_t factor = charge_i * charge[j] * inv_dist3;

            for (int d = 0; d < DIM; ++d) {
                force[d] += diff[d] * factor;
            }
        }
        __syncthreads();
    }

    // Write the computed force to global memory
    for (int d = 0; d < DIM; ++d) {
        result[d * N + i] = force[d];
    }
}

extern "C" void computeCoulombInteraction(data_t* r, data_t* charge, data_t* result, int N, int grid_size, int block_size)  
{  
    int shared_bytes = block_size * DIM * sizeof(data_t); // Shared memory size
    computeCoulombInteractionKernel<<<grid_size, block_size, shared_bytes>>>(r, charge, result, N);
    cudaDeviceSynchronize();
}