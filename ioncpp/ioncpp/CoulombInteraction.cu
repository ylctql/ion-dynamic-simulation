#include <cuda_runtime.h>
#include "RungeKutta.hpp"
#include "types.hpp"

using data_t = double; 
constexpr int DIM = 3;

__global__ void computeCoulombInteractionKernel(
    data_t* r, data_t* charge, data_t* result, int N) {

    extern __shared__ data_t shared_r[]; // Shared memory for particle positions

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    printf("blockDim.x:%d \n", blockDim.x);
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

// __global__ void computeCoulombPotentialKernel(
//     data_t* r, data_t* charge, data_t* result, int N) {

//     extern __shared__ data_t shared_r[]; // Shared memory for particle positions

//     int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x + tid;
//     if (i >= N) return;
    
//     data_t potential = 0.0;
//     data_t pos_i[DIM];
//     data_t charge_i = charge[i];

//     // Load the position of the current particle into registers
//     for (int d = 0; d < DIM; ++d) {
//         pos_i[d] = r[d * N + i];
//     }

//     // Iterate over tiles of particles
//     for (int tile_start = 0; tile_start < N; tile_start += blockDim.x) {
//         // Load a tile of particle positions into shared memory
//         int j = tile_start + tid;
//         for (int d = 0; d < DIM; ++d) {
//             if (j < N) {
//                 shared_r[d * blockDim.x + tid] = r[d * N + j];
//             } else {
//                 shared_r[d * blockDim.x + tid] = 0.0; // Padding for out-of-bounds threads
//             }
//         }
//         __syncthreads();

//         // Compute interactions within the tile
//         for (int j_in_tile = 0; j_in_tile < blockDim.x; ++j_in_tile) {
//             int j = tile_start + j_in_tile;
//             if (j >= N || i == j) continue;

//             data_t dist2 = 0.0;
//             data_t diff[DIM];

//             for (int d = 0; d < DIM; ++d) {
//                 diff[d] = pos_i[d] - shared_r[d * blockDim.x + j_in_tile];
//                 dist2 += diff[d] * diff[d];
//             }

//             data_t inv_dist = rsqrt(dist2);
//             potential += charge_i * charge[j] * inv_dist;

//         }
//         __syncthreads();
//     }

//     result[i] = potential;
// }

extern "C" void computeCoulombInteraction(data_t* r, data_t* charge, data_t* result, int N, int grid_size, int block_size)  
{  
    int shared_bytes = block_size * DIM * sizeof(data_t); // Shared memory size
    computeCoulombInteractionKernel<<<grid_size, block_size, shared_bytes>>>(r, charge, result, N);
    cudaDeviceSynchronize();    //在cpp中实现udaDeviceSynchronize()和判断
}

// extern "C" void computeCoulombPotential(data_t* r, data_t* charge, data_t* result, int N, int grid_size, int block_size)  
// {  
//     int shared_bytes = block_size * DIM * sizeof(data_t); // Shared memory size
//     computeCoulombPotentialKernel<<<grid_size, block_size, shared_bytes>>>(r, charge, result, N);
// }

int main() {
    // 这里调用你的 CUDA kernel 或测试代码
    printf("Hello from CUDA program!");
    using namespace ioncpp;
    const int DIM = 3;
    VecType r {{1, 0, 0}, {-1, 0, 0}};
    ArrayType charge{1, -1};
    const auto N = r.rows();
    VecType result(N, DIM);
    result.setZero();

    data_t* d_r;
    data_t* d_charge;
    data_t* d_result;
    cudaMalloc(&d_r, N * DIM * sizeof(data_t));
    cudaMalloc(&d_charge, N * sizeof(data_t));
    cudaMalloc(&d_result, N * DIM * sizeof(data_t));

    cudaMemcpy(d_r, r.data(), N * DIM * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_charge, charge.data(), N * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, N * DIM * sizeof(data_t));

    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeCoulombInteraction(d_r, d_charge, d_result, N, blocksPerGrid, threadsPerBlock);

    cudaMemcpy(result.data(), d_result, N * DIM * sizeof(data_t), cudaMemcpyDeviceToHost);

    cudaFree(d_r);
    cudaFree(d_charge);
    cudaFree(d_result);

    cudaDeviceSynchronize();  // 可选，确保设备同步
    return 0;
}