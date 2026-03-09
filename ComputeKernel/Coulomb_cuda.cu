/**
 * Coulomb 力 CUDA 加速实现
 * 参考 outline.md 与 ism-cuda/ioncpp/ioncpp/CoulombInteraction.cu
 * 供 Coulomb.cpp 在 device==cuda 时调用
 */
#include <cuda_runtime.h>

using data_t = double;
constexpr int DIM = 3;

namespace {

constexpr data_t EPS_DIST2 = 1e-20;  // 避免 dist2=0 时 rsqrt 发散

__global__ void computeCoulombInteractionKernel(
    data_t* r, data_t* charge, data_t* result, int N) {

    // shared_r: [0, DIM*blockDim.x), shared_charge: [DIM*blockDim.x, DIM*blockDim.x+blockDim.x)
    extern __shared__ data_t shared_mem[];
    data_t* shared_r = shared_mem;
    data_t* shared_charge = shared_mem + DIM * blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;

    data_t force[DIM] = {0.0};
    data_t pos_i[DIM];
    data_t charge_i = charge[i];

    for (int d = 0; d < DIM; ++d) {
        pos_i[d] = r[d * N + i];
    }

    for (int tile_start = 0; tile_start < N; tile_start += blockDim.x) {
        int j = tile_start + tid;
        for (int d = 0; d < DIM; ++d) {
            if (j < N) {
                shared_r[d * blockDim.x + tid] = r[d * N + j];
            } else {
                shared_r[d * blockDim.x + tid] = 0.0;
            }
        }
        shared_charge[tid] = (j < N) ? charge[j] : 0.0;
        __syncthreads();

        for (int j_in_tile = 0; j_in_tile < blockDim.x; ++j_in_tile) {
            int j = tile_start + j_in_tile;
            if (j >= N || i == j) continue;

            data_t dist2 = 0.0;
            data_t diff[DIM];

            for (int d = 0; d < DIM; ++d) {
                diff[d] = pos_i[d] - shared_r[d * blockDim.x + j_in_tile];
                dist2 += diff[d] * diff[d];
            }
            dist2 = fmax(dist2, EPS_DIST2);

            data_t inv_dist = rsqrt(dist2);
            data_t inv_dist3 = inv_dist * inv_dist * inv_dist;
            data_t factor = charge_i * shared_charge[j_in_tile] * inv_dist3;

            for (int d = 0; d < DIM; ++d) {
                force[d] += diff[d] * factor;
            }
        }
        __syncthreads();
    }

    for (int d = 0; d < DIM; ++d) {
        result[d * N + i] = force[d];
    }
}

}  // namespace

/**
 * 库仑力 CUDA 计算接口
 * r, result 布局: [x0..xN-1, y0..yN-1, z0..zN-1]，与 Eigen 列主序 (N,3) 一致
 */
extern "C" void computeCoulombInteraction(
    data_t* r, data_t* charge, data_t* result, int N,
    int grid_size, int block_size) {

    int shared_bytes = block_size * (DIM + 1) * sizeof(data_t);
    computeCoulombInteractionKernel<<<grid_size, block_size, shared_bytes>>>(
        r, charge, result, N);
    cudaDeviceSynchronize();
}
