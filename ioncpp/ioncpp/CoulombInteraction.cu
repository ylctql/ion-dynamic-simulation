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
                data_t diff = r[d * N + i] - r[d * N + j];
                dist2 += diff * diff;
                
            }
            dist2 = sqrt(dist2) * dist2;
            for (int d = 0; d < DIM; ++d) {
                atomicAdd(&result[d * N + i], (r[d * N + i] - r[d * N + j]) * charge[i] * charge[j] / dist2);
            }
        }
    }
}

__global__ void computeCoulombInteractionKernel_optimized(
    data_t* r, data_t* charge, data_t* result, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 使用局部变量累加力，避免原子操作
    data_t force_x = 0.0, force_y = 0.0, force_z = 0.0;
    data_t charge_i = charge[i];
    
    // 预先加载当前粒子的位置到寄存器
    data_t pos_i_x = r[i];           // r[0*N + i]
    data_t pos_i_y = r[N + i];       // r[1*N + i]  
    data_t pos_i_z = r[2*N + i];     // r[2*N + i]

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        // 加载粒子j的位置
        data_t pos_j_x = r[j];
        data_t pos_j_y = r[N + j];
        data_t pos_j_z = r[2*N + j];
        
        // 计算距离分量
        data_t dx = pos_i_x - pos_j_x;
        data_t dy = pos_i_y - pos_j_y;
        data_t dz = pos_i_z - pos_j_z;
        
        // 计算距离平方
        data_t dist2 = dx*dx + dy*dy + dz*dz;
        
        // 避免除零错误，并计算 1/r^3
        data_t inv_dist3 = 1.0 / (sqrt(dist2) * dist2);
        
        data_t factor = charge_i * charge[j] * inv_dist3;
        
        // 累加到局部变量
        force_x += dx * factor;
        force_y += dy * factor;
        force_z += dz * factor;
    }
    
    // 一次性写入结果，无竞争
    result[i] = force_x;           // result[0*N + i]
    result[N + i] = force_y;       // result[1*N + i]
    result[2*N + i] = force_z;     // result[2*N + i]
}

__global__ void computeCoulombInteractionKernel_fastmath(
    data_t* r, data_t* charge, data_t* result, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    data_t force[DIM] = {0.0};
    data_t pos_i[DIM];
    data_t charge_i = charge[i];
    
    // 预先加载当前粒子位置
    for (int d = 0; d < DIM; ++d) {
        pos_i[d] = r[d * N + i];
    }

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        data_t dist2 = 0.0;
        data_t diff[DIM];
        
        // 计算距离向量和距离平方
        for (int d = 0; d < DIM; ++d) {
            diff[d] = pos_i[d] - r[d * N + j];
            dist2 += diff[d] * diff[d];
        }
        
        // 使用快速数学函数：rsqrt(x) = 1/sqrt(x)
        data_t inv_dist = rsqrt(dist2);
        data_t inv_dist3 = inv_dist * inv_dist * inv_dist;
        
        data_t factor = charge_i * charge[j] * inv_dist3;
        
        // 累加力
        for (int d = 0; d < DIM; ++d) {
            force[d] += diff[d] * factor;
        }
    }
    
    // 写入结果
    for (int d = 0; d < DIM; ++d) {
        result[d * N + i] = force[d];
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
    computeCoulombInteractionKernel_fastmath<<<grid_size, block_size>>>(r, charge, result, N);
    cudaDeviceSynchronize();
}