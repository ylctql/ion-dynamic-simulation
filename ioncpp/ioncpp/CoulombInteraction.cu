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

    printf("Using fastmath ...\n");
    
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


__global__ void computeCoulombInteraction_block_based(
    data_t* r, data_t* charge, data_t* result, int N) {

    printf("Using block based...\n");
    
    // 每个块处理一个粒子i
    int i = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    extern __shared__ data_t shared_data[];
    data_t* shared_r = shared_data;
    data_t* shared_charge = &shared_data[DIM * block_size]; //shared_charge紧跟着shared_data存储
    
    data_t force_i[DIM] = {0.0};
    data_t pos_i[DIM];
    data_t charge_i = charge[i];
    
    // 加载粒子i的位置
    for (int d = 0; d < DIM; ++d) {
        pos_i[d] = r[d * N + i];
    }
    
    // 分块处理j > i的粒子
    for (int j_start = i + 1; j_start < N; j_start += block_size) {
        int j = j_start + tid;
        
        // 协作加载粒子数据到共享内存
        if (j < N) {
            for (int d = 0; d < DIM; ++d) {
                shared_r[d * block_size + tid] = r[d * N + j];
            }
            shared_charge[tid] = charge[j];
        }
        __syncthreads();
        
        // 计算与当前块的相互作用
        int j_end = min(j_start + block_size, N);
        for (int j_local = 0; j_local < j_end - j_start; ++j_local) {
            int j_global = j_start + j_local;
            
            data_t dist2 = 0.0;
            data_t diff[DIM];
            data_t force_ij[DIM] = {0.0};
            
            for (int d = 0; d < DIM; ++d) {
                diff[d] = pos_i[d] - shared_r[d * block_size + j_local];
                dist2 += diff[d] * diff[d];
            }
            
            data_t inv_dist = rsqrt(dist2);
            data_t inv_dist3 = inv_dist * inv_dist * inv_dist;
            data_t factor = charge_i * shared_charge[j_local] * inv_dist3;
            
            for (int d = 0; d < DIM; ++d) {
                force_ij[d] = diff[d] * factor;
                force_i[d] += force_ij[d];
            }
            
            for (int d = 0; d < DIM; ++d) {
                // result[d * N + j_global] -= force_ij[d];
                atomicAdd(&result[d * N + j_global], -force_ij[d]);
            }
        }
        __syncthreads();
    }
    
    // 写入粒子i的力（连续访问）
    for (int d = 0; d < DIM; ++d) {
        result[d * N + i] += force_i[d];
    }
}

__global__ void computeCoulombInteraction_pair(
    data_t* r, data_t* charge, data_t* result, int N) {

    // printf("Using pair ...\n");
    
    // 每个线程处理一个唯一的粒子对 (i,j)，其中 i < j
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = N * (N - 1) / 2;

    if (tid >= total_pairs) return;
    
    // 将线性索引映射到粒子对 (i,j)
    // 1) 计算 i：使用二次式的解并取 floor
    double a = (double)(2 * N - 1);
    double discr = a * a - 8.0 * (double)tid; // >=0
    int i = (int)floor((a - sqrt(discr)) * 0.5);

    // // 2) 计算 C(i) 并得到 j
    int Ci = i * (2 * N - i - 1) / 2;
    int j = int(tid - Ci + i + 1);

    
    data_t pos_i[DIM], pos_j[DIM];
    data_t charge_i = charge[i];
    data_t charge_j = charge[j];
    
    // 加载粒子位置
    for (int d = 0; d < DIM; ++d) {
        pos_i[d] = r[d * N + i];
        pos_j[d] = r[d * N + j];
    }
    
    // 计算距离和力
    data_t dist2 = 0.0;
    data_t diff[DIM];
    data_t force_ij[DIM] = {0.0};
    
    for (int d = 0; d < DIM; ++d) {
        diff[d] = pos_i[d] - pos_j[d];
        dist2 += diff[d] * diff[d];
    }
    
    data_t inv_dist = rsqrt(dist2);
    data_t inv_dist3 = inv_dist * inv_dist * inv_dist;
    data_t factor = charge_i * charge_j * inv_dist3;
    
    for (int d = 0; d < DIM; ++d) {
        force_ij[d] = diff[d] * factor;
    }
    
    // 原子更新两个粒子（每个粒子只被一个线程更新）
    for (int d = 0; d < DIM; ++d) {
        atomicAdd(&result[d * N + i], force_ij[d]);
        atomicAdd(&result[d * N + j], -force_ij[d]);
    }

    // printf("tid: %d, i: %d, j: %d\n", tid, i, j);
}


__global__ void computeCoulombInteraction_shared(
    const data_t* __restrict__ r,
    const data_t* __restrict__ charge,
    data_t* __restrict__ result,
    int N) 
{   
    printf("Using shared memories ... \n");
    extern __shared__ data_t sh_force[]; // 大小 = blockDim.x * DIM

    int i = blockIdx.x;  // 每个 block 负责一个粒子 i
    int tid = threadIdx.x;

    // 初始化共享内存
    for (int d = 0; d < DIM; ++d) {
        sh_force[tid * DIM + d] = 0.0;
    }

    __syncthreads();

    // 每个线程处理多个 j
    for (int j = tid; j < N; j += blockDim.x) {
        if (i == j) continue;

        data_t diff[DIM];
        data_t dist2 = 0.0;

        for (int d = 0; d < DIM; ++d) {
            diff[d] = r[d * N + i] - r[d * N + j];
            dist2 += diff[d] * diff[d];
        }

        data_t inv_dist = rsqrt(dist2);
        data_t inv_dist3 = inv_dist * inv_dist * inv_dist;
        data_t factor = charge[i] * charge[j] * inv_dist3;

        for (int d = 0; d < DIM; ++d) {
            sh_force[tid * DIM + d] += diff[d] * factor;
        }
    }

    __syncthreads();

    // 归约：把 block 内所有线程的贡献加起来
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int d = 0; d < DIM; ++d) {
                sh_force[tid * DIM + d] += sh_force[(tid + stride) * DIM + d];
            }
        }
        __syncthreads();
    }

    // 由线程 0 写回全局内存
    if (tid == 0) {
        for (int d = 0; d < DIM; ++d) {
            result[d * N + i] = sh_force[d];
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
    computeCoulombInteraction_block_based<<<grid_size, block_size>>>(r, charge, result, N);
    cudaDeviceSynchronize();
}
