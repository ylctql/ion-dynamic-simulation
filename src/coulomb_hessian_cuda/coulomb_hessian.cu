#include "coulomb_hessian.hpp"
#include <cuda_runtime.h>
#include <cmath>

#define MAX_N 2048
constexpr int DIM = 3;
constexpr int DIM2 = DIM*DIM;

__global__ void coulomb_hessian_kernel(
    const double* __restrict__ pos,
    const double* __restrict__ charges,
    int N,
    double prefactor,
    double* __restrict__ hessian)
{
    // Each thread deals with one ion!
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size_ion = N*DIM2;
    double res[MAX_N*DIM2] = {0.0};

    if (i >= N) return;

    for (int j=0; j<N; ++j){
        if (i != j){
            double dx = pos[DIM*i]   - pos[DIM*j];
            double dy = pos[DIM*i+1] - pos[DIM*j+1];
            double dz = pos[DIM*i+2] - pos[DIM*j+2];

            double r2 = dx*dx + dy*dy + dz*dz + 1e-16;

            double r = sqrt(r2);
            double inv_r3 = 1.0 / (r2 * r);

            double qiqj = prefactor * charges[i] * charges[j];
            double common = qiqj * inv_r3;

            double g = 3.0 / r2;

            double h_xx = common * (g * dx * dx - 1.0);
            double h_yy = common * (g * dy * dy - 1.0);
            double h_zz = common * (g * dz * dz - 1.0);
            double h_xy = common * g * dx * dy;
            double h_xz = common * g * dx * dz;
            double h_yz = common * g * dy * dz;

            double sign = -1.0;
            res[DIM2*j  ] = sign * h_xx;
            res[DIM2*j+1] = sign * h_xy;
            res[DIM2*j+2] = sign * h_xz;
            res[DIM2*j+3] = sign * h_xy;
            res[DIM2*j+4] = sign * h_yy;
            res[DIM2*j+5] = sign * h_yz;
            res[DIM2*j+6] = sign * h_xz;
            res[DIM2*j+7] = sign * h_yz;
            res[DIM2*j+8] = sign * h_zz;

            res[DIM2*i  ] -= sign * h_xx;
            res[DIM2*i+1] -= sign * h_xy;
            res[DIM2*i+2] -= sign * h_xz;
            res[DIM2*i+3] -= sign * h_xy;
            res[DIM2*i+4] -= sign * h_yy;
            res[DIM2*i+5] -= sign * h_yz;
            res[DIM2*i+6] -= sign * h_xz;
            res[DIM2*i+7] -= sign * h_yz;
            res[DIM2*i+8] -= sign * h_zz;
        }
    }
    for (int j=0; j<N; ++j){
        for (int d=0; d<DIM2; ++d){
            hessian[size_ion*i + DIM2*j + d] = res[DIM2*j + d];
        }
    }

}

__host__ void compute_coulomb_hessian_cuda(
    const double* h_pos,
    const double* h_charges,
    size_t N,
    double prefactor,
    double* h_hessian)
{
    double *d_pos = nullptr, *d_charges = nullptr, *d_hessian = nullptr;

    size_t size_pos = 3 * N * sizeof(double);
    size_t size_q   = N * sizeof(double);
    size_t size_h   = 9 * N * N * sizeof(double);

    cudaMalloc(&d_pos, size_pos);
    cudaMalloc(&d_charges, size_q);
    cudaMalloc(&d_hessian, size_h);

    cudaMemcpy(d_pos,     h_pos,     size_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_charges, h_charges, size_q,   cudaMemcpyHostToDevice);
    cudaMemset(d_hessian, 0, size_h);

    // dim3 threads(16, 16);
    // dim3 blocks((N + 15)/16, (N + 15)/16);
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    coulomb_hessian_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_pos, d_charges, static_cast<int>(N), prefactor, d_hessian);

    cudaDeviceSynchronize();

    cudaMemcpy(h_hessian, d_hessian, size_h, cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_charges);
    cudaFree(d_hessian);
}