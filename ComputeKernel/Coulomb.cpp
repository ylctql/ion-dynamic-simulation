/**
 * Coulomb 力计算核心：CPU 与 CUDA 双实现
 * 参考 outline.md 与 ism-hybrid/ism-cuda RungeKutta.cpp
 */
#include "Coulomb.hpp"

#include <cstddef>
#include <cstdint>

#ifdef IONCPP_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

#ifdef IONCPP_USE_CUDA
extern "C" void computeCoulombInteraction(
    double* r, double* charge, double* result, int N,
    int grid_size, int block_size);
#endif

}  // namespace

namespace ioncpp {

namespace {

// CPU 实现：参考 ism-hybrid RungeKutta.cpp CoulombInteraction
TensorType CoulombInteractionCpu(CRef<TensorType>& r,
                                  CRef<VectorType>& charge) {
    const auto N = r.rows();

    Eigen::Matrix<data_t, 1, Eigen::Dynamic> ones_col =
        Eigen::Matrix<data_t, 1, Eigen::Dynamic>::Ones(1, N);
    Eigen::Matrix<data_t, Eigen::Dynamic, 1> ones_row =
        Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Ones(N, 1);

    Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> charge_matrix =
        charge.matrix() * charge.transpose().matrix();

    Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2 =
        r.rowwise().squaredNorm().matrix() * ones_col +
        ones_row * r.rowwise().squaredNorm().transpose().matrix() -
        2 * (r.matrix() * r.matrix().transpose());
    dist2.diagonal() = ones_row;

    Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2_array =
        dist2.array();
    Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2_sqrt =
        dist2_array.sqrt();
    Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2_cubed =
        dist2_sqrt * dist2_array;

    TensorType result(N, DIM);
    for (uint8_t i = 0; i < DIM; ++i) {
        result.col(i) =
            ((r.col(i).matrix() * ones_col -
              ones_row * r.col(i).transpose().matrix())
                 .array() /
             dist2_cubed * charge_matrix.array())
                .rowwise()
                .sum();
    }

    return result;
}

#ifdef IONCPP_USE_CUDA
// CUDA 实现：支持持久化上下文，ctx 非空时复用显存（charge 已在 Create 时拷贝）
TensorType CoulombInteractionCudaWithContext(CRef<TensorType>& r,
                                              CRef<VectorType>& charge,
                                              CoulombCudaContext* ctx) {
    const auto N = r.rows();
    TensorType result(N, DIM);
    result.setZero();

    data_t* d_r = nullptr;
    data_t* d_charge = nullptr;
    data_t* d_result = nullptr;
    bool owns_buffers = (ctx == nullptr);

    if (ctx != nullptr && ctx->N == N) {
        d_r = ctx->d_r;
        d_charge = ctx->d_charge;
        d_result = ctx->d_result;
    } else if (ctx == nullptr) {
        cudaError_t err = cudaMalloc(&d_r, N * DIM * sizeof(data_t));
        if (err == cudaSuccess) err = cudaMalloc(&d_charge, N * sizeof(data_t));
        if (err == cudaSuccess)
            err = cudaMalloc(&d_result, N * DIM * sizeof(data_t));
        if (err != cudaSuccess) {
            cudaFree(d_r);
            cudaFree(d_charge);
            cudaFree(d_result);
            return CoulombInteractionCpu(r, charge);
        }
    }

    cudaMemcpy(d_r, r.data(), N * DIM * sizeof(data_t), cudaMemcpyHostToDevice);
    if (owns_buffers) {
        cudaMemcpy(d_charge, charge.data(), N * sizeof(data_t),
                   cudaMemcpyHostToDevice);
    }
    cudaMemset(d_result, 0, N * DIM * sizeof(data_t));

    const int threadsPerBlock = 128;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeCoulombInteraction(d_r, d_charge, d_result, N, blocksPerGrid,
                             threadsPerBlock);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (owns_buffers) {
            cudaFree(d_r);
            cudaFree(d_charge);
            cudaFree(d_result);
        }
        return CoulombInteractionCpu(r, charge);
    }

    cudaMemcpy(result.data(), d_result, N * DIM * sizeof(data_t),
               cudaMemcpyDeviceToHost);

    if (owns_buffers) {
        cudaFree(d_r);
        cudaFree(d_charge);
        cudaFree(d_result);
    }

    return result;
}
#endif  // IONCPP_USE_CUDA

}  // namespace

#ifdef IONCPP_USE_CUDA
// 置于 ioncpp 命名空间（非匿名），供 numerical_integration.cpp 链接
CoulombCudaContext* CoulombCudaContextCreate(size_t N,
                                            CRef<VectorType>& charge) {
    auto* ctx = new CoulombCudaContext;
    ctx->N = N;
    cudaError_t err = cudaMalloc(&ctx->d_r, N * DIM * sizeof(data_t));
    if (err == cudaSuccess) err = cudaMalloc(&ctx->d_charge, N * sizeof(data_t));
    if (err == cudaSuccess)
        err = cudaMalloc(&ctx->d_result, N * DIM * sizeof(data_t));
    if (err != cudaSuccess) {
        cudaFree(ctx->d_r);
        cudaFree(ctx->d_charge);
        cudaFree(ctx->d_result);
        delete ctx;
        return nullptr;
    }
    cudaMemcpy(ctx->d_charge, charge.data(), N * sizeof(data_t),
               cudaMemcpyHostToDevice);
    return ctx;
}

void CoulombCudaContextDestroy(CoulombCudaContext* ctx) {
    if (ctx == nullptr) return;
    cudaFree(ctx->d_r);
    cudaFree(ctx->d_charge);
    cudaFree(ctx->d_result);
    ctx->d_r = nullptr;
    ctx->d_charge = nullptr;
    ctx->d_result = nullptr;
    ctx->N = 0;
    delete ctx;
}
#endif  // IONCPP_USE_CUDA

TensorType CoulombInteraction(CRef<TensorType>& r, CRef<VectorType>& charge,
                              bool use_cuda, void* ctx) {
#ifdef IONCPP_USE_CUDA
    if (use_cuda) {
        return CoulombInteractionCudaWithContext(
            r, charge, static_cast<CoulombCudaContext*>(ctx));
    }
#endif
    (void)ctx;
    return CoulombInteractionCpu(r, charge);
}

}  // namespace ioncpp
