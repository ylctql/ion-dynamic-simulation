#ifndef ION_SIMULATION_COULOMB_HPP
#define ION_SIMULATION_COULOMB_HPP

#include <cstddef>

#include "types.hpp"

namespace ioncpp {

#ifdef IONCPP_USE_CUDA
/**
 * CUDA 显存持久化上下文：预分配 d_r, d_charge, d_result，供积分循环复用
 */
struct CoulombCudaContext {
    data_t* d_r = nullptr;
    data_t* d_charge = nullptr;
    data_t* d_result = nullptr;
    size_t N = 0;
};

/** 创建上下文并拷贝 charge 到 GPU（charge 在积分中不变） */
CoulombCudaContext* CoulombCudaContextCreate(size_t N,
                                            CRef<VectorType>& charge);
/** 释放上下文 */
void CoulombCudaContextDestroy(CoulombCudaContext* ctx);
#endif

/**
 * 库仑力计算
 * @param r 位置 (N, 3)
 * @param charge 电荷 (N,)
 * @param use_cuda true 使用 CUDA，false 使用 CPU
 * @param ctx 可选，CUDA 持久化上下文；非空时复用显存，避免每步 malloc/free
 */
TensorType CoulombInteraction(CRef<TensorType>& r, CRef<VectorType>& charge,
                              bool use_cuda, void* ctx = nullptr);

}  // namespace ioncpp

#endif
