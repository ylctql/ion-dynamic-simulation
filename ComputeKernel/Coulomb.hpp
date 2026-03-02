#ifndef ION_SIMULATION_COULOMB_HPP
#define ION_SIMULATION_COULOMB_HPP

#include "types.hpp"

namespace ioncpp {

/**
 * 库仑力计算
 * @param r 位置 (N, 3)
 * @param charge 电荷 (N,)
 * @param use_cuda true 使用 CUDA，false 使用 CPU
 */
TensorType CoulombInteraction(CRef<TensorType>& r, CRef<VectorType>& charge,
                              bool use_cuda);

}  // namespace ioncpp

#endif
