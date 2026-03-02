#ifndef ION_SIMULATION_NUMERICAL_INTEGRATION_HPP
#define ION_SIMULATION_NUMERICAL_INTEGRATION_HPP

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "types.hpp"

namespace ioncpp {

using ForceCallback =
    std::function<TensorType(CRef<TensorType>& r, CRef<TensorType>& v, data_t t)>;

/**
 * 4 阶 Runge-Kutta 积分
 */
std::pair<std::vector<TensorType>, std::vector<TensorType>> CalcTrajRK(
    CRef<TensorType>& init_r, CRef<TensorType>& init_v,
    CRef<VectorType>& charge, CRef<VectorType>& mass, size_t step,
    data_t time_start, data_t time_end, ForceCallback force, bool use_cuda);

/**
 * Velocity Verlet 积分
 */
std::pair<std::vector<TensorType>, std::vector<TensorType>> CalcTrajVV(
    CRef<TensorType>& init_r, CRef<TensorType>& init_v,
    CRef<VectorType>& charge, CRef<VectorType>& mass, size_t step,
    data_t time_start, data_t time_end, ForceCallback force, bool use_cuda);

/**
 * 统一积分接口，根据 calc_method 选择 RK4 或 VV
 * @param calc_method "RK4" 或 "VV"
 */
std::pair<std::vector<TensorType>, std::vector<TensorType>> CalcTraj(
    CRef<TensorType>& init_r, CRef<TensorType>& init_v,
    CRef<VectorType>& charge, CRef<VectorType>& mass, size_t step,
    data_t time_start, data_t time_end, ForceCallback force, bool use_cuda,
    const std::string& calc_method);

}  // namespace ioncpp

#endif
