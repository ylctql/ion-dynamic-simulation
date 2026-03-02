/**
 * pybind11 接口：暴露 calculate_trajectory 给 Python
 * 参考 outline.md 与 ism-hybrid/ionsim.cpp
 */
#include "Coulomb.hpp"
#include "numerical_integration.hpp"
#include "types.hpp"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace ioncpp;

// 将 Python force 回调包装为 ForceCallback
static std::pair<std::vector<TensorType>, std::vector<TensorType>>
calculate_trajectory_impl(
    const Eigen::Ref<const TensorType>& init_r,
    const Eigen::Ref<const TensorType>& init_v,
    const Eigen::Ref<const VectorType>& charge,
    const Eigen::Ref<const VectorType>& mass,
    size_t step,
    data_t time_start,
    data_t time_end,
    py::function force_py,
    bool use_cuda,
    const std::string& calc_method,
    bool use_zero_force) {
  if (time_start >= time_end) {
    throw std::invalid_argument("time_end must be greater than time_start");
  }
  const auto N = init_r.rows();
  if (init_v.rows() != N || charge.rows() != N || mass.rows() != N) {
    throw std::invalid_argument("init_r, init_v, charge, mass must have same N");
  }

  ForceCallback force;
  if (use_zero_force) {
    // 纯 C++ 零力，不调用 Python（用于排查 force 回调/序列化性能）
    force = [N](CRef<TensorType>&, CRef<TensorType>&, data_t) -> TensorType {
      return TensorType::Zero(N, DIM);
    };
  } else {
    force = [force_py = std::move(force_py)](
                CRef<TensorType>& r, CRef<TensorType>& v,
                data_t t) -> TensorType {
      py::buffer result = force_py(r, v, t);
      py::buffer_info info = result.request();

      if (info.format != py::format_descriptor<data_t>::format()) {
        throw std::runtime_error("force must return double array");
      }
      if (info.ndim != 2 || info.shape[1] != static_cast<py::ssize_t>(DIM)) {
        throw std::runtime_error("force must return (N, 3) array");
      }

      using Strides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
      Strides strides(
          info.strides[1] / static_cast<py::ssize_t>(sizeof(data_t)),
          info.strides[0] / static_cast<py::ssize_t>(sizeof(data_t)));

      Eigen::Map<const TensorType, 0, Strides> map(
          static_cast<const data_t*>(info.ptr),
          static_cast<Eigen::Index>(info.shape[0]),
          static_cast<Eigen::Index>(info.shape[1]),
          strides);

      return TensorType(map);
    };
  }

  return CalcTraj(init_r, init_v, charge, mass, step, time_start, time_end,
                  force, use_cuda, calc_method);
}

PYBIND11_MODULE(ionsim, m) {
  m.doc() = "Ion trap dynamics simulation with RK4 and Velocity Verlet";

#ifdef IONCPP_USE_CUDA
  m.attr("cuda_available") = py::cast(true);
#else
  m.attr("cuda_available") = py::cast(false);
#endif

  m.def(
      "calculate_trajectory",
      &calculate_trajectory_impl,
      py::arg("init_r").noconvert(),
      py::arg("init_v").noconvert(),
      py::arg("charge").noconvert(),
      py::arg("mass").noconvert(),
      py::arg("step"),
      py::arg("time_start"),
      py::arg("time_end"),
      py::arg("force"),
      py::arg("use_cuda") = false,
      py::arg("calc_method") = "RK4",
      py::arg("use_zero_force") = false,
      R"doc(
        计算离子轨迹

        Parameters
        ----------
        init_r : ndarray, shape (N, 3)
            初始位置
        init_v : ndarray, shape (N, 3)
            初始速度
        charge : ndarray, shape (N,)
            电荷量
        mass : ndarray, shape (N,)
            质量
        step : int
            积分步数
        time_start : float
            起始时间
        time_end : float
            结束时间
        force : callable
            force(r, v, t) -> (N, 3)，外力（不含库仑力）
        use_cuda : bool
            是否使用 CUDA 加速库仑力
        calc_method : str
            "RK4" 或 "VV"
        use_zero_force : bool
            若 True，不调用 Python force，使用 C++ 零力（排查用）

        Returns
        -------
        r_list : list of ndarray
            每步位置
        v_list : list of ndarray
            每步速度
        )doc");
}
