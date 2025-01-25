#ifndef ION_SIMULATION_TYPES_HPP
#define ION_SIMULATION_TYPES_HPP

#include <Eigen/Dense>

namespace ioncpp
{

constexpr std::size_t DIM = 3;

using data_t = double;

using ArrayType = Eigen::Array<data_t, Eigen::Dynamic, 1>;
using VecType = Eigen::Array<data_t, Eigen::Dynamic, DIM>;

template <class T>
using CRef = const Eigen::Ref<const T>;

}

#endif