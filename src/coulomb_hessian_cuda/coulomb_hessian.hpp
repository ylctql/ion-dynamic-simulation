#pragma once
#include <cstddef>

// host 函数声明（供 binding.cpp 使用）
void compute_coulomb_hessian_cuda(
    const double* positions,    // length = 3*N
    const double* charges,      // length = N
    size_t N,
    double prefactor,
    double* hessian_out);       // 预分配 (3N)*(3N)，调用前已清零