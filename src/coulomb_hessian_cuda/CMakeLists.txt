cmake_minimum_required(VERSION 3.18)
project(coulomb_hessian_cuda LANGUAGES CXX CUDA)

# 推荐 C++17 + CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# find_package(pybind11 REQUIRED)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../externals/pybind11
    ${CMAKE_CURRENT_BINARY_DIR}/pybind11_build
    )

pybind11_add_module(coulomb_hessian
    binding.cpp
    coulomb_hessian.cu
)

# 根据你的显卡改这里（支持常见卡）
set_target_properties(coulomb_hessian PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# 可选：提高性能
target_compile_options(coulomb_hessian PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas -dlcm=ca>
)