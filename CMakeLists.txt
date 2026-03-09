cmake_minimum_required(VERSION 3.18)
project(ism_sim LANGUAGES CXX)

# 可选 CUDA 加速
option(USE_CUDA "Enable CUDA acceleration for Coulomb force" OFF)
# 排查用：禁用库仑力以测试是否为库仑计算导致变慢
option(SKIP_COULOMB_FOR_TEST "Skip Coulomb force for performance debugging" OFF)
if(USE_CUDA)
  # 显式设置 CUDA 架构，避免 "Failed to detect" 或 "must be non-empty" 错误
  if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    add_compile_definitions(IONCPP_USE_CUDA)
  else()
    message(WARNING "USE_CUDA=ON but CUDAToolkit not found, building CPU-only")
    set(USE_CUDA OFF)
  endif()
endif()

# 编译选项
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

include(FetchContent)

# Eigen：支持三种方式
# 1. EIGEN_LOCAL_PATH - 本地路径（如 ../ism-hybrid/externals/eigen-4.3.0）
# 2. USE_SYSTEM_EIGEN - 系统安装的 Eigen
# 3. 默认 - FetchContent 自动下载
set(EIGEN_LOCAL_PATH "" CACHE PATH "Path to local Eigen (skip FetchContent)")
option(USE_SYSTEM_EIGEN "Use system Eigen (find_package)" OFF)

if(EIGEN_LOCAL_PATH AND EXISTS "${EIGEN_LOCAL_PATH}/Eigen")
  message(STATUS "Using local Eigen: ${EIGEN_LOCAL_PATH}")
  add_library(EigenLocal INTERFACE)
  target_include_directories(EigenLocal INTERFACE ${EIGEN_LOCAL_PATH})
  add_library(Eigen3::Eigen ALIAS EigenLocal)
elseif(USE_SYSTEM_EIGEN)
  find_package(Eigen3 3.4 QUIET)
  if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "USE_SYSTEM_EIGEN=ON but Eigen3 not found")
  endif()
else()
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen)
endif()

# pybind11：支持本地路径或 FetchContent
set(PYBIND11_LOCAL_PATH "" CACHE PATH "Path to local pybind11 (e.g. ../ism-hybrid/externals/pybind11)")
if(PYBIND11_LOCAL_PATH AND EXISTS "${PYBIND11_LOCAL_PATH}/CMakeLists.txt")
  message(STATUS "Using local pybind11: ${PYBIND11_LOCAL_PATH}")
  add_subdirectory(${PYBIND11_LOCAL_PATH} ${CMAKE_BINARY_DIR}/pybind11-build)
else()
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# 核心库 ioncpp
add_library(ioncpp)
if(SKIP_COULOMB_FOR_TEST)
  target_compile_definitions(ioncpp PRIVATE IONCPP_SKIP_COULOMB)
  message(STATUS "SKIP_COULOMB_FOR_TEST=ON: 库仑力已禁用（仅用于性能排查）")
endif()
target_compile_features(ioncpp PRIVATE cxx_std_17)
set_target_properties(ioncpp PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(ioncpp PUBLIC Eigen3::Eigen)

add_subdirectory(ComputeKernel)

# pybind11 模块 ionsim：将 ioncpp 源文件直接编译进 ionsim，避免链接时遗漏 CoulombCudaContext* 等符号
set(ionsim_sources
  ComputeKernel/ionsim.cpp
  ComputeKernel/Coulomb.cpp
  ComputeKernel/numerical_integration.cpp
)
if(USE_CUDA)
  list(APPEND ionsim_sources ComputeKernel/Coulomb_cuda.cu)
endif()
pybind11_add_module(ionsim ${ionsim_sources})
target_compile_features(ionsim PRIVATE cxx_std_17)
target_include_directories(ionsim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ComputeKernel)
target_link_libraries(ionsim PRIVATE Eigen3::Eigen)
if(USE_CUDA)
  target_compile_definitions(ionsim PRIVATE IONCPP_USE_CUDA)
  target_link_libraries(ionsim PRIVATE CUDA::cudart)
endif()

# 输出到 build 目录，便于 Python 直接 import
set_target_properties(ionsim PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
