#!/bin/bash
# 一键构建 ionsim C++ 扩展
# 用法: ./build.sh [--no-cuda] [--clean] [--local DIR]
# 默认尝试 CUDA，使用 --no-cuda 强制仅 CPU
set -e
cd "$(dirname "$0")"

USE_CUDA=ON
CMAKE_ARGS=(-B build -DCMAKE_BUILD_TYPE=Release)
BUILD_ARGS=(--build build)

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cuda)
            USE_CUDA=OFF
            shift
            ;;
        --clean)
            rm -rf build
            shift
            ;;
        --local)
            LOCAL="$(cd "$2" && pwd)"
            shift 2
            if [[ -d "$LOCAL/eigen-4.3.0/Eigen" ]]; then
                CMAKE_ARGS+=(-DEIGEN_LOCAL_PATH="$LOCAL/eigen-4.3.0")
            elif [[ -d "$LOCAL/externals/eigen-4.3.0/Eigen" ]]; then
                CMAKE_ARGS+=(-DEIGEN_LOCAL_PATH="$LOCAL/externals/eigen-4.3.0")
            fi
            if [[ -f "$LOCAL/pybind11/CMakeLists.txt" ]]; then
                CMAKE_ARGS+=(-DPYBIND11_LOCAL_PATH="$LOCAL/pybind11")
            elif [[ -f "$LOCAL/externals/pybind11/CMakeLists.txt" ]]; then
                CMAKE_ARGS+=(-DPYBIND11_LOCAL_PATH="$LOCAL/externals/pybind11")
            fi
            ;;
        *)
            echo "未知选项: $1"
            echo "用法: ./build.sh [--no-cuda] [--clean] [--local DIR]"
            exit 1
            ;;
    esac
done

CMAKE_ARGS+=(-DUSE_CUDA="$USE_CUDA")

echo "配置 CMake..."
cmake "${CMAKE_ARGS[@]}"

echo "编译..."
cmake "${BUILD_ARGS[@]}"

if ls build/ionsim*.so 1>/dev/null 2>&1; then
    echo "构建成功: $(ls build/ionsim*.so)"
else
    echo "构建失败: 未找到 ionsim*.so"
    exit 1
fi
