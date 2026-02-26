#!/bin/bash
# 清理 ism-cpu 项目中的编译记录

echo "正在清理编译记录..."

# 删除 Python 编译文件
echo "1. 删除 __pycache__ 目录..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 删除 .pyc 和 .pyo 文件
echo "2. 删除 .pyc 和 .pyo 文件..."
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# 删除 Python 包元数据
echo "3. 删除 *.egg-info 目录..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# 删除 build 目录
echo "4. 删除 build 目录..."
find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

# 删除 dist 目录
echo "5. 删除 dist 目录..."
find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true

# 删除 CMake 构建文件
echo "6. 删除 CMake 构建文件..."
find . -name "CMakeCache.txt" -delete
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "cmake_install.cmake" -delete
find . -name "Makefile" -path "*/build/*" -delete

# 删除编译的 .so 文件（在 ionsim 目录中）
echo "7. 删除编译的 .so 文件..."
find ./ionsim -name "*.so" -delete 2>/dev/null || true

echo "清理完成！"
