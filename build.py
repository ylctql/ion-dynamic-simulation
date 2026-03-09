#!/usr/bin/env python3
"""
一键构建脚本：配置并编译 ionsim C++ 扩展
默认尝试启用 CUDA，若不可用则回退为 CPU-only。构建产物支持运行时通过 --device 选择 cpu/cuda。
用法: python build.py [选项]
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
BUILD_DIR = _ROOT / "build"


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    """执行命令，失败时返回 False"""
    cwd = cwd or _ROOT
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    return r.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="一键构建 ionsim C++ 扩展",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build.py              # 默认尝试 CUDA；若存在 externals 则优先使用本地依赖
  python build.py --no-cuda   # 强制仅 CPU 构建
  python build.py --local DIR # 指定 externals 目录（覆盖自动检测）

构建后通过 main.py 的 --device cpu 或 --device cuda 选择计算设备。
        """,
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="强制仅 CPU 构建（不尝试 CUDA）",
    )
    parser.add_argument(
        "--local",
        type=str,
        metavar="DIR",
        help="指定 externals 目录（覆盖自动检测，用于 EIGEN_LOCAL_PATH 和 PYBIND11_LOCAL_PATH）",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="构建前清理 build 目录",
    )
    parser.add_argument(
        "-j",
        type=int,
        default=0,
        metavar="N",
        help="并行编译 jobs 数，0 表示自动",
    )
    args = parser.parse_args()

    # 清理
    if args.clean and BUILD_DIR.exists():
        print(f"清理 {BUILD_DIR}...")
        shutil.rmtree(BUILD_DIR)

    # 确定本地依赖目录：--local 指定 > 项目根 externals 目录
    local_base: Path | None = None
    if args.local:
        local_base = Path(args.local).resolve()
    elif (_ROOT / "externals").is_dir():
        local_base = _ROOT / "externals"
        print(f"检测到 externals 目录，优先使用本地 pybind11 和 Eigen")

    # CMake 配置
    def make_cmake_args(use_cuda: bool) -> list[str]:
        args_list = [
            "cmake",
            "-B",
            str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DUSE_CUDA=ON" if use_cuda else "-DUSE_CUDA=OFF",
        ]
        if local_base:
            for base in [local_base, local_base / "externals"]:
                eigen = base / "eigen-4.3.0"
                if not eigen.exists():
                    eigen = base / "eigen"
                if (eigen / "Eigen").exists():
                    args_list.append(f"-DEIGEN_LOCAL_PATH={eigen}")
                    break
            for base in [local_base, local_base / "externals"]:
                pybind = base / "pybind11"
                if (pybind / "CMakeLists.txt").exists():
                    args_list.append(f"-DPYBIND11_LOCAL_PATH={pybind}")
                    break
        return args_list

    use_cuda = not args.no_cuda
    cmake_args = make_cmake_args(use_cuda)

    print("配置 CMake...")
    if not run(cmake_args):
        if use_cuda:
            print("CUDA 构建失败，回退为 CPU-only 构建...")
            # 回退时需清理 build 目录，避免残留 CUDA 配置导致链接错误
            if BUILD_DIR.exists():
                print(f"清理 {BUILD_DIR} 后重新配置...")
                shutil.rmtree(BUILD_DIR)
            cmake_args = make_cmake_args(False)
            if not run(cmake_args):
                return 1
        else:
            return 1

    # 编译
    build_args = ["cmake", "--build", str(BUILD_DIR)]
    if args.j > 0:
        build_args.extend(["-j", str(args.j)])

    print("编译...")
    if not run(build_args):
        return 1

    so_files = list(BUILD_DIR.glob("ionsim*.so"))
    if so_files:
        print(f"\n构建成功: {so_files[0].name}")
        return 0
    print("\n构建失败: 未找到 ionsim*.so")
    return 1


if __name__ == "__main__":
    sys.exit(main())
