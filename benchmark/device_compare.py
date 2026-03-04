#!/usr/bin/env python3
"""
CPU vs CUDA 对照实验：在特定离子数下比较 CPU 与 CUDA 的运行时间。

统一不启用 plot，每组实验跑 100 μs，取平均值（即总时间/10 作为每 10 μs 的耗时）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 path 中，支持 python benchmark/device_compare.py 直接运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from benchmark.common import (
    BENCHMARK_RESULTS_DIR,
    SIM_DURATION_US,
    run_simulation,
    time_per_10us,
)

# ============ 可配置参数 ============
ION_COUNTS = [10, 30, 50, 80, 100, 200, 300, 400, 500, 600, 700]

OUTPUT_IMAGE = BENCHMARK_RESULTS_DIR / "benchmark_device_compare.png"
OUTPUT_CSV = BENCHMARK_RESULTS_DIR / "benchmark_device_compare.csv"


def main():
    print("=" * 60)
    print("CPU vs CUDA 对照实验")
    print("=" * 60)
    print(f"离子数列表: {ION_COUNTS}")
    print(f"每组模拟时长: {SIM_DURATION_US} μs (no plot)")
    print()

    times_cpu: list[float] = []
    times_cuda: list[float] = []

    for n in ION_COUNTS:
        print(f"测试 N={n} ...", end=" ", flush=True)

        # CPU
        try:
            t_cpu = run_simulation(n, use_plot=False, device="cpu")
            t_per_10_cpu = time_per_10us(t_cpu)
            times_cpu.append(t_per_10_cpu)
            print(f"CPU: {t_per_10_cpu:.3f}s/10μs", end="  ", flush=True)
        except Exception as e:
            print(f"CPU 失败: {e}")
            times_cpu.append(np.nan)

        # CUDA
        try:
            t_cuda = run_simulation(n, use_plot=False, device="cuda")
            t_per_10_cuda = time_per_10us(t_cuda)
            times_cuda.append(t_per_10_cuda)
            print(f"CUDA: {t_per_10_cuda:.3f}s/10μs", flush=True)
        except Exception as e:
            print(f"CUDA 失败: {e}")
            times_cuda.append(np.nan)

    # 保存 CSV
    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("离子数,CPU耗时(s/10μs),CUDA耗时(s/10μs)\n")
        for n, t_cpu, t_cuda in zip(ION_COUNTS, times_cpu, times_cuda):
            f.write(f"{n},{t_cpu:.6f},{t_cuda:.6f}\n")
    print(f"\n结果已保存到: {OUTPUT_CSV}")

    # 绘制折线图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array(ION_COUNTS)

    ax.plot(
        x,
        times_cpu,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="CPU",
        color="#3498db",
    )
    ax.plot(
        x,
        times_cuda,
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="CUDA",
        color="#9b59b6",
    )

    ax.set_xlabel("Number of ions N", fontsize=12)
    ax.set_ylabel("Time per 10 μs (s)", fontsize=12)
    ax.set_title(
        f"Dynamics Simulation Performance: CPU vs CUDA\n"
        f"(Average over {SIM_DURATION_US} μs per run, no plot)",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"折线图已保存到: {OUTPUT_IMAGE}")
    print("\n测试完成。")


if __name__ == "__main__":
    main()
