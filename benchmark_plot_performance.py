#!/usr/bin/env python3
"""
性能测试脚本：测试不同离子数下，使用 plot 与不使用 plot 时，
动力学模拟程序每计算 10 μs 所需的时间。

每组实验跑 100 μs，取平均值（即总时间/10 作为每 10 μs 的耗时）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 项目根目录
_ROOT = Path(__file__).resolve().parent

# ============ 可配置参数 ============
# 要测试的离子数列表（可修改；快速测试可用 [10, 30, 50]）
ION_COUNTS = [10, 20, 30, 50, 80, 100, 150, 200]

# 每组实验模拟时长 (μs)
SIM_DURATION_US = 100

# 每 10 μs 的基准（用于计算平均值）
INTERVAL_10US = 10

# 主程序路径
MAIN_SCRIPT = _ROOT / "main.py"

# 结果输出目录
BENCHMARK_RESULTS_DIR = _ROOT / "benchmark_results"
OUTPUT_IMAGE = BENCHMARK_RESULTS_DIR / "benchmark_plot_performance.png"
OUTPUT_CSV = BENCHMARK_RESULTS_DIR / "benchmark_plot_performance.csv"


def run_simulation(n_ions: int, use_plot: bool) -> float:
    """
    运行一次模拟，返回总耗时（秒）。

    Parameters
    ----------
    n_ions : int
        离子数
    use_plot : bool
        是否启用 plot 功能

    Returns
    -------
    float
        模拟 100 μs 的总耗时（秒）
    """
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--N", str(n_ions),
        "--time", str(SIM_DURATION_US),
    ]
    if use_plot:
        cmd.append("--plot")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # 无头模式，避免弹窗
    env["ISM_LOG_LEVEL"] = "WARNING"  # 减少日志输出

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 小时超时
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"模拟失败 (N={n_ions}, plot={use_plot}):\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )

    return elapsed


def time_per_10us(total_seconds: float) -> float:
    """将总耗时转换为每 10 μs 的耗时（秒）。"""
    return total_seconds / (SIM_DURATION_US / INTERVAL_10US)


def main():
    print("=" * 60)
    print("动力学模拟性能测试：plot 功能影响")
    print("=" * 60)
    print(f"离子数列表: {ION_COUNTS}")
    print(f"每组模拟时长: {SIM_DURATION_US} μs")
    print(f"每 10 μs 取平均")
    print()

    times_with_plot: list[float] = []
    times_without_plot: list[float] = []

    for n in ION_COUNTS:
        print(f"测试 N={n} ...", end=" ", flush=True)

        # 无 plot
        try:
            t_no_plot = run_simulation(n, use_plot=False)
            t_per_10_no = time_per_10us(t_no_plot)
            times_without_plot.append(t_per_10_no)
            print(f"无 plot: {t_per_10_no:.3f}s/10μs", end="  ", flush=True)
        except Exception as e:
            print(f"无 plot 失败: {e}")
            times_without_plot.append(np.nan)

        # 有 plot
        try:
            t_plot = run_simulation(n, use_plot=True)
            t_per_10_plot = time_per_10us(t_plot)
            times_with_plot.append(t_per_10_plot)
            print(f"有 plot: {t_per_10_plot:.3f}s/10μs", flush=True)
        except Exception as e:
            print(f"有 plot 失败: {e}")
            times_with_plot.append(np.nan)

    # 保存 CSV
    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("离子数,无plot耗时(s/10μs),有plot耗时(s/10μs)\n")
        for n, t_no, t_yes in zip(ION_COUNTS, times_without_plot, times_with_plot):
            f.write(f"{n},{t_no:.6f},{t_yes:.6f}\n")
    print(f"\n结果已保存到: {OUTPUT_CSV}")

    # 绘制折线图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array(ION_COUNTS)

    ax.plot(
        x,
        times_without_plot,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="Without plot",
        color="#2ecc71",
    )
    ax.plot(
        x,
        times_with_plot,
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="With plot",
        color="#e74c3c",
    )

    ax.set_xlabel("Number of ions N", fontsize=12)
    ax.set_ylabel("Time per 10 μs (s)", fontsize=12)
    ax.set_title(
        f"Dynamics Simulation Performance: Plot vs No Plot\n"
        f"(Average over {SIM_DURATION_US} μs per run)",
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
