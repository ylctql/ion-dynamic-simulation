"""
Benchmark 公共模块：运行模拟、时间换算等共享逻辑。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# 项目根目录（benchmark 的父目录）
_ROOT = Path(__file__).resolve().parent.parent

# 每组实验模拟时长 (μs)
SIM_DURATION_US = 100

# 每 10 μs 的基准（用于计算平均值）
INTERVAL_10US = 10

# 主程序路径
MAIN_SCRIPT = _ROOT / "main.py"

# 结果输出目录（位于 benchmark 目录下）
BENCHMARK_RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"


def run_simulation(
    n_ions: int,
    use_plot: bool,
    device: str = "cpu",
) -> float:
    """
    运行一次模拟，返回总耗时（秒）。

    Parameters
    ----------
    n_ions : int
        离子数
    use_plot : bool
        是否启用 plot 功能
    device : str
        计算设备: cpu 或 cuda

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
        "--device", device,
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
            f"模拟失败 (N={n_ions}, plot={use_plot}, device={device}):\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )

    return elapsed


def time_per_10us(total_seconds: float) -> float:
    """将总耗时转换为每 10 μs 的耗时（秒）。"""
    return total_seconds / (SIM_DURATION_US / INTERVAL_10US)
