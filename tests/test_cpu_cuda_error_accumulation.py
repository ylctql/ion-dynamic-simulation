"""
CPU 与 CUDA 计算误差积累测试

两质量相同的正负离子在库仑吸引力下绕原点做匀速圆周运动，
通过粒子到圆心距离偏差衡量 r 误差，通过速度与理论值之差衡量 v 误差，
记录各采样时刻的偏差并支持可视化比较。

时间单位与 main.py 一致：time 为 μs。
对标 main.py 模拟 10 μs 的计算量：10 μs ≈ 1108 dt（dt ≈ 9.02 ns @ 35.28 MHz）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 确保 ionsim 可导入
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from setup_path import ensure_build_in_path

ensure_build_in_path(_ROOT)

try:
    import ionsim
except ImportError as e:
    raise ImportError("请先编译 ionsim: cd ism-main && cmake -B build && cmake --build build") from e


def circular_orbit_initial_conditions(R: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    两离子（q1=+1, q2=-1，质量均为 1）在库仑吸引力下绕原点圆周运动的初始条件。

    库仑力 F = q1*q2*(r1-r2)/|r1-r2|^3，对粒子 1：|F1| = 1/(4R^2)
    圆周运动：v^2/R = 1/(4R^2) => v = 1/(2*sqrt(R))

    Parameters
    ----------
    R : float
        每粒子到原点的距离（无量纲）

    Returns
    -------
    r0 : (2, 3) 初始位置
    v0 : (2, 3) 初始速度
    charge : (2,) 电荷 [+1, -1]
    mass : (2,) 质量 [1, 1]
    """
    v = 0.5 / np.sqrt(R)  # 理论圆周速度
    r0 = np.array([[R, 0.0, 0.0], [-R, 0.0, 0.0]], dtype=np.float64, order="F")
    v0 = np.array([[0.0, v, 0.0], [0.0, -v, 0.0]], dtype=np.float64, order="F")
    charge = np.array([1.0, -1.0], dtype=np.float64)
    mass = np.array([1.0, 1.0], dtype=np.float64)
    return r0, v0, charge, mass


def run_trajectory(
    r0: np.ndarray,
    v0: np.ndarray,
    charge: np.ndarray,
    mass: np.ndarray,
    time_end: float,
    n_sample: int,
    use_cuda: bool,
    calc_method: str = "VV",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    运行轨迹并返回采样时刻的 r, v, t。

    Parameters
    ----------
    n_sample : int
        采样点数，总积分步数 = n_sample * 10（保证每段足够精细）
    """
    total_steps = max(n_sample * 10, 100)
    if time_end <= 0:
        raise ValueError("time_end 须为正")

    def zero_force(r, v, t):
        return np.zeros_like(r)

    r_list, v_list = ionsim.calculate_trajectory(
        r0,
        v0,
        charge,
        mass,
        step=total_steps,
        time_start=0.0,
        time_end=time_end,
        force=zero_force,
        use_cuda=use_cuda,
        calc_method=calc_method,
        use_zero_force=True,
    )

    # 均匀采样 n_sample 个点
    indices = np.linspace(0, len(r_list) - 1, n_sample, dtype=int)
    r_sampled = np.array([r_list[i] for i in indices])
    v_sampled = np.array([v_list[i] for i in indices])
    t_sampled = np.linspace(0, time_end, n_sample, endpoint=True)
    return r_sampled, v_sampled, t_sampled


def compute_errors(
    r_sampled: np.ndarray,
    v_sampled: np.ndarray,
    R: float,
    v_theory: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 r 偏差（到圆心距离 - R）和 v 偏差（|v| - v_theory）。

    Returns
    -------
    dr : (n_sample, 2) 两粒子每时刻的 |r|-R
    dv : (n_sample, 2) 两粒子每时刻的 |v|-v_theory
    """
    dist_to_origin = np.linalg.norm(r_sampled, axis=2)  # (n_sample, 2)
    speed = np.linalg.norm(v_sampled, axis=2)  # (n_sample, 2)
    dr = dist_to_origin - R
    dv = speed - v_theory
    return dr, dv


def run_comparison(
    time: float = 10.0,
    R: float = 1.0,
    n_sample: int = 100,
) -> dict:
    """
    在 CPU 和 CUDA 下运行并比较误差。

    Parameters
    ----------
    time : float
        演化总时长（μs）
    R : float
        初始轨道半径
    n_sample : int
        采样点数

    Returns
    -------
    dict with keys: t, dr_cpu, dv_cpu, dr_cuda, dv_cuda, R, v_theory, cuda_available
    """
    r0, v0, charge, mass = circular_orbit_initial_conditions(R)
    v_theory = 0.5 / np.sqrt(R)

    time_dt = time / 10.0 * TIME_EQUIVALENT_10US
    result = {"R": R, "v_theory": v_theory, "time": time}

    # CPU
    r_cpu, v_cpu, t = run_trajectory(
        r0, v0, charge, mass,
        time_end=time_dt,
        n_sample=n_sample,
        use_cuda=False,
    )
    dr_cpu, dv_cpu = compute_errors(r_cpu, v_cpu, R, v_theory)
    result["t"] = t
    result["dr_cpu"] = dr_cpu
    result["dv_cpu"] = dv_cpu

    # CUDA（若可用）
    cuda_available = getattr(ionsim, "cuda_available", False)
    result["cuda_available"] = cuda_available
    if cuda_available:
        r_cuda, v_cuda, _ = run_trajectory(
            r0, v0, charge, mass,
            time_end=time_dt,
            n_sample=n_sample,
            use_cuda=True,
        )
        dr_cuda, dv_cuda = compute_errors(r_cuda, v_cuda, R, v_theory)
        result["dr_cuda"] = dr_cuda
        result["dv_cuda"] = dv_cuda
    else:
        result["dr_cuda"] = None
        result["dv_cuda"] = None

    return result


def plot_comparison(result: dict, out_path: str | None = None) -> None:
    """Visualize CPU vs CUDA error comparison，时间轴为 μs"""
    import matplotlib.pyplot as plt

    t_dt = result["t"]
    time_us = result["time"]
    time_dt = time_us / 10.0 * TIME_EQUIVALENT_10US
    t_us = t_dt * time_us / time_dt
    R = result["R"]
    v_theory = result["v_theory"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # r error: ion 0
    ax = axes[0, 0]
    ax.plot(t_us, result["dr_cpu"][:, 0], label="CPU", alpha=0.8)
    if result["cuda_available"] and result["dr_cuda"] is not None:
        ax.plot(t_us, result["dr_cuda"][:, 0], label="CUDA", alpha=0.8, linestyle="--")
    ax.set_xlabel("time (μs)")
    ax.set_ylabel("|r| - R (ion 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Position error: distance to origin - R")

    # r error: ion 1
    ax = axes[0, 1]
    ax.plot(t_us, result["dr_cpu"][:, 1], label="CPU", alpha=0.8)
    if result["cuda_available"] and result["dr_cuda"] is not None:
        ax.plot(t_us, result["dr_cuda"][:, 1], label="CUDA", alpha=0.8, linestyle="--")
    ax.set_xlabel("time (μs)")
    ax.set_ylabel("|r| - R (ion 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # v error: ion 0
    ax = axes[1, 0]
    ax.plot(t_us, result["dv_cpu"][:, 0], label="CPU", alpha=0.8)
    if result["cuda_available"] and result["dv_cuda"] is not None:
        ax.plot(t_us, result["dv_cuda"][:, 0], label="CUDA", alpha=0.8, linestyle="--")
    ax.set_xlabel("time (μs)")
    ax.set_ylabel("|v| - v_theory (ion 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocity error: |v| - v_theory")

    # v error: ion 1
    ax = axes[1, 1]
    ax.plot(t_us, result["dv_cpu"][:, 1], label="CPU", alpha=0.8)
    if result["cuda_available"] and result["dv_cuda"] is not None:
        ax.plot(t_us, result["dv_cuda"][:, 1], label="CUDA", alpha=0.8, linestyle="--")
    ax.set_xlabel("time (μs)")
    ax.set_ylabel("|v| - v_theory (ion 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"CPU vs CUDA error accumulation (R={R}, v_theory={v_theory:.4f}, time={time_us:.2f} μs)"
    )
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


def test_circular_orbit_conservation():
    """pytest：验证圆周运动初始条件在短时内误差较小"""
    r0, v0, charge, mass = circular_orbit_initial_conditions(R=1.0)
    v_theory = 0.5 / np.sqrt(1.0)

    r_cpu, v_cpu, t = run_trajectory(
        r0, v0, charge, mass,
        time_end=1.0,
        n_sample=50,
        use_cuda=False,
    )
    dr, dv = compute_errors(r_cpu, v_cpu, R=1.0, v_theory=v_theory)

    # 短时内误差应保持较小
    assert np.max(np.abs(dr)) < 0.1, f"r 偏差过大: max|dr|={np.max(np.abs(dr))}"
    assert np.max(np.abs(dv)) < 0.1, f"v 偏差过大: max|dv|={np.max(np.abs(dv))}"


# 对标 main.py 10 μs 的 time（dt ≈ 9.02 ns @ freq_RF=35.28 MHz）
TIME_EQUIVALENT_10US = 1108.35


def main():
    parser = argparse.ArgumentParser(description="CPU/CUDA 计算误差积累测试")
    parser.add_argument(
        "--time",
        type=float,
        default=10.0,
        help="演化时长 (μs)，与 main.py 一致，默认 10",
    )
    parser.add_argument("--R", type=float, default=1.0, help="轨道半径")
    parser.add_argument("--n_sample", type=int, default=100, help="采样点数")
    parser.add_argument("--plot", action="store_true", help="显示可视化")
    parser.add_argument(
        "--out",
        type=str,
        nargs="?",
        default=None,
        const="tests/test_results",
        metavar="DIR",
        help="保存目录，保存为 {时间}us.png；不传参时默认 tests/test_results",
    )
    args = parser.parse_args()

    result = run_comparison(
        time=args.time,
        R=args.R,
        n_sample=args.n_sample,
    )

    time_us = result["time"]
    time_dt = time_us / 10.0 * TIME_EQUIVALENT_10US

    print(f"R={result['R']}, v_theory={result['v_theory']:.6f}, time={time_us:.2f} μs ({time_dt:.1f} dt)")
    print(f"CPU: max|dr|={np.max(np.abs(result['dr_cpu'])):.6e}, max|dv|={np.max(np.abs(result['dv_cpu'])):.6e}")
    if result["cuda_available"]:
        print(f"CUDA: max|dr|={np.max(np.abs(result['dr_cuda'])):.6e}, max|dv|={np.max(np.abs(result['dv_cuda'])):.6e}")
    else:
        print("CUDA 不可用，仅 CPU 结果")

    if args.plot or args.out is not None:
        out_path = None
        if args.out is not None:
            out_dir = Path(args.out)
            if out_dir.suffix.lower() in (".png", ".pdf", ".jpg", ".jpeg"):
                out_dir = out_dir.parent
            out_path = out_dir / f"{time_us:.1f}us.png"
        if out_path and not args.plot:
            import matplotlib
            matplotlib.use("Agg")  # headless when only saving
        plot_comparison(result, out_path=out_path)


if __name__ == "__main__":
    main()
