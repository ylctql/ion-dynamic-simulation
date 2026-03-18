"""
CPU 与 CUDA 轨迹距离比较测试

在完全相同的初始条件下（与 main.py 一致：多离子、外势场），
分别用 CPU 和 CUDA 计算离子阱中离子运动轨迹，
计算同一离子在 CPU 与 CUDA 下的坐标距离 |r_cpu - r_cuda|，
并随演化时间可视化。

时间指定与 main.py 一致：--time 为 μs，程序内部换算为 dt 单位。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from setup_path import ensure_build_in_path

ensure_build_in_path(_ROOT)

# 抑制日志
logging.getLogger("Interface.cli").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

try:
    import ionsim
except ImportError as e:
    raise ImportError("请先编译 ionsim: cd ism-main && cmake -B build && cmake --build build") from e

# 默认演化时长 10 μs（与 main.py --time 10 一致）
DEFAULT_TIME_US = 10.0
# 每 10 μs 输出一次 Simulation Time 日志（与 main.py 一致）
LOG_INTERVAL_US = 10.0


def _build_force_from_parsed(parsed, root: Path):
    """复用 main 的 force 构建逻辑"""
    from main import _build_force

    return _build_force(
        parsed.field_settings,
        parsed.config,
        np.asarray(parsed.params.q, dtype=np.float64),
        root,
        smooth_axes=parsed.smooth_axes,
        smooth_sg=parsed.smooth_sg,
    )


def run_full_trajectory_iontrap(
    parsed,
    force,
    time_end: float,
    use_cuda: bool,
    n_steps: int | None = None,
    *,
    r0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    log_interval_us: float | None = LOG_INTERVAL_US,
) -> tuple[list, list, np.ndarray]:
    """
    在离子阱势场下运行完整轨迹，返回每步的 r_list, v_list 及时间数组 t。

    若传入 r0, v0 则使用之，否则从 parsed.params 读取（用于保证 CPU/CUDA 初始状态一致）。
    log_interval_us: 每 N μs 输出一次 Simulation Time 日志，与 main.py 一致；None 则不输出。
    """
    p = parsed.params
    cfg = parsed.config
    dt_si = cfg.dt
    if r0 is None:
        r0 = np.asarray(p.get_r0(), dtype=np.float64, order="F")
    else:
        r0 = np.asarray(r0, dtype=np.float64, order="F")
    if v0 is None:
        v0 = np.asarray(p.get_v0(), dtype=np.float64, order="F")
    else:
        v0 = np.asarray(v0, dtype=np.float64, order="F")
    charge = np.asarray(p.q, dtype=np.float64)
    mass = np.asarray(p.m, dtype=np.float64)

    if log_interval_us is None:
        if n_steps is None:
            n_steps = max(int(time_end * 2), 500)
        r_in, v_in, q_in, m_in = _to_ionsim_format(r0, v0, charge, mass)
        r_list, v_list = ionsim.calculate_trajectory(
            r_in, v_in, q_in, m_in,
            step=n_steps,
            time_start=0.0,
            time_end=time_end,
            force=force,
            use_cuda=use_cuda,
            calc_method=p.calc_method,
            use_zero_force=False,
        )
        dt_step = time_end / len(r_list)
        t = (np.arange(len(r_list)) + 1) * dt_step
        return r_list, v_list, t

    chunk_dt = log_interval_us / (dt_si * 1e6)

    all_r, all_v, all_t = [], [], []
    r_cur = np.asarray(r0, dtype=np.float64, order="F")
    v_cur = np.asarray(v0, dtype=np.float64, order="F")
    t_cur = 0.0
    chunk_idx = 0
    device_name = "cuda" if use_cuda else "cpu"
    last_log_time_us = -1.0

    while t_cur < time_end:
        t_end_chunk = min((chunk_idx + 1) * chunk_dt, time_end)
        duration_chunk = t_end_chunk - t_cur
        if duration_chunk <= 0:
            break
        n_chunks_est = max(1, int(np.ceil(time_end / chunk_dt)))
        steps_chunk = (
            max(int(duration_chunk * 2), 100)
            if n_steps is None
            else max(1, n_steps // n_chunks_est)
        )

        r_in, v_in, q_in, m_in = _to_ionsim_format(r_cur, v_cur, charge, mass)
        r_list, v_list = ionsim.calculate_trajectory(
            r_in,
            v_in,
            q_in,
            m_in,
            step=steps_chunk,
            time_start=t_cur,
            time_end=t_end_chunk,
            force=force,
            use_cuda=use_cuda,
            calc_method=p.calc_method,
            use_zero_force=False,
        )

        n_pts = len(r_list)
        dt_step = duration_chunk / max(n_pts - 1, 1)
        t_chunk = t_cur + np.arange(n_pts) * dt_step
        all_r.append(r_list)
        all_v.append(v_list)
        all_t.append(t_chunk)

        time_us = t_end_chunk * dt_si * 1e6
        if time_us - last_log_time_us >= log_interval_us * 0.99:
            logger.info("Simulation Time: %.3f μs (%s)", time_us, device_name)
            last_log_time_us = time_us

        r_cur = np.asarray(r_list[-1], dtype=np.float64, order="F")
        v_cur = np.asarray(v_list[-1], dtype=np.float64, order="F")
        t_cur = t_end_chunk
        chunk_idx += 1

    r_list = list(all_r[0])
    v_list = list(all_v[0])
    t_parts = [all_t[0]]
    for i in range(1, len(all_r)):
        r_list.extend(all_r[i][1:])
        v_list.extend(all_v[i][1:])
        t_parts.append(all_t[i][1:])
    t = np.concatenate(t_parts) if len(t_parts) > 1 else t_parts[0]
    return r_list, v_list, t


def _to_ionsim_format(r, v, charge, mass):
    """与 backend 一致的格式转换"""
    r = np.asarray(r, dtype=np.float64, order="F")
    v = np.asarray(v, dtype=np.float64, order="F")
    charge = np.asarray(charge, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    return r, v, charge, mass


def compute_trajectory_distances(
    r_cpu: list,
    r_cuda: list,
) -> np.ndarray:
    """
    计算每步、每离子的 |r_cpu - r_cuda|。

    Returns
    -------
    dist : (n_steps, n_ions) 各离子 CPU-CUDA 坐标距离
    """
    r_cpu_arr = np.array(r_cpu)
    r_cuda_arr = np.array(r_cuda)
    dist = np.linalg.norm(r_cpu_arr - r_cuda_arr, axis=2)
    return dist


def run_with_fixed_initial_conditions(
    parsed,
    force,
    duration_dt: float,
    *,
    use_cuda: bool,
    n_steps: int | None,
    r0: np.ndarray,
    v0: np.ndarray,
) -> tuple[list, list, np.ndarray]:
    """使用固定初始条件运行一次轨迹演化。"""
    return run_full_trajectory_iontrap(
        parsed,
        force,
        duration_dt,
        use_cuda=use_cuda,
        n_steps=n_steps,
        # 传入副本，避免底层实现原地修改影响后续比较
        r0=np.array(r0, dtype=np.float64, order="F", copy=True),
        v0=np.array(v0, dtype=np.float64, order="F", copy=True),
    )


def run_comparison(
    parsed,
    force,
    duration_dt: float,
    n_steps: int | None = None,
    enable_self_compare: bool = False,
) -> dict:
    """
    在相同初始条件下分别运行 CPU 和 CUDA，返回轨迹距离数据。

    显式取一次 r0, v0 并传入两次调用，确保 CPU 与 CUDA 演化初始状态完全相同
    （否则随机初始化时每次 get_r0() 会生成不同随机数）。
    """
    if not getattr(ionsim, "cuda_available", False):
        raise RuntimeError("CUDA 不可用，无法比较 CPU-CUDA 轨迹距离")

    # 仅取一次初始状态，保证 CPU 与 CUDA 完全一致
    r0 = np.asarray(parsed.params.get_r0(), dtype=np.float64, order="F")
    v0 = np.asarray(parsed.params.get_v0(), dtype=np.float64, order="F")

    r_cpu_1, v_cpu_1, t = run_with_fixed_initial_conditions(
        parsed,
        force,
        duration_dt,
        use_cuda=False,
        n_steps=n_steps,
        r0=r0,
        v0=v0,
    )
    r_cuda_1, v_cuda_1, _ = run_with_fixed_initial_conditions(
        parsed,
        force,
        duration_dt,
        use_cuda=True,
        n_steps=n_steps,
        r0=r0,
        v0=v0,
    )
    dist_cpu_cuda = compute_trajectory_distances(r_cpu_1, r_cuda_1)
    dist_cpu_cpu = None
    dist_cuda_cuda = None
    v_cpu_2 = None
    v_cuda_2 = None

    if enable_self_compare:
        r_cpu_2, v_cpu_2, _ = run_with_fixed_initial_conditions(
            parsed,
            force,
            duration_dt,
            use_cuda=False,
            n_steps=n_steps,
            r0=r0,
            v0=v0,
        )
        r_cuda_2, v_cuda_2, _ = run_with_fixed_initial_conditions(
            parsed,
            force,
            duration_dt,
            use_cuda=True,
            n_steps=n_steps,
            r0=r0,
            v0=v0,
        )
        dist_cpu_cpu = compute_trajectory_distances(r_cpu_1, r_cpu_2)
        dist_cuda_cuda = compute_trajectory_distances(r_cuda_1, r_cuda_2)

    return {
        "t": t,
        "dist": dist_cpu_cuda,
        "dist_cpu_cuda": dist_cpu_cuda,
        "dist_cpu_cpu": dist_cpu_cpu,
        "dist_cuda_cuda": dist_cuda_cuda,
        "n_ions": dist_cpu_cuda.shape[1],
        "duration": duration_dt,
        "v_cpu_1": np.array(v_cpu_1),
        "v_cpu_2": np.array(v_cpu_2),
        "v_cuda_1": np.array(v_cuda_1),
        "v_cuda_2": np.array(v_cuda_2),
    }


def plot_distance_vs_time(
    result: dict,
    out_path: str | None = None,
    dt: float | None = None,
) -> None:
    """可视化各离子 CPU-CUDA 轨迹距离随时间的演化，时间轴为 μs"""
    import matplotlib.pyplot as plt

    t_dt = result["t"]
    t_us = t_dt * dt * 1e6 if dt is not None else t_dt
    dist = result["dist"]
    n_ions = result["n_ions"]
    duration_us = result["duration"] * dt * 1e6 if dt is not None else result["duration"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 各离子距离曲线（离子多时只画前 10 个 + 最大）
    ax = axes[0]
    n_show = min(10, n_ions)
    for i in range(n_show):
        ax.plot(t_us, dist[:, i], label=f"ion {i}", alpha=0.7)
    if n_ions > n_show:
        dist_max_over_ions = np.max(dist, axis=1)
        ax.plot(t_us, dist_max_over_ions, "k--", alpha=0.8, label="max over all")
    ax.set_xlabel("time (μs)")
    ax.set_ylabel("|r_cpu - r_cuda| (μm)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("CPU-CUDA trajectory distance per ion")

    # 最大距离（所有离子），同时展示跨设备与同设备对比
    ax = axes[1]
    dist_max = np.max(result["dist_cpu_cuda"], axis=1) if "dist_cpu_cuda" in result else np.max(dist, axis=1)
    ax.plot(t_us, dist_max, "k-", alpha=0.8, label="cpu-cuda")
    if "dist_cpu_cpu" in result and result["dist_cpu_cpu"] is not None:
        ax.plot(t_us, np.max(result["dist_cpu_cpu"], axis=1), alpha=0.8, label="cpu-cpu")
    if "dist_cuda_cuda" in result and result["dist_cuda_cuda"] is not None:
        ax.plot(t_us, np.max(result["dist_cuda_cuda"], axis=1), alpha=0.8, label="cuda-cuda")
    ax.set_xlabel("time (μs)")
    ax.set_ylabel("max_i distance (μm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Max distance over all ions")

    fig.suptitle(
        f"CPU vs CUDA trajectory distance in ion trap (N={n_ions}, duration={duration_us:.2f} μs)"
    )
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


def test_cpu_cuda_trajectory_comparison_runs():
    """pytest：若 CUDA 可用且 csv 存在，能完成 CPU-CUDA 轨迹比较"""
    from Interface import cli

    parser = cli.create_parser()
    args = parser.parse_args(
        ["--N", "10", "--time", "10", "--csv", "data/monolithic20241118.csv"]
    )
    try:
        parsed = cli.parse_and_build(args, _ROOT)
    except FileNotFoundError:
        return  # skip when csv/config missing

    if not getattr(ionsim, "cuda_available", False):
        return

    force = _build_force_from_parsed(parsed, _ROOT)
    result = run_comparison(parsed, force, duration_dt=parsed.params.duration, n_steps=100)
    assert result["dist"].shape[0] > 0
    assert result["dist"].shape[1] == 10
    assert result["dist_cpu_cpu"] is None
    assert result["dist_cuda_cuda"] is None


def test_cpu_cuda_trajectory_self_comparison_runs():
    """pytest：开启自比较时，CPU-CPU 与 CUDA-CUDA 结果可用"""
    from Interface import cli

    parser = cli.create_parser()
    args = parser.parse_args(
        ["--N", "10", "--time", "10", "--csv", "data/monolithic20241118.csv"]
    )
    try:
        parsed = cli.parse_and_build(args, _ROOT)
    except FileNotFoundError:
        return  # skip when csv/config missing

    if not getattr(ionsim, "cuda_available", False):
        return

    force = _build_force_from_parsed(parsed, _ROOT)
    result = run_comparison(
        parsed,
        force,
        duration_dt=parsed.params.duration,
        n_steps=100,
        enable_self_compare=True,
    )
    assert result["dist_cpu_cpu"].shape == result["dist"].shape
    assert result["dist_cuda_cuda"].shape == result["dist"].shape


def main():
    from Interface import cli

    parser = argparse.ArgumentParser(
        description="比较 CPU 与 CUDA 下离子阱中同一离子轨迹坐标距离"
    )
    parser.add_argument("--N", type=int, default=50, help="离子数")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/monolithic20241118.csv",
        help="电场 CSV 路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="电压配置 JSON（如 circle.json），不传则用默认",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=DEFAULT_TIME_US,
        help="模拟终止时刻 (μs)，与 main.py 一致，默认 10",
    )
    parser.add_argument("--n_steps", type=int, default=None, help="积分步数，默认自动")
    parser.add_argument("--alpha", type=float, default=0.0, help="同位素参杂比例；单同位素模式下为该同位素丰度")
    parser.add_argument(
        "--isotope",
        type=str,
        default=None,
        choices=["Ba133", "Ba134", "Ba135", "Ba136", "Ba137", "Ba138"],
        help="单同位素模式：指定同位素种类，alpha 为该同位素丰度，其余为 Ba135；不指定则使用混合模式",
    )
    parser.add_argument("--plot", action="store_true", help="显示图形")
    parser.add_argument(
        "--self_compare",
        action="store_true",
        help="开启同设备自比较（CPU-CPU 与 CUDA-CUDA）；默认关闭，仅比较 CPU-CUDA",
    )
    parser.add_argument(
        "--out",
        type=str,
        nargs="?",
        default=None,
        const="tests/test_results/traj_dist",
        metavar="DIR",
        help="保存目录，保存为 N{N}_{时间}.png（指定同位素时加 _isotope 后缀）；不传参时默认 tests/test_results/traj_dist",
    )
    args = parser.parse_args()

    # 构建与 main.py 一致的 args，--time 为 μs，parse_and_build 内部换算为 dt
    main_arg_list = [
        "--N", str(args.N),
        "--time", str(args.time),
        "--csv", args.csv,
    ]
    if args.config:
        main_arg_list += ["--config", args.config]
    if args.alpha != 0.0:
        main_arg_list += ["--alpha", str(args.alpha)]
    if args.isotope is not None:
        main_arg_list += ["--isotope", args.isotope]
    main_args = cli.create_parser().parse_args(main_arg_list)

    try:
        parsed = cli.parse_and_build(main_args, _ROOT)
    except FileNotFoundError as e:
        print(f"配置或数据文件不存在: {e}")
        sys.exit(1)

    if not getattr(ionsim, "cuda_available", False):
        print("CUDA 不可用，无法比较 CPU-CUDA 轨迹距离")
        sys.exit(1)

    force = _build_force_from_parsed(parsed, _ROOT)

    # parsed.params.duration 已由 --time (μs) 换算为 dt 单位
    result = run_comparison(
        parsed, force,
        duration_dt=parsed.params.duration,
        n_steps=args.n_steps,
        enable_self_compare=args.self_compare,
    )

    # 无量纲距离 → μm：dist_um = dist * dl * 1e6
    scale = parsed.config.dl * 1e6
    result["dist"] = result["dist"] * scale
    result["dist_cpu_cuda"] = result["dist_cpu_cuda"] * scale
    if result["dist_cpu_cpu"] is not None:
        result["dist_cpu_cpu"] = result["dist_cpu_cpu"] * scale
    if result["dist_cuda_cuda"] is not None:
        result["dist_cuda_cuda"] = result["dist_cuda_cuda"] * scale

    max_dist = np.max(result["dist_cpu_cuda"])
    time_us = result["duration"] * parsed.config.dt * 1e6
    print(f"N_ions: {result['n_ions']}, time: {time_us:.2f} μs ({result['duration']:.1f} dt)")
    print(f"Max |r_cpu1 - r_cuda1|: {max_dist:.6e} μm")
    n_show = min(5, result["n_ions"])
    if args.self_compare:
        max_dist_cpu_cpu = np.max(result["dist_cpu_cpu"])
        max_dist_cuda_cuda = np.max(result["dist_cuda_cuda"])
        print(f"Max |r_cpu1 - r_cpu2|: {max_dist_cpu_cpu:.6e} μm")
        print(f"Max |r_cuda1 - r_cuda2|: {max_dist_cuda_cuda:.6e} μm")
        for i in range(n_show):
            print(
                f"  ion {i}: cpu-cuda={np.max(result['dist_cpu_cuda'][:, i]):.6e} μm, "
                f"cpu-cpu={np.max(result['dist_cpu_cpu'][:, i]):.6e} μm, "
                f"cuda-cuda={np.max(result['dist_cuda_cuda'][:, i]):.6e} μm"
            )
    else:
        for i in range(n_show):
            print(f"  ion {i}: cpu-cuda={np.max(result['dist_cpu_cuda'][:, i]):.6e} μm")

    if args.plot or args.out is not None:
        out_path = None
        if args.out is not None:
            out_dir = Path(args.out)
            if out_dir.suffix.lower() in (".png", ".pdf", ".jpg", ".jpeg"):
                out_dir = out_dir.parent
            use_isotope = args.alpha > 0 or args.isotope is not None
            suffix = "_isotope" if use_isotope else ""
            out_path = out_dir / f"N{result['n_ions']}_{time_us:.1f}{suffix}.png"
        if out_path and not args.plot:
            import matplotlib
            matplotlib.use("Agg")
        plot_distance_vs_time(
            result,
            out_path=str(out_path) if out_path is not None else None,
            dt=parsed.config.dt,
        )


if __name__ == "__main__":
    main()
