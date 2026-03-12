"""
端到端：从初始条件运行动力学模拟，计算平衡位置并保存

用户指定初始位置速度（npz 格式，可选）、延时（μs）、演化时间（μs），
输出各离子平衡位置（μm，npz 格式）到 TrajectoryAnalysis/equi_pos/。

大 N 模拟（N≥1000）：轨迹内存 ≈ 48 × 步数 × N 字节。100 μs 时约 1 GB (N=1000)
或 10 GB (N=10000)。超过 --memory-limit 时自动分块计算，避免 OOM。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 项目根目录，用于 resolve 路径
_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

# 输出目录，相对于 TrajectoryAnalysis 包
EQUI_POS_DIR = Path(__file__).resolve().parent / "equi_pos"

# 轨迹内存：每步存 r(N,3) + v(N,3)，各 8 字节/double
_BYTES_PER_STEP_PER_ION = 2 * 3 * 8  # 48 bytes

# 内存阈值：超过则警告；超过此值的 2 倍则自动分块
_MEMORY_WARN_MB = 500
_MEMORY_CHUNK_MB = 1000


def _format_bytes(b: float) -> str:
    """将字节数格式化为人类可读字符串"""
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def _estimate_trajectory_memory(n_steps: int, n_ions: int) -> int:
    """估算轨迹 r_list + v_list 占用的内存（字节）"""
    return n_steps * n_ions * _BYTES_PER_STEP_PER_ION


def _run_chunked(
    *,
    ionsim,
    r0: np.ndarray,
    v0: np.ndarray,
    charge: np.ndarray,
    mass: np.ndarray,
    force,
    t0_dt: float,
    duration_dt: float,
    n_ions: int,
    skip_initial_dt: float | None,
    device: str,
    calc_method: str,
    limit_mb: float,
) -> np.ndarray:
    """
    分块运行轨迹，用递推均值计算平衡位置，避免一次性加载全部轨迹。
    """
    target_bytes_per_chunk = limit_mb * 0.5 * 1024 * 1024  # 每块约 limit/2 MB
    steps_per_chunk = max(
        100,
        int(target_bytes_per_chunk / (n_ions * _BYTES_PER_STEP_PER_ION)),
    )
    n_steps_total = max(int(duration_dt * 2), 500)
    n_chunks = max(1, (n_steps_total + steps_per_chunk - 1) // steps_per_chunk)
    chunk_duration = duration_dt / n_chunks
    steps_per_chunk = max(100, int(chunk_duration * 2))

    r_sum = np.zeros((n_ions, 3), dtype=np.float64)
    n_valid = 0
    r_cur = np.asarray(r0, dtype=np.float64, order="F")
    v_cur = np.asarray(v0, dtype=np.float64, order="F")

    for i in range(n_chunks):
        t_start = t0_dt + i * chunk_duration
        t_end = t0_dt + min((i + 1) * chunk_duration, duration_dt)
        if t_end <= t_start:
            break
        steps_this = max(10, int((t_end - t_start) * 2))

        r_in = np.asarray(r_cur, dtype=np.float64, order="F")
        v_in = np.asarray(v_cur, dtype=np.float64, order="F")
        r_list, v_list = ionsim.calculate_trajectory(
            r_in,
            v_in,
            charge,
            mass,
            step=steps_this,
            time_start=t_start,
            time_end=t_end,
            force=force,
            use_cuda=(device == "cuda"),
            calc_method=calc_method,
            use_zero_force=False,
        )
        r_cur = np.asarray(r_list[-1], dtype=np.float64, order="F")
        v_cur = np.asarray(v_list[-1], dtype=np.float64, order="F")

        dt_step = (t_end - t_start) / len(r_list)
        t_chunk = t_start + (np.arange(len(r_list)) + 1) * dt_step

        if skip_initial_dt is not None:
            mask = t_chunk >= skip_initial_dt
        else:
            mask = np.ones(len(r_list), dtype=bool)
        if not np.any(mask):
            continue
        r_arr = np.asarray(r_list, dtype=np.float64)[mask]
        r_sum += np.sum(r_arr, axis=0)
        n_valid += int(np.sum(mask))

        if (i + 1) % max(1, n_chunks // 5) == 0 or i == n_chunks - 1:
            logger.info("分块 %d/%d 完成", i + 1, n_chunks)

    if n_valid == 0:
        raise ValueError(
            "分块后无有效数据（skip_initial_dt 可能过大或轨迹过短）"
        )
    return (r_sum / n_valid).astype(np.float64, order="C")


def _ensure_paths() -> None:
    """确保 sys.path 含项目根与 build，以便导入 ionsim"""
    from setup_path import ensure_build_in_path

    ensure_build_in_path(_ROOT)
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def _get_t0_from_npz(path: Path) -> float | None:
    """从 npz 提取 t_us 或从文件名 t{X}us.npz 推断，用于 init_file 续跑"""
    import re

    data = dict(np.load(path, allow_pickle=True))
    if "t_us" in data:
        return float(np.asarray(data["t_us"]).item())
    if "t" in data:
        return float(np.asarray(data["t"]).item())
    m = re.match(r"t(\d+(?:\.\d+)?)us\.npz$", path.name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _build_parsed_and_force(
    root: Path,
    n_ions: int,
    init_file: str | None,
    duration_us: float,
    device: str,
) -> tuple:
    """
    构建 ParsedRun 与 force。
    返回 (parsed, force)。
    init_file 时 t0 从文件推断，演化从 t0 起运行 duration_us；否则 t0=0。
    """
    from Interface.cli import create_parser, parse_and_build

    t0_us = 0.0
    if init_file:
        init_path = Path(init_file)
        if not init_path.is_absolute():
            init_path = root / init_path
        if init_path.exists():
            t0_from_file = _get_t0_from_npz(init_path)
            if t0_from_file is not None:
                t0_us = t0_from_file

    parser = create_parser()
    args = parser.parse_args([])
    args.N = n_ions
    args.t0 = t0_us
    args.time = t0_us + duration_us  # 终止时刻 = t0 + 演化时长
    args.device = device
    args.init_file = init_file or ""

    parsed = parse_and_build(args, root)

    cfg = parsed.config
    parsed.params.t0 = t0_us / (cfg.dt * 1e6)
    parsed.params.duration = duration_us / (cfg.dt * 1e6)

    from main import _build_force

    force = _build_force(
        parsed.field_settings,
        parsed.config,
        np.asarray(parsed.params.q, dtype=np.float64),
        root,
    )
    return parsed, force


def run_equilibrium(
    *,
    init_file: str | Path | None = None,
    delay_us: float = 0.0,
    evolution_time_us: float = 100.0,
    n_ions: int | None = None,
    root: Path | None = None,
    out_dir: Path | None = None,
    device: str = "cpu",
    memory_limit_mb: float | None = None,
    force_chunked: bool = False,
    save_equilibrium_image: str | Path | None = None,
    show_plot: bool = False,
) -> np.ndarray:
    """
    端到端运行：动力学模拟 → 平衡位置 → 保存 npz。

    Parameters
    ----------
    init_file : str | Path | None
        初始 r/v 的 npz 路径（r: μm, v: m/s）。None 则随机初始化
    delay_us : float
        延时 (μs)，跳过前 N μs 的数据用于排除瞬态，默认 0
    evolution_time_us : float
        演化时间 (μs)，默认 100
    n_ions : int | None
        离子数。若指定 init_file 则从文件推断，否则必须指定
    root : Path | None
        项目根目录，None 则自动推断
    out_dir : Path | None
        输出目录，None 则用 TrajectoryAnalysis/equi_pos
    device : str
        "cpu" 或 "cuda"
    memory_limit_mb : float | None
        单次轨迹内存上限 (MB)。超过则自动分块计算，避免 OOM。None 用默认 1000
    force_chunked : bool
        强制分块模式，即使估算内存未超限
    save_equilibrium_image : str | Path | None
        平衡位置图保存路径
    show_plot : bool
        是否弹窗显示平衡位置图

    Returns
    -------
    r_eq_um : (N, 3) array
        平衡位置 (μm)
    """
    _ensure_paths()
    import ionsim

    root = root or _ROOT
    out_dir = out_dir or EQUI_POS_DIR

    if init_file is not None:
        init_path = Path(init_file)
        if not init_path.is_absolute():
            init_path = root / init_path
        data = dict(np.load(init_path, allow_pickle=True))
        r_um = np.asarray(data["r"], dtype=float)
        n_ions = r_um.shape[0]
    elif n_ions is not None:
        n_ions = int(n_ions)
    else:
        raise ValueError("须指定 init_file 或 n_ions")

    if delay_us < 0 or evolution_time_us <= 0:
        raise ValueError("delay_us >= 0 且 evolution_time_us > 0")

    parsed, force = _build_parsed_and_force(
        root,
        n_ions,
        str(init_file) if init_file else None,
        evolution_time_us,
        device,
    )
    cfg = parsed.config
    p = parsed.params
    t0_dt = 0.0
    duration_dt = p.duration_in_dt()
    time_end_dt = t0_dt + duration_dt
    skip_initial_dt = delay_us / (cfg.dt * 1e6) if delay_us > 0 else None

    r0 = np.asarray(p.get_r0(), dtype=np.float64, order="F")
    v0 = np.asarray(p.get_v0(), dtype=np.float64, order="F")
    charge = np.asarray(p.q, dtype=np.float64)
    mass = np.asarray(p.m, dtype=np.float64)

    n_steps = max(int(duration_dt * 2), 500)
    mem_bytes = _estimate_trajectory_memory(n_steps, n_ions)
    mem_mb = mem_bytes / (1024 * 1024)
    limit_mb = memory_limit_mb if memory_limit_mb is not None else _MEMORY_CHUNK_MB

    if mem_mb > _MEMORY_WARN_MB:
        logger.warning(
            "轨迹内存估算: %s (N=%d, %d 步)。"
            "若内存不足可设 memory_limit_mb 或 --memory-limit 启用分块",
            _format_bytes(mem_bytes),
            n_ions,
            n_steps,
        )
    if mem_mb > limit_mb or force_chunked:
        logger.info(
            "启用分块模式 (估算 %s > %d MB)，按块计算平衡位置以控制内存",
            _format_bytes(mem_bytes),
            int(limit_mb),
        )
        r_eq_dim = _run_chunked(
            ionsim=ionsim,
            r0=r0,
            v0=v0,
            charge=charge,
            mass=mass,
            force=force,
            t0_dt=t0_dt,
            duration_dt=duration_dt,
            n_ions=n_ions,
            skip_initial_dt=skip_initial_dt,
            device=device,
            calc_method=p.calc_method,
            limit_mb=limit_mb,
        )
    else:
        r_in = np.asarray(r0, dtype=np.float64, order="F")
        v_in = np.asarray(v0, dtype=np.float64, order="F")

        r_list, v_list = ionsim.calculate_trajectory(
            r_in,
            v_in,
            charge,
            mass,
            step=n_steps,
            time_start=t0_dt,
            time_end=time_end_dt,
            force=force,
            use_cuda=(device == "cuda"),
            calc_method=p.calc_method,
            use_zero_force=False,
        )
        dt_step = duration_dt / len(r_list)
        t = t0_dt + (np.arange(len(r_list)) + 1) * dt_step

        from TrajectoryAnalysis.equilibrium import equilibrium_from_trajectory

        r_eq_dim = equilibrium_from_trajectory(
            r_list,
            t,
            skip_initial_dt=skip_initial_dt,
            method="mean",
        )
    r_eq_um = r_eq_dim * cfg.dl * 1e6

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"N{n_ions}.npz"
    np.savez(npz_path, r=r_eq_um)
    logger.info("已保存平衡位置: %s", npz_path)

    if save_equilibrium_image is not None or show_plot:
        from TrajectoryAnalysis.visualize import plot_equilibrium

        out_path = save_equilibrium_image
        if out_path is not None:
            out_path = Path(out_path)
            if out_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
                out_path = out_path / f"N{n_ions}.png"
        plot_equilibrium(
            r_eq_um,
            out_path=out_path,
            show=show_plot,
            title=f"Equilibrium (N={n_ions})",
        )
        if out_path is not None:
            logger.info("已保存平衡位置图: %s", out_path)

    return r_eq_um


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从动力学模拟计算离子晶格平衡位置并保存"
    )
    parser.add_argument(
        "--init_file",
        type=str,
        default=None,
        help="初始 r/v 的 npz 路径 (r: μm, v: m/s)；未指定则随机初始化",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="延时 (μs)，跳过前 N μs 的数据用于排除瞬态，默认 0",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=100.0,
        help="演化时长 (μs)，默认 100",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="离子数；未指定 init_file 时必填；指定 init_file 时从文件推断",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认 TrajectoryAnalysis/equi_pos",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        metavar="MB",
        help="轨迹内存上限 (MB)，超过则自动分块。默认 1000；N≥1000 时建议设 500",
    )
    parser.add_argument(
        "--force-chunked",
        action="store_true",
        help="强制分块模式，即使估算内存未超限",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="弹窗显示平衡位置图",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="平衡位置图保存路径；若为目录则保存为 {out}/N{N}.png",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 解析 --out：若为目录则拼接 N{N}.png
    save_equilibrium_image = None
    if args.out:
        out_p = Path(args.out)
        if out_p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
            save_equilibrium_image = out_p
        else:
            save_equilibrium_image = out_p  # 目录，run_equilibrium 内拼接 N{N}.png

    try:
        run_equilibrium(
            init_file=args.init_file,
            delay_us=args.delay,
            evolution_time_us=args.duration,
            n_ions=args.N,
            device=args.device,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            memory_limit_mb=args.memory_limit,
            force_chunked=args.force_chunked,
            save_equilibrium_image=save_equilibrium_image,
            show_plot=args.plot,
        )
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
