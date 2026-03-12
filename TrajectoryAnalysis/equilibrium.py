"""
从动力学轨迹推算各离子 secular motion 的平衡位置

通过时间平均消除 micromotion 与 secular 振荡，得到晶格中各离子的平衡位置。
"""
from __future__ import annotations

from typing import Literal

import numpy as np


def equilibrium_from_trajectory(
    r_list: list[np.ndarray] | np.ndarray,
    t: np.ndarray | None = None,
    *,
    skip_initial_dt: float | None = None,
    method: Literal["mean", "rf_cycle_mean"] = "mean",
    n_per_rf_cycle: int = 5,
) -> np.ndarray:
    """
    根据动力学轨迹计算各离子 secular motion 的平衡位置。

    通过时间平均消除 micromotion（RF 频率）与 secular 振荡（阱频），
    得到晶格中各离子的平衡位置。要求轨迹时长覆盖多个 secular 周期
    （典型阱频 1–5 MHz，周期 0.2–1 μs，建议总时长 ≥ 5 μs）。

    Parameters
    ----------
    r_list : list of (N, 3) array, or (n_steps, N, 3) array
        轨迹位置序列，无量纲单位 (dl)
    t : (n_steps,) array, optional
        时间序列 (dt 单位)。若为 None，假定均匀采样
    skip_initial_dt : float, optional
        跳过前 skip_initial_dt (dt 单位) 的数据，用于排除初始瞬态
    method : {"mean", "rf_cycle_mean"}
        - "mean": 直接对全轨迹取时间平均
        - "rf_cycle_mean": 先按 RF 周期分组平均（消 micromotion），再取平均
    n_per_rf_cycle : int, default 5
        method="rf_cycle_mean" 时，每个 RF 周期内的采样点数。
        1 RF 周期 ≈ π dt，程序默认 dt 对应 2 rad 相位，故约 4–5 点/周期

    Returns
    -------
    r_eq : (N, 3) array
        各离子平衡位置，与输入相同的无量纲单位 (dl)

    Examples
    --------
    >>> r_list, v_list, t = run_simulation(...)
    >>> r_eq = equilibrium_from_trajectory(r_list, t, skip_initial_dt=100)
    >>> r_eq_um = r_eq * cfg.dl * 1e6  # 转为 μm
    """
    r_arr = np.asarray(r_list, dtype=np.float64)
    if r_arr.ndim == 2:
        r_arr = r_arr[np.newaxis, ...]
    if r_arr.ndim != 3:
        raise ValueError("r_list 须为 (n_steps, N, 3) 或 list of (N, 3)")

    n_steps, n_ions, _ = r_arr.shape
    if n_steps < 2:
        raise ValueError("轨迹至少需要 2 个时间步")

    # 跳过初始瞬态
    if skip_initial_dt is not None and t is not None:
        if len(t) != n_steps:
            raise ValueError(
                f"t 长度 {len(t)} 与轨迹步数 {n_steps} 不一致"
            )
        mask = t >= skip_initial_dt
        if not np.any(mask):
            raise ValueError(
                f"skip_initial_dt={skip_initial_dt} 过大，无剩余数据 "
                f"(t 范围 [{float(t.min()):.1f}, {float(t.max()):.1f}])"
            )
        r_arr = r_arr[mask]
        n_steps = r_arr.shape[0]
    elif skip_initial_dt is not None and t is None:
        # 无 t 时按索引估算：假定均匀 dt_step
        n_skip = int(skip_initial_dt)
        if n_skip >= n_steps:
            raise ValueError(
                f"skip_initial_dt={skip_initial_dt} 对应的索引超出轨迹长度 {n_steps}"
            )
        r_arr = r_arr[n_skip:]
        n_steps = r_arr.shape[0]

    if method == "mean":
        return np.mean(r_arr, axis=0).astype(np.float64, order="C")

    if method == "rf_cycle_mean":
        # 每 n_per_rf_cycle 点为一组，组内平均消 micromotion，再对组取平均
        n_groups = n_steps // n_per_rf_cycle
        if n_groups < 1:
            raise ValueError(
                f"轨迹长度 {n_steps} 不足以分组 "
                f"(n_per_rf_cycle={n_per_rf_cycle})，请用 method='mean'"
            )
        r_trimmed = r_arr[: n_groups * n_per_rf_cycle]
        r_groups = r_trimmed.reshape(n_groups, n_per_rf_cycle, n_ions, 3)
        r_rf_avg = np.mean(r_groups, axis=1)  # (n_groups, N, 3)
        return np.mean(r_rf_avg, axis=0).astype(np.float64, order="C")

    raise ValueError(f"method 须为 'mean' 或 'rf_cycle_mean'，当前为 {method!r}")
