"""
micromotion 分析绘图（延迟 import matplotlib，供 CLI 无头与 notebook 复用）。
"""
from __future__ import annotations

import numpy as np

from .micromotion import (
    CrossCheck,
    IonAxisResult,
    MicromotionReport,
    _AXIS_INDEX,
)


def _plt():
    """延迟导入 matplotlib，调用方负责后端设置。"""
    import matplotlib.pyplot as plt
    return plt


def _equilibrium_positions(
    report: MicromotionReport,
) -> tuple[np.ndarray, np.ndarray]:
    """各离子平衡位置 (N,3) µm 与每离子最大 warmup 裁剪帧数 (N,)。

    平衡位置取收敛段（裁掉各轴 warmup 后剩余帧）的均值，避免初始瞬态偏置
    （与 ``analyze_run`` 的 per-(ion,axis) 裁剪一致：每离子取其各轴裁剪量的
    最大值，保守地丢弃瞬态段）。轨迹短于裁剪量时回退整段均值。
    """
    r_um = report.trajectory.r_um
    n_ions = r_um.shape[1]
    drop_per_ion = np.zeros(n_ions, dtype=int)
    for (i, _ax), res in report.results.items():
        if 0 <= i < n_ions:
            drop_per_ion[i] = max(drop_per_ion[i], int(res.dropped_frames))
    r_eq = np.array([
        (r_um[drop_per_ion[i]:, i, :] if r_um.shape[0] > drop_per_ion[i]
         else r_um[:, i, :]).mean(axis=0)
        for i in range(n_ions)
    ])
    return r_eq, drop_per_ion


def plot_ion_timeseries(
    res: IonAxisResult,
    traj_t_us: np.ndarray,
    r_axis_um: np.ndarray,
    *,
    axis_label: str,
    ion_idx: int,
    freq_rf_MHz: float,
):
    """单离子单轴时序：r(t) + secular 包络 X_sec + micromotion 残差 δr。

    ``traj_t_us`` / ``r_axis_um`` 须为**完整（未裁剪）**轨迹；本函数按
    ``res.dropped_frames`` 内部裁掉 warmup 段，与 ``compute_micromotion`` 作用区间
    一致（``secular_envelope_um`` 已是裁后长度）。若发生过裁剪，画竖虚线标注 t*。
    """
    plt = _plt()
    i0 = int(res.dropped_frames)
    t_us = np.asarray(traj_t_us)[i0:]
    r_um = np.asarray(r_axis_um)[i0:]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
    axes[0].plot(t_us, r_um, lw=0.6, label="r(t)")
    if res.secular_envelope_um is not None:
        axes[0].plot(t_us, res.secular_envelope_um, "r-", lw=1.2,
                     label="X_sec envelope")
    if res.dropped_frames > 0 and np.isfinite(res.t_star_us):
        for ax in axes:
            ax.axvline(res.t_star_us, color="k", ls=":", lw=1, alpha=0.7,
                       label=f"t*={res.t_star_us:.2f}µs")
    axes[0].set_ylabel(f"ion {ion_idx} {axis_label} (µm)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    delta = r_um - res.secular_envelope_um if res.secular_envelope_um is not None \
        else r_um - np.mean(r_um)
    axes[1].plot(t_us, delta, lw=0.5, color="C2")
    axes[1].set_ylabel(f"δr = r − X_sec (µm)\n(micromotion + noise)")
    axes[1].set_xlabel("Simulation time (µs)")
    axes[1].grid(True, alpha=0.3)
    qstr = f"{res.q_eff:.4f}" if np.isfinite(res.q_eff) else "n/a"
    fig.suptitle(
        f"Ion {ion_idx} / {axis_label} — q_eff={qstr}, RF={freq_rf_MHz:g} MHz",
        fontsize=10,
    )
    return fig


def plot_qeff_histogram(report: MicromotionReport):
    """各轴 q_eff 直方图（横向比较 RF 径向轴 vs 轴向）。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for ax_label in report.axes:
        ai = _AXIS_INDEX[ax_label]
        col = report.q_eff[:, ai]
        col = col[np.isfinite(col)]
        if col.size:
            ax.hist(col, bins=min(20, max(5, col.size // 2)),
                    alpha=0.6, label=f"{ax_label} (med={np.median(col):.4f})")
    ax.set_xlabel("q_eff (measured modulation depth)")
    ax.set_ylabel("ion count")
    ax.set_title("Per-ion q_eff distribution by axis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_qeff_vs_displacement(report: MicromotionReport, cross: CrossCheck | None = None):
    """
    各轴：q_eff vs 离子平衡位置偏离阱中心的距离 |r_eq − center|。
    理论上线性阱下 q_eff 与位置无关（≈ q_theory），偏离指示 excess micromotion。
    """
    plt = _plt()
    r_eq, _ = _equilibrium_positions(report)   # (N,3) 平衡位置，裁掉各轴瞬态
    n_ions = r_eq.shape[0]
    center = np.array(cross.center_um) if cross is not None else r_eq.mean(axis=0)
    fig, axes = plt.subplots(1, len(report.axes), figsize=(4 * len(report.axes), 4),
                             sharey=True, layout="constrained")
    if len(report.axes) == 1:
        axes = [axes]
    for k, ax_label in enumerate(report.axes):
        ai = _AXIS_INDEX[ax_label]
        disp = np.abs(r_eq[:, ai] - center[ai])
        q = report.q_eff[:, ai]
        m = np.isfinite(q)
        axes[k].scatter(disp[m], q[m], s=12)
        if cross is not None:
            qt = cross.q_theory.get(ax_label, np.nan)
            if np.isfinite(qt):
                axes[k].axhline(qt, color="r", ls="--", lw=1,
                                label=f"q_theory={qt:.4f}")
            axes[k].legend(fontsize=8)
        axes[k].set_xlabel(f"|r_eq − center| {ax_label} (µm)")
        if k == 0:
            axes[k].set_ylabel("q_eff")
        axes[k].set_title(f"axis {ax_label}")
        axes[k].grid(True, alpha=0.3)
    fig.suptitle("q_eff vs displacement from RF null (excess micromotion)", fontsize=10)
    return fig


def plot_beta_vs_secular(report: MicromotionReport, cross: CrossCheck | None = None):
    """
    各轴：所有窗的 β(t) vs |X_sec(t) − center| 散点。
    线性阱预期 β = (q/2)·|X_sec − center|，叠加理论斜率线 q_theory/2。
    """
    plt = _plt()
    center = np.array(cross.center_um) if cross is not None else np.zeros(3)
    fig, axes = plt.subplots(1, len(report.axes), figsize=(4 * len(report.axes), 4),
                             sharey=True, layout="constrained")
    if len(report.axes) == 1:
        axes = [axes]
    for k, ax_label in enumerate(report.axes):
        ai = _AXIS_INDEX[ax_label]
        betas, disps = [], []
        for (i, ax), res in report.results.items():
            if ax != ax_label:
                continue
            betas.append(res.beta_t)
            disps.append(np.abs(res.r_secular_t - center[ai]))
        if not betas:
            continue
        b = np.concatenate(betas)
        d = np.concatenate(disps)
        axes[k].scatter(d, b, s=4, alpha=0.4)
        if cross is not None:
            qt = cross.q_theory.get(ax_label, np.nan)
            if np.isfinite(qt) and d.size:
                dmax = float(np.nanmax(d)) if d.size else 1.0
                xx = np.linspace(0, dmax, 50)
                axes[k].plot(xx, 0.5 * abs(qt) * xx, "r--", lw=1.2,
                             label=f"slope q/2={abs(qt)/2:.4f}")
                axes[k].legend(fontsize=8)
        axes[k].set_xlabel(f"|X_sec − center| {ax_label} (µm)")
        if k == 0:
            axes[k].set_ylabel("β(t) micromotion amp (µm)")
        axes[k].set_title(f"axis {ax_label}")
        axes[k].grid(True, alpha=0.3)
    fig.suptitle("β vs secular displacement (slope ≈ q/2)", fontsize=10)
    return fig


def plot_lattice_micromotion(
    report: MicromotionReport,
    *,
    rf_axis: str = "x",
    axial_axis: str = "z",
    amp_stat: str = "median",
    show_ion_index: bool = True,
    cross: CrossCheck | None = None,
):
    """离子晶格 (axial_axis × rf_axis) 平面平衡位置 + rf_axis 方向 micromotion 幅度竖线。

    默认 zox 平面（横轴 axial_axis=z 离子链，纵轴 rf_axis=x RF 径向，与 Plotter 的
    zox 视图约定一致）：在每个离子平衡位置画散点，并以竖线（沿 rf_axis 方向）标注该
    离子的 rf_axis micromotion 幅度——竖线以平衡位置为中心、半长 β（总长 2β =
    peak-to-peak），即离子在 RF 驱动下的 micromotion 振荡范围。偏离 RF 零场的离子
    β ∝ |X_sec| 更大、竖线变长 → excess micromotion 直接可视化。

    Parameters
    ----------
    rf_axis : micromotion 幅度所在轴（竖线方向），默认 "x"
    axial_axis : 离子链方向（横轴），默认 "z" → 默认 zox 平面
    amp_stat : 竖线半长统计量，"median"（β(t) 中位，稳健默认）或 "max"（峰值）
    show_ion_index : 是否在散点旁标注离子索引
    cross : 提供 RF 零场参考线（rf_axis 方向水平虚线 = center[rf_axis]）

    Notes
    -----
    不强制等比坐标（与 Plotter 的晶格视图不同）：离子链轴向范围通常远大于 RF 径向
    micromotion 幅度，等比会压扁竖线致不可读；此处以竖线（幅度信息）可读性优先。
    """
    plt = _plt()
    if rf_axis not in _AXIS_INDEX:
        raise ValueError(f"rf_axis 需为 x/y/z，收到 '{rf_axis}'")
    if axial_axis not in _AXIS_INDEX:
        raise ValueError(f"axial_axis 需为 x/y/z，收到 '{axial_axis}'")
    if rf_axis == axial_axis:
        raise ValueError("rf_axis 与 axial_axis 不能相同")
    if amp_stat not in ("median", "max"):
        raise ValueError(f"amp_stat 需为 'median' 或 'max'，收到 '{amp_stat}'")

    r_eq, _ = _equilibrium_positions(report)
    n_ions = r_eq.shape[0]
    ai_rf = _AXIS_INDEX[rf_axis]
    ai_axial = _AXIS_INDEX[axial_axis]

    # 每离子 rf_axis micromotion 幅度 β（半长）
    beta = np.full(n_ions, np.nan)
    for i in range(n_ions):
        res = report.results.get((i, rf_axis))
        if res is None or res.beta_t.size == 0:
            continue
        beta[i] = float(np.max(res.beta_t)) if amp_stat == "max" \
            else float(np.median(res.beta_t))
    has_beta = np.isfinite(beta)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

    # RF 零场参考线（若提供 cross）
    if cross is not None:
        null = float(np.array(cross.center_um)[ai_rf])
        ax.axhline(null, color="gray", ls="--", lw=1, alpha=0.7,
                   label=f"RF null {rf_axis}={null:.2f} µm")

    # 平衡位置散点
    ax.scatter(r_eq[:, ai_axial], r_eq[:, ai_rf], s=35, zorder=3,
               color="C0", edgecolor="white", linewidth=0.5,
               label="equilibrium position")

    # micromotion 竖线：中心对齐平衡位置，半长 β（总长 2β = ptp）
    plotted = False
    for i in range(n_ions):
        if not has_beta[i]:
            continue
        zc = r_eq[i, ai_axial]
        xc = r_eq[i, ai_rf]
        b = beta[i]
        ax.plot([zc, zc], [xc - b, xc + b], color="C3", lw=2.2, alpha=0.85,
                solid_capstyle="round", zorder=2,
                label=("micromotion ±β (total 2β = ptp)" if not plotted else None))
        plotted = True
        if show_ion_index:
            ax.annotate(str(i), (zc, xc), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color="C0")

    if not plotted:
        ax.text(0.5, 0.5,
                f"无 {rf_axis} 轴 micromotion 数据（--axes 未含 {rf_axis}）",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)

    ax.set_xlabel(f"{axial_axis} (µm)")
    ax.set_ylabel(f"{rf_axis} (µm)")
    stat_label = "median" if amp_stat == "median" else "max"
    fig.suptitle(
        f"Ion lattice ({axial_axis}o{rf_axis}) — {rf_axis} micromotion amplitude "
        f"(β {stat_label}, line total = 2β)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return fig
