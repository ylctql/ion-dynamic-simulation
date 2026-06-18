"""
micromotion 分析绘图（延迟 import matplotlib，供 CLI 无头与 notebook 复用）。
"""
from __future__ import annotations

import logging

import numpy as np

from .micromotion import (
    CrossCheck,
    IonAxisResult,
    MicromotionReport,
    _AXIS_INDEX,
)

logger = logging.getLogger(__name__)


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
    # 偏离参考用真 RF null（赝势最小）；无 cross 时用离子质心。
    if cross is not None:
        ref = cross.rf_null_um if cross.rf_null_um is not None else cross.center_um
        center = np.array(ref)
    else:
        center = r_eq.mean(axis=0)
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
                # q_eff 是非负幅度，对标 |q_theory|（q 符号仅约定方向，不影响幅度）；
                # 与 plot_beta_vs_secular / plot_lattice_micromotion 一致。
                aqt = abs(qt)
                axes[k].axhline(aqt, color="r", ls="--", lw=1,
                                label=f"|q_theory|={aqt:.4f}")
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
    if cross is not None:
        ref = cross.rf_null_um if cross.rf_null_um is not None else cross.center_um
        center = np.array(ref)
    else:
        center = np.zeros(3)
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


def _theory_offset_auto(axial_pos_um: np.ndarray) -> float:
    """理论竖线的自动 z 偏移量 (µm)，挂钩于中位离子间距。

    稀疏晶格（N < 4 或间距退化）回退到轴向跨度的 1.5%。密集晶格下偏移 ≈ 0.35·中位间距
    （绿线中心落在两红线之间约 1/3 处，清晰且不蹭邻居）；间距退化（< 1e-6µm，离子近重合）
    时记 warning，提示绿线可能蹭邻居、建议缩窄 --ions 范围或手动设 theory_z_offset。
    """
    pos = np.asarray(axial_pos_um, dtype=np.float64)
    n = pos.size
    if n < 4:
        return 0.015 * float(np.ptp(pos)) if n > 1 else 0.0
    gaps = np.diff(np.sort(pos))
    gaps = gaps[gaps > 0]            # 忽略近重合离子（实测可有间距 ~1e-4 的对）
    if gaps.size == 0:
        return 0.015 * float(np.ptp(pos))
    med_gap = float(np.median(gaps))
    offset = 0.35 * med_gap
    if med_gap < 1e-6:              # 间距退化（离子近重合），偏移无意义
        logger.warning(
            "晶格过密（N=%d, 中位轴向间距=%.3gµm 退化）：绿线可能蹭邻居离子，"
            "建议缩窄 --ions 范围或手动设 theory_z_offset", n, med_gap,
        )
    return offset


def plot_lattice_micromotion(
    report: MicromotionReport,
    *,
    rf_axis: str = "x",
    axial_axis: str = "z",
    amp_stat: str = "last",
    show_ion_index: bool = True,
    show_theory: bool = False,
    theory_z_offset: float | None = None,
    equal_aspect: bool = True,
    axis_ranges: dict[str, tuple[float, float]] | None = None,
    cross: CrossCheck | None = None,
):
    """离子晶格 (axial_axis × rf_axis) 平面**末端帧瞬时构型** + rf_axis micromotion 幅度竖线。

    默认 zox 平面（横轴 axial_axis=z 离子链，纵轴 rf_axis=x RF 径向，与 Plotter 的
    zox 视图约定一致）：在每个离子**最后一帧瞬时位置** r(T_end)（含该时刻 RF
    micromotion 偏移）画散点，并以竖线（沿 rf_axis 方向）标注该离子的 rf_axis
    micromotion 幅度——竖线以该瞬时位置为中心、半长 β（总长 2β = peak-to-peak），
    即离子在 RF 驱动下的 micromotion 振荡范围。偏离 RF 零场的离子 β ∝ |X_sec|
    更大、竖线变长 → excess micromotion 直接可视化。

    用末端帧而非时间均值：后者在 secular 振荡下把运动抹平、unphysical；末端瞬时构型
    与末端窗 β(t_end) 同快照，位置与幅度自洽。

    Parameters
    ----------
    rf_axis : micromotion 幅度所在轴（竖线方向），默认 "x"
    axial_axis : 离子链方向（横轴），默认 "z" → 默认 zox 平面
    amp_stat : 竖线半长统计量，"last"（末端窗 β(t_end)，与末端构型同快照，默认）、
        "median"（β(t) 中位）或 "max"（峰值）
    show_ion_index : 是否在散点旁标注离子索引
    show_theory : 叠加**理论** micromotion 幅度竖线 β_theory = |q_theory|/2·|x_last − x_null|
        （q_theory、x_null 来自 cross / trap_stability，绿色虚线），与数值竖线（红色
        实线）并列对比 excess micromotion（数值 > 理论）。仅用于比对，主幅度恒为数值
        phase-folding 测量的 β。需提供 cross，否则 warning 跳过。
    theory_z_offset : 理论竖线沿 axial_axis 的平移量 (µm)，避免被实测竖线（同 z 同中心）
        完全覆盖。None 时自动取 **中位轴向离子间距的 35%**（稀疏晶格 N<4 时回退到轴向跨度
        的 1.5%）：实测线留在离子原位，理论线错开到 +z 侧并排比对高度。偏移挂钩于离子间距
        而非全局跨度，保证密集晶格（N≫100，间距 ≪ 跨度）下绿线紧贴各自红线、不跨越邻居
        导致肉眼把绿_i 错配到红_{i+k}（看似"中心不一/位移乱"的视觉错觉）。设 0 可禁用平移。
    equal_aspect : 两维（axial_axis × rf_axis）是否等比——横纵 1µm 代表相同物理长度
        （默认 True）。等比下按 xlim/ylim 跨度比动态调 figsize（短边基准 5"，比例限
        [0.4, 5]），避免固定画框把晶格压扁/拉伸；晶格过扁长时建议配合 axis_ranges 缩放到
        感兴趣区段。设 False 回退自由比例（matplotlib auto）。
    axis_ranges : 各**物理轴**的显示范围 (µm)，dict key="x"/"y"/"z"，值 (lo, hi)。仅取与
        本图相关的两轴——横轴 axial_axis、纵轴 rf_axis 对应的物理轴；未提供该轴时按数据
        （含竖线端点）自动适配。如默认 zox 平面，axis_ranges={"z":(0,30),"x":(-5,5)}
        等价于 CLI --z-range 0,30 --x-range -5,5。
    cross : 提供 RF 零场参考线（rf_axis 方向水平虚线 = center[rf_axis]）；show_theory
        时同时提供理论 q_theory 与 x_null

    Notes
    -----
    默认等比坐标（两维 µm 尺度一致），与 Plotter 的晶格视图一致：保证 micromotion 竖线长度
    在两维上不被扭曲，晶格几何形状（间距、链长）忠实反映。等比下 figsize 按数据跨度比动态
    设置；晶格过扁长（轴向 ≫ 径向）时整图会很长，建议用 axis_ranges（CLI --x-range/
    --y-range/--z-range）缩放到感兴趣区段。
    """
    plt = _plt()
    if rf_axis not in _AXIS_INDEX:
        raise ValueError(f"rf_axis 需为 x/y/z，收到 '{rf_axis}'")
    if axial_axis not in _AXIS_INDEX:
        raise ValueError(f"axial_axis 需为 x/y/z，收到 '{axial_axis}'")
    if rf_axis == axial_axis:
        raise ValueError("rf_axis 与 axial_axis 不能相同")
    if amp_stat not in ("last", "median", "max"):
        raise ValueError(f"amp_stat 需为 'last'/'median'/'max'，收到 '{amp_stat}'")

    # 末端帧瞬时构型位置（含该时刻 RF micromotion 偏移）；时间均值在 secular 振荡下 unphysical
    r_last = np.asarray(report.trajectory.r_um[-1], dtype=np.float64)   # (N,3)
    n_ions = r_last.shape[0]
    ai_rf = _AXIS_INDEX[rf_axis]
    ai_axial = _AXIS_INDEX[axial_axis]

    # 每离子 rf_axis micromotion 幅度 β（半长）
    beta = np.full(n_ions, np.nan)
    for i in range(n_ions):
        res = report.results.get((i, rf_axis))
        if res is None or res.beta_t.size == 0:
            continue
        if amp_stat == "max":
            beta[i] = float(np.max(res.beta_t))
        elif amp_stat == "median":
            beta[i] = float(np.median(res.beta_t))
        else:   # "last"：末端窗 β，与末端构型同快照
            beta[i] = float(res.beta_t[-1])
    has_beta = np.isfinite(beta)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

    # RF 零场参考线：用真 RF null（赝势最小，与 DC 偏置无关），而非 center_um
    # （后者为 V_total 最小，含 DC 偏置会被推离 RF null，误标为 "RF null"）。
    if cross is not None:
        ref = cross.rf_null_um if cross.rf_null_um is not None else cross.center_um
        null = float(np.array(ref)[ai_rf])
        ax.axhline(null, color="gray", ls="--", lw=1, alpha=0.7,
                   label=f"RF null {rf_axis}={null:.2f} µm")

    # 末端帧瞬时位置散点
    ax.scatter(r_last[:, ai_axial], r_last[:, ai_rf], s=35, zorder=3,
               color="C0", edgecolor="white", linewidth=0.5,
               label="last-frame position")

    # 可选理论比对：β_theory = |q_theory|/2·|x_eq − x_null|（q_theory/x_null 来自 cross）
    beta_theory = None
    q_th = np.nan
    if show_theory:
        if cross is None:
            logger.warning(
                "show_theory=True 需提供 cross（trap_stability 理论 q），跳过理论比对"
            )
        else:
            q_th = float(cross.q_theory.get(rf_axis, np.nan))
            if not np.isfinite(q_th):
                logger.warning("cross.q_theory[%r] 非 finite，跳过理论比对", rf_axis)
            else:
                ref = cross.rf_null_um if cross.rf_null_um is not None else cross.center_um
                x_null = float(np.array(ref)[ai_rf])
                beta_theory = 0.5 * abs(q_th) * np.abs(r_last[:, ai_rf] - x_null)

    # 理论竖线（绿色虚线）：沿 z 错开 theory_z_offset 与实测线并排，避免被红实线完全
    # 覆盖（加粗方案在 excess 情形实测>理论时理论线整段被盖、仅露细翼，不可读）。
    if beta_theory is not None:
        if theory_z_offset is None:
            # 偏移挂钩于中位离子间距（密集晶格不跨邻居），稀疏晶格回退到跨度 1.5%。
            # 旧的 1.5%·span 在 N=1000 时偏移 ≈ 0.15·span ≫ 间距，绿线整体右移跨过
            # 几十个邻居 → 肉眼把绿_i 错配到红_{i+k}，看似"中心不一/位移乱"。
            theory_z_offset = _theory_offset_auto(r_last[:, ai_axial])
        plotted_th = False
        for i in range(n_ions):
            if not has_beta[i] or not np.isfinite(beta_theory[i]):
                continue
            zc = r_last[i, ai_axial] + theory_z_offset   # 错开到 +z 侧
            xc = r_last[i, ai_rf]
            bt = float(beta_theory[i])
            ax.plot([zc, zc], [xc - bt, xc + bt], color="C2", ls="--", lw=2.0,
                    alpha=0.85, solid_capstyle="round", zorder=2,
                    label=(rf"β theory = |q_th|/2·|x−x_null|  (q_th={q_th:.4f}, "
                           rf"z+{theory_z_offset:.2f}µm)" if not plotted_th else None))
            plotted_th = True

    # 数值竖线（phase-folding 实测，红色实线）：中心对齐末端瞬时位置，半长 β（总长 2β = ptp）
    plotted = False
    for i in range(n_ions):
        if not has_beta[i]:
            continue
        zc = r_last[i, ai_axial]
        xc = r_last[i, ai_rf]
        b = beta[i]
        ax.plot([zc, zc], [xc - b, xc + b], color="C3", lw=2.2, alpha=0.9,
                solid_capstyle="round", zorder=3,
                label=("micromotion ±β measured (total 2β = ptp)" if not plotted else None))
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
    stat_label = amp_stat   # last / median / max
    fig.suptitle(
        f"Ion lattice ({axial_axis}o{rf_axis}) — {rf_axis} micromotion amplitude "
        f"(β {stat_label}, line total = 2β)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # 等比坐标（两维 µm 尺度一致）+ 各物理轴显示范围
    _apply_lattice_aspect(
        ax, fig, axial_axis, rf_axis, axis_ranges, equal_aspect,
        r_last[:, ai_axial], r_last[:, ai_rf],
    )
    return fig


def _apply_lattice_aspect(
    ax, fig, axial_axis: str, rf_axis: str,
    axis_ranges: dict[str, tuple[float, float]] | None, equal_aspect: bool,
    axial_pos: np.ndarray, rf_pos: np.ndarray,
):
    """对晶格图应用各物理轴显示范围 + 可选等比坐标（按跨度比动态调 figsize）。

    横轴 = axial_axis 物理轴、纵轴 = rf_axis 物理轴；axis_ranges 按物理轴名 ("x"/"y"/"z")
    取 (lo, hi)，未指定的轴按散点 + 竖线端点 + RF null 自动适配。

    手动收集数据范围而非 ``ax.relim()``：后者**不收集 scatter (PathCollection)**，密集晶格
    下会漏掉散点、只按竖线缩放（如 1000 离子只画 4 条竖线时 xlim 塌缩到竖线 z 附近）。
    """
    ranges = axis_ranges or {}
    x_range = ranges.get(axial_axis)   # 横轴（axial 物理轴）
    y_range = ranges.get(rf_axis)      # 纵轴（rf 物理轴）

    xs = list(np.asarray(axial_pos, dtype=np.float64).ravel())
    ys = list(np.asarray(rf_pos, dtype=np.float64).ravel())
    for ln in ax.get_lines():
        xd, yd = ln.get_xdata(), ln.get_ydata()
        if len(xd) != 2 or len(yd) != 2:
            continue
        x_eq = np.allclose(xd[0], xd[1])
        y_eq = np.allclose(yd[0], yd[1])
        if x_eq and not y_eq:        # 竖线（micromotion 竖线）：x 数据、y 端点
            xs.append(float(xd[0]))
            ys.extend([float(yd[0]), float(yd[1])])
        elif y_eq and not x_eq:      # 水平参考线（RF null）：y 数据（x 为 axes 坐标，丢弃）
            ys.append(float(yd[0]))
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]

    def _span(vals):
        if vals.size == 0:
            return -1.0, 1.0
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return lo - 0.5, hi + 0.5
        m = 0.05 * (hi - lo)
        return lo - m, hi + m

    xlo, xhi = _span(xs)
    ylo, yhi = _span(ys)
    if x_range is not None:
        xlo, xhi = float(x_range[0]), float(x_range[1])
    if y_range is not None:
        ylo, yhi = float(y_range[0]), float(y_range[1])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    if equal_aspect:
        xs_span, ys_span = (xhi - xlo), (yhi - ylo)
        if xs_span > 0 and ys_span > 0:
            ratio = min(max(xs_span / ys_span, 0.4), 5.0)
            base = 5.0
            if ratio >= 1.0:
                w, h = base * ratio, base
            else:
                w, h = base, base / ratio
            fig.set_size_inches(w, h)
        ax.set_aspect("equal", adjustable="box")
