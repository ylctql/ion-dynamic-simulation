"""radial_trap 绘图（延迟 import matplotlib，供 CLI 无头与 notebook 复用）。"""
from __future__ import annotations

import numpy as np


def _plt():
    """延迟导入 matplotlib，调用方负责后端设置。"""
    import matplotlib.pyplot as plt

    return plt


def plot_lattice(
    result,
    *,
    out_path=None,
    show: bool = False,
    show_ion_index: bool = True,
):
    """径向平面平衡构型 + 每离子解析 micromotion + |a_mm|/ρ 汇总双联图。

    左图：xoy 平面构型（横轴 y，纵轴 x，等比坐标）。黑色散点为平衡位置，
    红色线段为该离子解析 micromotion 的 peak-to-peak 范围（端点
    r_eq ± a_mm，总长 2|a_mm|，方向即 a_mm 矢量方向）。

    右图：各离子 |a_mm|（左轴，µm）与 excess 比 ρ_a（右轴，仅有限值，
    四极极限=1 参考线）随离子序号变化。

    Parameters
    ----------
    out_path : 保存路径（None 不保存）
    show : 弹出交互窗口（需 GUI 后端）
    show_ion_index : 左图散点旁标注离子索引
    """
    plt = _plt()
    eq = result.equilibrium
    mm = result.micromotion
    r = eq.r_eq_um
    n = r.shape[0]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), layout="constrained"
    )

    # 左：xoy 构型 + micromotion peak-to-peak 线段（r ± a_mm）
    for i in np.argsort(r[:, 1]):
        a = mm.a_mm_um[i, :2]
        ax1.plot(
            [r[i, 1] - a[1], r[i, 1] + a[1]],
            [r[i, 0] - a[0], r[i, 0] + a[0]],
            color="red", lw=1.5, alpha=0.8, zorder=2,
        )
    ax1.scatter(r[:, 1], r[:, 0], s=28, color="black", zorder=3, label="r_eq")
    if show_ion_index:
        for i in range(n):
            ax1.annotate(
                str(i), (r[i, 1], r[i, 0]), fontsize=7,
                xytext=(3, 3), textcoords="offset points",
            )
    ax1.axhline(0.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.axvline(0.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlabel("y (µm)")
    ax1.set_ylabel("x (µm)")
    ax1.set_title(f"xoy 平面构型 (N={n})")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(loc="upper right", fontsize=8)

    # 右：|a_mm|（左轴）与 excess 比 ρ（右轴）vs 离子序号
    idx = np.arange(n)
    ax2.plot(idx, mm.mag_um, "o-", color="tab:blue", ms=4, label="|a_mm|")
    ax2.set_xlabel("离子序号")
    ax2.set_ylabel("|a_mm| (µm)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("解析 micromotion 幅度与 excess 比")
    axr = ax2.twinx()
    for ai, (comp, color) in enumerate((("x", "tab:red"), ("y", "tab:green"))):
        rho = mm.excess_ratio[:, ai]
        finite = np.isfinite(rho)
        if finite.any():
            axr.plot(
                idx[finite], rho[finite], "s", ms=4, color=color, alpha=0.7,
                label=f"ρ_{comp} (有限值)",
            )
    axr.set_ylabel("excess 比 ρ（四极极限=1）")
    axr.axhline(1.0, color="gray", lw=0.7, ls=":")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"图输出: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
