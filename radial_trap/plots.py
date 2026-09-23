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

    左图：xoy 平面构型（横轴 x = 链/弱轴，对应 3D xoz 面晶格；纵轴 y，等比
    坐标）。黑色散点为平衡位置，红色线段为该离子解析 micromotion 的
    peak-to-peak 范围（端点 r_eq ± a_mm，总长 2|a_mm|，方向即 a_mm 矢量方向）。

    右图：各离子 |a_mm|（左轴，µm）与 excess 比 ρ_a（右轴，仅有限值，
    四极极限=1 参考线）随离子序号变化。

    离子标号按链轴坐标（链沿 x 时即 x 坐标）**排序后重新编号**（0..N−1，
    沿链单调递增）：数组序号来自随机初始化、与空间位置无关，直接使用会使
    右图曲线/左图标号呈无序跳动，掩盖 micromotion 的空间依赖。

    Parameters
    ----------
    out_path : 保存路径（None 不保存）
    show : 弹出交互窗口（需 GUI 后端）
    show_ion_index : 左图散点旁标注离子标号（链轴排序后的序号）
    """
    plt = _plt()
    eq = result.equilibrium
    mm = result.micromotion
    r = eq.r_eq_um
    n = r.shape[0]

    # 链轴 = 展宽更大的横向轴（默认 x；E>0 时链镜像翻转到 y 轴）
    chain_axis = 0 if np.ptp(r[:, 0]) >= np.ptp(r[:, 1]) else 1
    axis_name = ("x", "y")[chain_axis]
    # order[k] = 按链轴坐标排第 k 位的离子数组下标
    order = np.argsort(r[:, chain_axis])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), layout="constrained"
    )

    # 左：xoy 构型 + micromotion peak-to-peak 线段（r ± a_mm），x 水平
    for i in order:
        a = mm.a_mm_um[i, :2]
        ax1.plot(
            [r[i, 0] - a[0], r[i, 0] + a[0]],
            [r[i, 1] - a[1], r[i, 1] + a[1]],
            color="red", lw=1.5, alpha=0.8, zorder=2,
        )
    ax1.scatter(r[:, 0], r[:, 1], s=28, color="black", zorder=3, label="r_eq")
    if show_ion_index:
        for rank, i in enumerate(order):
            ax1.annotate(
                str(rank), (r[i, 0], r[i, 1]), fontsize=7,
                xytext=(3, 3), textcoords="offset points",
            )
    ax1.axhline(0.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.axvline(0.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlabel("x (µm)")
    ax1.set_ylabel("y (µm)")
    # 图内文本统一英文（matplotlib 默认字体无 CJK 字形，中文会渲染为方框）
    ax1.set_title(f"xoy-plane lattice (N={n})")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(loc="upper right", fontsize=8)

    # 右：|a_mm|（左轴）与 excess 比 ρ（右轴）vs 离子序号（链轴排序后）
    idx = np.arange(n)
    ax2.plot(idx, mm.mag_um[order], "o-", color="tab:blue", ms=4, label="|a_mm|")
    ax2.set_xlabel(f"ion index ({axis_name}-sorted)")
    ax2.set_ylabel("|a_mm| (µm)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Analytic micromotion amplitude and excess ratio")
    axr = ax2.twinx()
    for ai, (comp, color) in enumerate((("x", "tab:red"), ("y", "tab:green"))):
        rho = mm.excess_ratio[order, ai]
        finite = np.isfinite(rho)
        if finite.any():
            axr.plot(
                idx[finite], rho[finite], "s", ms=4, color=color, alpha=0.7,
                label=f"ρ_{comp} (finite)",
            )
    axr.set_ylabel("excess ratio ρ (quadrupole limit = 1)")
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


def _fill_potential_panel(ax, x, y, v, title, clip_pct, colorbar=True):
    """单面板势场填充图：自动分符号结构选 cmap + 稳健百分位裁剪。

    - 数据跨零（lo < 0 < hi）→ RdBu_r 关于 0 对称（DC/bias 类符号变化场）
    - 单侧符号（如赝势 ≥ 0）→ viridis 顺序 cmap
    - 全零面板（如 D=0 的 bias）→ 平色 + 标题注记，避免 contourf 空 levels
    """
    finite = v[np.isfinite(v)]
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    # v 形状 (nx, ny)（indexing='ij'），contourf/pcolormesh 一维 x/y 需转置为 (ny, nx)
    if finite.size == 0 or float(np.max(np.abs(finite))) == 0.0:
        ax.pcolormesh(x, y, np.zeros_like(v).T, cmap="viridis", shading="auto")
        ax.set_title(f"{title} (zero everywhere)")
        return
    lo, hi = np.nanpercentile(v, clip_pct)
    if lo < 0.0 < hi:
        m = max(abs(lo), abs(hi))
        cmap, vmin, vmax = "RdBu_r", -m, m
    else:
        cmap, vmin, vmax = "viridis", lo, hi
    levels = np.linspace(vmin, vmax, 65)
    im = ax.contourf(x, y, v.T, levels=levels, cmap=cmap, extend="both")
    ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="V", shrink=0.9)


def plot_potential_maps(
    params,
    *,
    r_eq_um=None,
    out_path=None,
    show: bool = False,
    n_pts: tuple[int, int] = (241, 161),
    clip_pct: tuple[float, float] = (0.5, 99.5),
):
    """径向平面势场分布 2×2 面板：RF 赝势 / RF bias / DC / 总势（叠加离子）。

    网格取 params.x_range_um × params.y_range_um；色标范围按 clip_pct
    百分位稳健裁剪（B/F≠0 时角落 r⁴ 增长会淹没中心结构，必须裁）。
    色图按各组分符号结构自动选择（赝势 ≥0 顺序色；bias/DC 跨零对称色）。

    Parameters
    ----------
    r_eq_um : 平衡位置 (N,3) µm（None 不叠加；给定时画在总势面板上）
    out_path : 保存路径（None 不保存）
    show : 弹出交互窗口（需 GUI 后端）
    n_pts : (nx, ny) 网格点数
    clip_pct : 色标范围的百分位裁剪（lo, hi）
    """
    plt = _plt()
    from radial_trap.potential import (
        bias_potential_V,
        dc_potential_V,
        pseudopotential_V,
        total_potential_V,
    )

    x = np.linspace(params.x_range_um[0], params.x_range_um[1], n_pts[0])
    y = np.linspace(params.y_range_um[0], params.y_range_um[1], n_pts[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    _fill_potential_panel(
        axes[0, 0], x, y, pseudopotential_V(xx, yy, params),
        "RF pseudopotential α|E_rf|²", clip_pct,
    )
    _fill_potential_panel(
        axes[0, 1], x, y, bias_potential_V(xx, yy, params),
        "RF bias D·φ_rf", clip_pct,
    )
    _fill_potential_panel(
        axes[1, 0], x, y, dc_potential_V(xx, yy, params),
        "DC E(x²−y²)+F(x⁴−6x²y²+y⁴)", clip_pct,
    )
    _fill_potential_panel(
        axes[1, 1], x, y, total_potential_V(xx, yy, params),
        "Total" + (f" + ions (N={r_eq_um.shape[0]})" if r_eq_um is not None else ""),
        clip_pct,
    )
    if r_eq_um is not None:
        axes[1, 1].scatter(
            r_eq_um[:, 0], r_eq_um[:, 1], s=12, color="black",
            zorder=3, label="r_eq",
        )
        axes[1, 1].legend(loc="upper right", fontsize=8)

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"势场图输出: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
