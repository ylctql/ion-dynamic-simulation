"""
平衡位置可视化：2D 投影散点图

支持 xoy、zoy、zox 视角，可保存图片或弹窗显示。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from Plotter.color import get_colors

PlotView = Literal["xoy", "zoy", "zox"]


def _get_xy(r_um: np.ndarray, view: PlotView) -> np.ndarray:
    """根据视角返回 (x_display, y_display)"""
    if view == "xoy":
        return np.column_stack((r_um[:, 0], r_um[:, 1]))
    if view == "zoy":
        return np.column_stack((r_um[:, 2], r_um[:, 1]))
    # zox
    return np.column_stack((r_um[:, 2], r_um[:, 0]))


def plot_equilibrium(
    r_um: np.ndarray,
    *,
    views: tuple[PlotView, ...] = ("zoy", "zox"),
    ion_size: float = 5.0,
    color_mode: Literal["y_pos", "index", None] = "y_pos",
    mass: np.ndarray | None = None,
    x_range: float | None = None,
    y_range: float | None = None,
    z_range: float | None = None,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    title: str | None = "Equilibrium positions",
    out_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """
    绘制离子平衡位置的 2D 投影散点图。

    Parameters
    ----------
    r_um : (N, 3) array
        平衡位置 (μm)，列序为 x, y, z
    views : tuple of {"xoy", "zoy", "zox"}
        子图视角，默认 ("zoy", "zox") 与轨迹绘图一致
    ion_size : float
        散点大小
    color_mode : "y_pos" | "index" | None
        "y_pos": 按 y 坐标上色（蓝→红）
        "index": 按离子索引上色（便于区分）
        None: 统一红色
    mass : (N,) array, optional
        质量，color_mode="isotope" 时用于同位素着色（当前仅支持 y_pos/index/None）
    x_range, y_range, z_range : float, optional
        各轴显示半宽 (μm)。None 则从数据自动推断（取 max 的 1.2 倍 + 10 μm 余量）
    x0, y0, z0 : float
        显示中心 (μm)
    title : str, optional
        总标题
    out_path : str | Path, optional
        保存路径
    show : bool
        是否弹窗显示
    dpi : int
        保存分辨率
    """
    r_um = np.asarray(r_um, dtype=np.float64)
    if r_um.ndim != 2 or r_um.shape[1] != 3:
        raise ValueError(f"r_um 须为 (N, 3)，当前形状: {r_um.shape}")

    n_ions = r_um.shape[0]
    v_dummy = np.zeros_like(r_um)  # get_colors 需要 v，平衡位置无速度

    if color_mode == "y_pos":
        colors = get_colors(r_um, v_dummy, "y_pos", mass, cmap_name="RdBu")
    elif color_mode == "index":
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.arange(n_ions) % 10)
    else:
        colors = get_colors(r_um, v_dummy, None, mass)

    # 自动范围
    def _auto_range(arr: np.ndarray, center: float) -> float:
        span = np.ptp(arr) if arr.size > 0 else 20.0
        return max(span * 0.6 + 10.0, 20.0)

    xr = x_range if x_range is not None else _auto_range(r_um[:, 0], x0)
    yr = y_range if y_range is not None else _auto_range(r_um[:, 1], y0)
    zr = z_range if z_range is not None else _auto_range(r_um[:, 2], z0)

    n_axes = len(views)
    fig, axes = plt.subplots(n_axes, 1, figsize=(6, 5 * n_axes))
    if n_axes == 1:
        axes = [axes]

    for ax, view in zip(axes, views):
        xy = _get_xy(r_um, view)
        ax.scatter(xy[:, 0], xy[:, 1], s=ion_size, c=colors)
        ax.set_aspect("equal")
        if view == "xoy":
            ax.set_xlim(x0 - xr, x0 + xr)
            ax.set_ylim(y0 - yr, y0 + yr)
            ax.set_xlabel("x (μm)", fontsize=12)
            ax.set_ylabel("y (μm)", fontsize=12)
        elif view == "zoy":
            ax.set_xlim(z0 - zr, z0 + zr)
            ax.set_ylim(y0 - yr, y0 + yr)
            ax.set_xlabel("z (μm)", fontsize=12)
            ax.set_ylabel("y (μm)", fontsize=12)
        else:
            ax.set_xlim(z0 - zr, z0 + zr)
            ax.set_ylim(x0 - xr, x0 + xr)
            ax.set_xlabel("z (μm)", fontsize=12)
            ax.set_ylabel("x (μm)", fontsize=12)
        ax.set_title(view.upper(), fontsize=12)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(f"{title} (N={n_ions})", fontsize=14)
    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """CLI：从 npz 加载平衡位置并绘图"""
    import argparse

    parser = argparse.ArgumentParser(description="平衡位置可视化")
    parser.add_argument("npz", type=str, help="平衡位置 npz 路径（含 r 键）")
    parser.add_argument("-o", "--out", type=str, default=None, help="保存路径")
    parser.add_argument("--no-show", action="store_true", help="不弹窗，仅保存")
    parser.add_argument("--views", type=str, default="zoy,zox", help="视角，逗号分隔，如 zoy,zox")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    if "r" not in data:
        raise ValueError(f"npz 须含 'r' 键，当前键: {list(data.keys())}")
    r_um = np.asarray(data["r"], dtype=np.float64)

    views_tuple = tuple(v.strip().lower() for v in args.views.split(",") if v.strip())
    if not views_tuple:
        views_tuple = ("zoy", "zox")

    plot_equilibrium(
        r_um,
        views=views_tuple,
        out_path=args.out,
        show=not args.no_show,
    )
