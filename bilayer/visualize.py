"""
双层 / 一般库仑 Hessian 可视化（matplotlib）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def subsample_hessian(
    hessian: np.ndarray,
    stride: int,
) -> np.ndarray:
    """对方阵按步长子采样，便于高维矩阵预览。"""
    h = np.asarray(hessian, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"须为方阵，当前 {h.shape}")
    s = int(stride)
    if s < 1:
        raise ValueError("stride 须 >= 1")
    return h[::s, ::s]


def plot_hessian_heatmap(
    hessian_eV_per_um2: np.ndarray,
    *,
    title: str = "Hessian (Coulomb)",
    out_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 220,
    figsize: tuple[float, float] | None = None,
    ions_per_layer: int | None = None,
) -> None:
    """
    将 Hessian 以热力图显示；可选在 ``3 * ions_per_layer`` 处画层间分界线。

    Parameters
    ----------
    hessian_eV_per_um2
        方阵，单位 eV/μm²。
    ions_per_layer
        若给定，在 DOF 索引 ``3 * ions_per_layer`` 处绘制白线标示两层分界。
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    h = np.asarray(hessian_eV_per_um2, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"Hessian 应为方阵，当前 {h.shape}")

    n = h.shape[0]
    if figsize is None:
        side = min(12.0, max(6.0, n / 150.0))
        figsize = (side, side * 0.92)

    hmin = float(np.min(h))
    hmax = float(np.max(h))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if hmin < 0.0 < hmax:
        norm = colors.TwoSlopeNorm(vmin=hmin, vcenter=0.0, vmax=hmax)
        im = ax.imshow(h, cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
    else:
        im = ax.imshow(h, cmap="viridis", origin="lower", aspect="auto", vmin=hmin, vmax=hmax)

    if ions_per_layer is not None and ions_per_layer > 0:
        cut = 3 * int(ions_per_layer)
        if 0 < cut < n:
            lw = 0.8
            ax.axhline(cut - 0.5, color="white", linewidth=lw, alpha=0.9)
            ax.axvline(cut - 0.5, color="white", linewidth=lw, alpha=0.9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("eV/μm²")
    ax.set_xlabel("DOF index")
    ax.set_ylabel("DOF index")
    ax.set_title(title)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi)
        print(f"Hessian 热力图已保存: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bilayer_hessian(
    hessian_eV_per_um2: np.ndarray,
    *,
    n_ions_per_layer: int,
    layer_spacing_um: float,
    out_path: str | Path | None = None,
    show: bool = False,
    stride: int = 1,
) -> None:
    """
    双层库仑 Hessian 热力图：默认带层间分界线；``stride>1`` 时先子采样再绘制。
    """
    h = np.asarray(hessian_eV_per_um2, dtype=float)
    if stride > 1:
        h = subsample_hessian(h, stride)
        title = (
            f"Bilayer Coulomb Hessian (subsample stride={stride}), "
            f"dy={layer_spacing_um:g} μm"
        )
        ions_marker = None
    else:
        title = f"Bilayer Coulomb Hessian, dy={layer_spacing_um:g} μm"
        ions_marker = n_ions_per_layer

    plot_hessian_heatmap(
        h,
        title=title,
        out_path=out_path,
        show=show,
        ions_per_layer=ions_marker,
    )
