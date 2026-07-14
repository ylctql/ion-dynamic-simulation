"""
解耦热力图：仅读 scan.log，画 x²(横轴) × x⁴(纵轴)、颜色 = 选定列（默认 σ_y）。

**不依赖动力学、ionsim 或 .npy**——给它任何符合 logio 列格式的 CSV 都能出图。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VALUE_COLS = ["sigma_y_final_um", "sim_time_us", "reached_threshold", "wall_time_s"]


def _round_key(v: float) -> float:
    """把坐标规范化为固定精度，避免浮点漂移导致同一点被拆成多格。"""
    return round(float(v), 10)


def load_grid(log_path: str | Path, value_col: str = "sigma_y_final_um"):
    """
    由 log 构建 (x2s, x4s, Z)。

    Returns
    -------
    x2s : (nx2,) 排序去重的 x² 系数
    x4s : (nx4,) 排序去重的 x⁴ 系数
    Z   : (nx2, nx4) 数值矩阵；缺失点为 NaN（pcolormesh 自动留白）
    """
    from .logio import read_log

    rows = read_log(log_path)
    if not rows:
        raise ValueError(f"log 无数据: {log_path}")

    x2_keys = sorted({_round_key(r["x2_coeff"]) for r in rows})
    x4_keys = sorted({_round_key(r["x4_coeff"]) for r in rows})
    ix2 = {v: i for i, v in enumerate(x2_keys)}
    ix4 = {v: i for i, v in enumerate(x4_keys)}

    Z = np.full((len(x2_keys), len(x4_keys)), np.nan)
    for r in rows:
        Z[ix2[_round_key(r["x2_coeff"])], ix4[_round_key(r["x4_coeff"])]] = r[value_col]
    return np.asarray(x2_keys, float), np.asarray(x4_keys, float), Z


def plot_heatmap(
    log_path: str | Path,
    out_path: str | Path | None = None,
    *,
    value_col: str = "sigma_y_final_um",
    log_color: bool = False,
    title: str | None = None,
) -> None:
    """读 log 画热力图；out_path 为 None 时弹窗，否则存图。"""
    import matplotlib

    matplotlib.use("Agg" if out_path else matplotlib.get_backend())
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    x2s, x4s, Z = load_grid(log_path, value_col)

    valid = Z[np.isfinite(Z)]
    norm = LogNorm() if (log_color and valid.size and np.nanmin(valid) > 0) else None
    vmin = vmax = None
    if norm is None and valid.size:
        vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid))

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.pcolormesh(
        x2s, x4s, Z, shading="auto", cmap="viridis",
        norm=norm, vmin=vmin, vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(value_col)
    ax.set_xlabel("x² coefficient (V)")
    ax.set_ylabel("x⁴ coefficient (V)")
    ax.set_title(title or f"{value_col} at simulation end")
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="由 scan.log 画 (x², x⁴, value) 热力图")
    p.add_argument("--log", required=True, help="scan.log 路径")
    p.add_argument("--out", default=None, help="输出图片路径；省略则弹窗")
    p.add_argument("--value-col", default="sigma_y_final_um", choices=VALUE_COLS,
                   help="着色列（默认结束时 σ_y）")
    p.add_argument("--log-color", action="store_true", help="颜色取对数刻度（σ_y 跨数量级时有用）")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    plot_heatmap(
        args.log, args.out,
        value_col=args.value_col, log_color=args.log_color, title=args.title,
    )


if __name__ == "__main__":
    main()
