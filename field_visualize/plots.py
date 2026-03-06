"""
电场可视化绘图：势场 1D/2D、阱频扫描 1D/2D
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np

from FieldParser.potential_fit import eval_fit, fit_potential_1d, get_center_and_k2

from .core import (
    AXIS_INDEX,
    apply_offset_min,
    build_grid_1d,
    build_grid_2d,
    compute_potentials,
    norm_to_um,
    set_ylim_from_data,
)

CoordAxis = Literal["x", "y", "z"]


def _save_or_show(path: str | None, suffix: str, fig, out_path: str | None = None) -> None:
    import matplotlib.pyplot as plt

    path = path or out_path
    if path:
        base, ext = os.path.splitext(path)
        p = f"{base}{suffix}{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    else:
        plt.show()


def plot_freq_scan_1d(
    coord_um: np.ndarray,
    f_x: np.ndarray,
    f_y: np.ndarray,
    f_z: np.ndarray,
    scan_axis: CoordAxis,
    const_um: tuple[float, float, float],
    out_path: str | None = None,
) -> None:
    """
    绘制阱频沿单轴分布曲线。扫描方向阱频为常量不绘制，仅输出；只绘垂直于扫描方向的两个
    """
    import matplotlib.pyplot as plt

    all_data = {"f_x": f_x, "f_y": f_y, "f_z": f_z}
    f_scan_label = f"f_{scan_axis}"
    f_scan_val = all_data[f_scan_label][0]
    if np.isnan(f_scan_val):
        print(f"{f_scan_label} (扫描方向，常量): N/A (k2<=0 或拟合失败)")
    else:
        print(f"{f_scan_label} (扫描方向，常量): {f_scan_val:.4f} MHz")

    items = [
        (l, all_data[l], c)
        for l, c in [("f_x", "blue"), ("f_y", "green"), ("f_z", "red")]
        if l != f_scan_label
    ]
    valid_items = [(l, arr, c) for l, arr, c in items if not np.all(np.isnan(arr))]
    nan_labels = [l for l, arr, _ in items if np.all(np.isnan(arr))]
    if nan_labels:
        print("Warning: 以下阱频全为 NaN，已跳过绘制: " + ", ".join(nan_labels))

    if not valid_items:
        return

    title_base = f"Trap frequency vs {scan_axis} (x={const_um[0]:.1f}, y={const_um[1]:.1f}, z={const_um[2]:.1f} μm)"
    n_plot = len(valid_items)

    if n_plot == 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, (label, arr, color) in enumerate(valid_items):
        ax = axes[idx]
        ax.plot(coord_um, arr, color=color)
        ax.set_xlabel(f"{scan_axis} (μm)")
        ax.set_ylabel("Trap frequency (MHz)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        valid = arr[~np.isnan(arr)]
        if valid.size:
            set_ylim_from_data(ax, valid)

    fig.suptitle(title_base)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _save_or_show(None, "_freq_scan", fig, out_path)
    plt.close(fig)


def plot_freq_scan_2d(
    cc1_um: np.ndarray,
    cc2_um: np.ndarray,
    f_x_2d: np.ndarray,
    f_y_2d: np.ndarray,
    f_z_2d: np.ndarray,
    scan_axes: tuple[CoordAxis, CoordAxis],
    const_um: float,
    const_axis: CoordAxis,
    mode: Literal["heatmap", "3d"],
    out_path: str | None = None,
) -> None:
    """
    绘制阱频沿两轴分布。扫描方向 f_a1、f_a2 仅在垂直方向变化，不绘 2D 图，仅输出范围；
    只绘 f_const（固定轴方向）的 2D 分布
    """
    import matplotlib.pyplot as plt

    a1, a2 = scan_axes
    all_data = {"f_x": f_x_2d, "f_y": f_y_2d, "f_z": f_z_2d}
    f_const_label = f"f_{const_axis}"

    for ax_name, desc in [(a1, f"沿 {a2} 变化"), (a2, f"沿 {a1} 变化")]:
        lab = f"f_{ax_name}"
        arr = all_data[lab]
        valid = arr[~np.isnan(arr)]
        if valid.size:
            print(f"{lab} (扫描方向 {desc}): {np.min(valid):.4f} ~ {np.max(valid):.4f} MHz")
        else:
            print(f"{lab} (扫描方向 {desc}): N/A")

    data = all_data[f_const_label]
    if np.all(np.isnan(data)):
        print(f"Warning: {f_const_label} 全为 NaN，已跳过绘制")
        return

    suptitle = f"Trap frequency {f_const_label} ({const_axis}={const_um:.1f} μm)"
    valid = data[~np.isnan(data)]
    vmin, vmax = (np.min(valid), np.max(valid)) if valid.size else (0, 1)

    if mode == "heatmap":
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(
            cc1_um, cc2_um, data, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        plt.colorbar(im, ax=ax, label="MHz")
        ax.set_xlabel(f"{a1} (μm)")
        ax.set_ylabel(f"{a2} (μm)")
        ax.set_aspect("equal")
        ax.set_title(f_const_label)
    else:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(cc1_um, cc2_um, data, cmap="viridis", alpha=0.9)
        ax.set_xlabel(f"{a1} (μm)")
        ax.set_ylabel(f"{a2} (μm)")
        ax.set_zlabel("MHz")
        ax.set_title(f_const_label)

    fig.suptitle(suptitle)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _save_or_show(None, "_freq_scan", fig, out_path)
    plt.close(fig)


def plot_1d(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    vary_axis: CoordAxis,
    vary_range: tuple[float, float],
    x_const: float = 0.0,
    y_const: float = 0.0,
    z_const: float = 0.0,
    n_pts: int = 500,
    offset_min: bool = False,
    show_rf_amp: bool = False,
    fit_degree: int | None = None,
    out_path: str | None = None,
) -> None:
    """1D 绘图：电势随单坐标变化"""
    import matplotlib.pyplot as plt

    r = build_grid_1d(vary_axis, vary_range, x_const, y_const, z_const, n_pts)
    idx = AXIS_INDEX[vary_axis]
    coord_norm = r[:, idx]
    coord_um = norm_to_um(coord_norm, cfg.dl)

    V_dc, V_rf_amp, V_pseudo, V_total = compute_potentials(
        potential_interps, field_interps, voltage_list, cfg, r
    )
    if offset_min:
        V_dc, V_pseudo, V_total = apply_offset_min(V_dc, V_pseudo, V_total)

    x_um = norm_to_um(x_const, cfg.dl)
    y_um = norm_to_um(y_const, cfg.dl)
    z_um = norm_to_um(z_const, cfg.dl)
    title_base = f"Potential vs {vary_axis} (x={x_um:.1f} μm, y={y_um:.1f} μm, z={z_um:.1f} μm)"

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(coord_um, V_dc, label="Static potential (DC)", color="blue")
    ax1.plot(coord_um, V_pseudo, label="RF pseudopotential", color="green")
    ax1.plot(coord_um, V_total, label="Total potential", color="red", linestyle="--")

    fit_labels: list[str] = []
    if fit_degree is not None and fit_degree in (2, 4):
        for V_arr, label, color in [
            (V_dc, "DC fit", "blue"),
            (V_pseudo, "Pseudopotential fit", "green"),
            (V_total, "Total fit", "red"),
        ]:
            try:
                fit_result, r2 = fit_potential_1d(coord_um, V_arr, degree=fit_degree)
                V_fit = eval_fit(coord_um, fit_result, fit_degree)
                ax1.plot(
                    coord_um,
                    V_fit,
                    linestyle=":",
                    color=color,
                    alpha=0.8,
                    label=f"{label} (R²={r2:.4f})",
                )
                center, k2 = get_center_and_k2(fit_result, fit_degree)
                fit_labels.append(f"{label}: center={center:.1f} μm, k2={k2:.2e}")
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                pass
        if fit_labels:
            ax1.set_title(
                f"{title_base} — Potentials\n" + "; ".join(fit_labels), fontsize=9
            )

    ax1.set_xlabel(f"{vary_axis} (μm)")
    ax1.set_ylabel("Potential (V)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if not fit_labels:
        ax1.set_title(f"{title_base} — Potentials")
    set_ylim_from_data(ax1, np.concatenate([V_dc, V_pseudo, V_total]))
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _save_or_show(None, "_potentials", fig1, out_path)
    plt.close()

    if show_rf_amp:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(coord_um, V_rf_amp, label="RF amplitude", color="orange")
        ax2.set_xlabel(f"{vary_axis} (μm)")
        ax2.set_ylabel("Potential (V)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"{title_base} — RF amplitude")
        set_ylim_from_data(ax2, V_rf_amp)
        plt.tight_layout()
        _save_or_show(None, "_rf_amp", fig2, out_path)
        plt.close()


def plot_2d(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    vary_axes: tuple[CoordAxis, CoordAxis],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    const_val: float = 0.0,
    n_pts: tuple[int, int] = (100, 100),
    mode: Literal["heatmap", "3d"] = "heatmap",
    offset_min: bool = False,
    show_rf_amp: bool = False,
    out_path: str | None = None,
) -> None:
    """2D 绘图：热力图或三维曲面"""
    import matplotlib.pyplot as plt

    r, (cc1, cc2) = build_grid_2d(vary_axes, x_range, y_range, const_val, n_pts)
    V_dc, V_rf_amp, V_pseudo, V_total = compute_potentials(
        potential_interps, field_interps, voltage_list, cfg, r
    )
    if offset_min:
        V_dc, V_pseudo, V_total = apply_offset_min(V_dc, V_pseudo, V_total)
    a1, a2 = vary_axes
    cc1_um = norm_to_um(cc1, cfg.dl)
    cc2_um = norm_to_um(cc2, cfg.dl)
    const_um = norm_to_um(const_val, cfg.dl)
    V_dc_2d = V_dc.reshape(cc1.shape)
    V_rf_2d = V_rf_amp.reshape(cc1.shape)
    V_pseudo_2d = V_pseudo.reshape(cc1.shape)
    V_total_2d = V_total.reshape(cc1.shape)

    other = next(c for c in "xyz" if c not in vary_axes)
    suptitle_base = f"Potential distribution ({other}={const_um:.1f} μm)"

    if mode == "heatmap":
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0]]
        axes[1, 1].set_visible(False)
        for ax, data, title in zip(
            axes_flat,
            [V_dc_2d, V_pseudo_2d, V_total_2d],
            ["Static potential", "RF pseudopotential", "Total potential"],
        ):
            im = ax.pcolormesh(cc1_um, cc2_um, data, shading="auto", cmap="RdBu_r")
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
            cbar.set_label("V")
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_aspect("equal")
            ax.set_title(title)
        fig.suptitle(suptitle_base)
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _save_or_show(None, "_potentials", fig, out_path)
        plt.close(fig)

        if show_rf_amp:
            fig2, ax = plt.subplots(figsize=(6, 5))
            im = ax.pcolormesh(cc1_um, cc2_um, V_rf_2d, shading="auto", cmap="RdBu_r")
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
            cbar.set_label("V")
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_aspect("equal")
            ax.set_title("RF amplitude")
            fig2.suptitle(suptitle_base)
            plt.tight_layout()
            _save_or_show(None, "_rf_amp", fig2, out_path)
            plt.close(fig2)
    else:
        fig = plt.figure(figsize=(10, 10))
        for i, (data, title) in enumerate(
            [
                (V_dc_2d, "Static potential"),
                (V_pseudo_2d, "RF pseudopotential"),
                (V_total_2d, "Total potential"),
            ]
        ):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            ax.plot_surface(cc1_um, cc2_um, data, cmap="RdBu_r", alpha=0.9)
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_zlabel("Potential (V)")
            ax.set_title(title)
        fig.add_subplot(2, 2, 4).set_visible(False)
        fig.suptitle(suptitle_base)
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _save_or_show(None, "_potentials", fig, out_path)
        plt.close(fig)

        if show_rf_amp:
            fig2 = plt.figure(figsize=(7, 5))
            ax = fig2.add_subplot(111, projection="3d")
            ax.plot_surface(cc1_um, cc2_um, V_rf_2d, cmap="RdBu_r", alpha=0.9)
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_zlabel("Potential (V)")
            ax.set_title("RF amplitude")
            fig2.suptitle(suptitle_base)
            plt.tight_layout()
            _save_or_show(None, "_rf_amp", fig2, out_path)
            plt.close(fig2)
