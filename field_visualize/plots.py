"""
电场可视化绘图：势场 1D/2D、阱频扫描 1D/2D
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np

from FieldConfiguration.constants import m as ION_MASS
from FieldParser.potential_fit import (
    eval_fit,
    fit_potential_1d,
    get_center_and_k2,
    k2_to_trap_freq_MHz,
)

from .core import (
    AXIS_INDEX,
    apply_offset_min,
    build_grid_1d,
    build_grid_2d,
    compute_potentials,
    norm_to_um,
    set_ylim_from_data,
    um_to_norm,
)

CoordAxis = Literal["x", "y", "z"]
_DcRfMode = Literal["both", "dc_only", "pseudo_only", "both_zero"]


def _max_abs_finite(V: np.ndarray) -> float:
    a = np.asarray(V)
    m = a[np.isfinite(a)]
    if m.size == 0:
        return 0.0
    return float(np.max(np.abs(m)))


def _classify_dc_rf_coverage(
    V_dc: np.ndarray, V_pseudo: np.ndarray, *, eps: float = 1e-12
) -> _DcRfMode:
    """按当前采样网格判断 DC / RF 赝势是否可忽略（|V|<=eps 视为全空间为 0）。"""
    dc_m = _max_abs_finite(V_dc)
    ps_m = _max_abs_finite(V_pseudo)
    dc_z = dc_m <= eps
    ps_z = ps_m <= eps
    if dc_z and ps_z:
        return "both_zero"
    if dc_z:
        return "pseudo_only"
    if ps_z:
        return "dc_only"
    return "both"


def _note_for_dc_rf_mode(mode: _DcRfMode) -> str | None:
    if mode == "pseudo_only":
        return "Note: static potential (DC) is zero everywhere on this grid."
    if mode == "dc_only":
        return "Note: RF pseudopotential is zero everywhere on this grid."
    if mode == "both_zero":
        return "Note: DC and RF pseudopotential are both zero on this grid."
    return None


def _index_argmin_valid_1d(V: np.ndarray) -> int | None:
    V = np.asarray(V, dtype=float)
    m = np.isfinite(V)
    if not np.any(m):
        return None
    masked = np.where(m, V, np.inf)
    return int(np.argmin(masked))


def _index_argmin_valid_2d(V: np.ndarray) -> tuple[int, int] | None:
    V = np.asarray(V, dtype=float)
    m = np.isfinite(V)
    if not np.any(m):
        return None
    masked = np.where(m, V, np.inf)
    flat = int(np.argmin(masked.ravel()))
    ij = np.unravel_index(flat, V.shape)
    return int(ij[0]), int(ij[1])


def _col_for_total_min_marker(
    col_defs: list[tuple[str, tuple[float, float], int]],
) -> int:
    """Column index where total potential (idx 3) is drawn; else last column."""
    for col, (_title, _vm, idx) in enumerate(col_defs):
        if idx == 3:
            return col
    return len(col_defs) - 1


def _mark_total_min_1d(
    ax,
    coord_um: np.ndarray,
    V_total: np.ndarray,
    vary_axis: CoordAxis,
) -> None:
    i = _index_argmin_valid_1d(V_total)
    if i is None:
        return
    x_m = float(coord_um[i])
    v_m = float(V_total[i])
    ax.axvline(x_m, color="black", linestyle="--", alpha=0.45, linewidth=1.0)
    ax.scatter(
        [x_m],
        [v_m],
        s=85,
        c="magenta",
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
    )
    ax.annotate(
        f"min total: {vary_axis}={x_m:.3f} μm, V={v_m:.4f} V",
        xy=(x_m, v_m),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.8},
    )


def _mark_total_min_2d_heatmap(
    ax,
    V_tot: np.ndarray,
    cc1_um: np.ndarray,
    cc2_um: np.ndarray,
    lab1: str,
    lab2: str,
) -> None:
    ij = _index_argmin_valid_2d(V_tot)
    if ij is None:
        return
    i, j = ij
    u1 = float(cc1_um[i, j])
    u2 = float(cc2_um[i, j])
    v = float(V_tot[i, j])
    ax.scatter(
        [u1],
        [u2],
        s=100,
        c="magenta",
        edgecolors="black",
        linewidths=0.7,
        zorder=10,
    )
    ax.annotate(
        f"min total: {lab1}={u1:.2f}, {lab2}={u2:.2f} μm\nV={v:.4f} V",
        xy=(u1, u2),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.8},
    )


def _mark_total_min_2d_surface(
    ax,
    V_tot: np.ndarray,
    cc1_um: np.ndarray,
    cc2_um: np.ndarray,
    lab1: str,
    lab2: str,
) -> None:
    ij = _index_argmin_valid_2d(V_tot)
    if ij is None:
        return
    i, j = ij
    u1 = float(cc1_um[i, j])
    u2 = float(cc2_um[i, j])
    v = float(V_tot[i, j])
    ax.scatter(
        [u1],
        [u2],
        [v],
        s=90,
        c="magenta",
        edgecolors="black",
        linewidths=0.6,
        zorder=10,
    )
    zlim = ax.get_zlim3d()
    span = float(zlim[1] - zlim[0])
    dz = 0.02 * span if np.isfinite(span) and span > 0 else 0.01
    ax.text(
        u1,
        u2,
        v + dz,
        f"min total: {lab1}={u1:.2f}, {lab2}={u2:.2f} μm\nV={v:.4f} V",
        fontsize=7,
        color="black",
    )


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
    mark_potential_min: bool = False,
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

    decomp_mode = _classify_dc_rf_coverage(V_dc, V_pseudo)
    decomp_note = _note_for_dc_rf_mode(decomp_mode)

    x_um = norm_to_um(x_const, cfg.dl)
    y_um = norm_to_um(y_const, cfg.dl)
    z_um = norm_to_um(z_const, cfg.dl)
    title_base = f"Potential vs {vary_axis} (x={x_um:.1f} μm, y={y_um:.1f} μm, z={z_um:.1f} μm)"

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    y_series: list[np.ndarray] = []
    if decomp_mode == "both":
        ax1.plot(coord_um, V_dc, label="Static potential (DC)", color="blue")
        ax1.plot(coord_um, V_pseudo, label="RF pseudopotential", color="green")
        ax1.plot(coord_um, V_total, label="Total potential", color="red", linestyle="--")
        y_series = [V_dc, V_pseudo, V_total]
    elif decomp_mode == "dc_only":
        ax1.plot(coord_um, V_dc, label="Static potential (DC)", color="blue")
        y_series = [V_dc]
    elif decomp_mode == "pseudo_only":
        ax1.plot(coord_um, V_pseudo, label="RF pseudopotential", color="green")
        y_series = [V_pseudo]
    else:
        ax1.plot(coord_um, V_total, label="Total potential", color="red", linestyle="--")
        y_series = [V_total]

    fit_labels: list[str] = []
    trap_freq_note: str | None = None
    if fit_degree is not None and fit_degree in (2, 4):
        fit_specs: list[tuple[np.ndarray, str, str]] = []
        if decomp_mode == "both":
            fit_specs = [
                (V_dc, "DC fit", "blue"),
                (V_pseudo, "Pseudopotential fit", "green"),
                (V_total, "Total fit", "red"),
            ]
        elif decomp_mode == "dc_only":
            fit_specs = [(V_dc, "DC fit", "blue")]
        elif decomp_mode == "pseudo_only":
            fit_specs = [(V_pseudo, "Pseudopotential fit", "green")]
        else:
            fit_specs = [(V_total, "Total fit", "red")]
        for V_arr, label, color in fit_specs:
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
                if (
                    label == "Total fit"
                    and fit_degree == 2
                    and k2 > 0
                ):
                    f_mhz = k2_to_trap_freq_MHz(k2, ION_MASS)
                    if np.isfinite(f_mhz):
                        trap_freq_note = (
                            f"Trap frequency along {vary_axis}: {f_mhz:.4f} MHz"
                        )
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                pass
        if fit_labels:
            _t = f"{title_base} — Potentials\n" + "; ".join(fit_labels)
            if decomp_note:
                _t += "\n" + decomp_note
            ax1.set_title(_t, fontsize=9)
        if trap_freq_note is not None:
            ax1.text(
                0.02,
                0.98,
                trap_freq_note,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "red",
                    "alpha": 0.9,
                },
            )

    ax1.set_xlabel(f"{vary_axis} (μm)")
    ax1.set_ylabel("Potential (V)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if not fit_labels:
        _main = f"{title_base} — Potentials"
        if decomp_note:
            _main += "\n" + decomp_note
        ax1.set_title(_main)
    set_ylim_from_data(ax1, np.concatenate(y_series))
    if mark_potential_min:
        _mark_total_min_1d(ax1, coord_um, V_total, vary_axis)
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


def plot_bilayer(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    y0_um: float,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    n_pts: tuple[int, int] = (100, 100),
    mode: Literal["heatmap", "3d"] = "heatmap",
    offset_min: bool = False,
    show_rf_amp: bool = False,
    out_path: str | None = None,
    mark_potential_min: bool = False,
) -> None:
    """
    在 y = ±y0 的 zox 平面上去采样并绘制静电势、赝势、总电势。
    复用 build_grid_2d / compute_potentials / apply_offset_min，布局为两行（±y0）× 三列（DC / pseudo / total）。
    图中横轴为 z、纵轴为 x；n_pts 仍为 (nx, nz)。
    """
    import matplotlib.pyplot as plt

    nx_pts, nz_pts = n_pts[0], n_pts[1]
    y_pos_norm = um_to_norm(y0_um, cfg.dl)
    y_neg_norm = um_to_norm(-y0_um, cfg.dl)
    # 先 z 后 x：pcolormesh / plot_surface 的 X→横向 z，Y→纵向 x
    vary_axes: tuple[CoordAxis, CoordAxis] = ("z", "x")

    slices: list[tuple[float, float]] = [
        (y0_um, y_pos_norm),
        (-y0_um, y_neg_norm),
    ]

    cc1_um: np.ndarray | None = None
    cc2_um: np.ndarray | None = None
    layers_data: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    for _y_um, y_const_norm in slices:
        r, (cc1, cc2) = build_grid_2d(
            vary_axes,
            z_range,
            x_range,
            y_const_norm,
            (nz_pts, nx_pts),
        )
        V_dc, V_rf_amp, V_pseudo, V_total = compute_potentials(
            potential_interps, field_interps, voltage_list, cfg, r
        )
        if offset_min:
            V_dc, V_pseudo, V_total = apply_offset_min(V_dc, V_pseudo, V_total)
        if cc1_um is None:
            cc1_um = norm_to_um(cc1, cfg.dl)
            cc2_um = norm_to_um(cc2, cfg.dl)
        layers_data.append(
            (
                V_dc.reshape(cc1.shape),
                V_rf_amp.reshape(cc1.shape),
                V_pseudo.reshape(cc1.shape),
                V_total.reshape(cc1.shape),
            )
        )

    assert cc1_um is not None and cc2_um is not None

    def _vmin_vmax(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        stacked = np.concatenate([a.ravel(), b.ravel()])
        valid = stacked[~np.isnan(stacked)]
        if valid.size == 0:
            return 0.0, 1.0
        return float(np.min(valid)), float(np.max(valid))

    V_dc_p, V_rf_p, V_ps_p, V_tot_p = layers_data[0]
    V_dc_n, V_rf_n, V_ps_n, V_tot_n = layers_data[1]
    vm_dc = _vmin_vmax(V_dc_p, V_dc_n)
    vm_ps = _vmin_vmax(V_ps_p, V_ps_n)
    vm_tot = _vmin_vmax(V_tot_p, V_tot_n)
    vm_rf = _vmin_vmax(V_rf_p, V_rf_n)

    dc_stack = np.concatenate([V_dc_p.ravel(), V_dc_n.ravel()])
    ps_stack = np.concatenate([V_ps_p.ravel(), V_ps_n.ravel()])
    decomp_mode = _classify_dc_rf_coverage(dc_stack, ps_stack)
    decomp_note = _note_for_dc_rf_mode(decomp_mode)

    if decomp_mode == "both":
        col_defs: list[tuple[str, tuple[float, float], int]] = [
            ("Static potential", vm_dc, 0),
            ("RF pseudopotential", vm_ps, 2),
            ("Total potential", vm_tot, 3),
        ]
    elif decomp_mode == "dc_only":
        col_defs = [("Static potential", vm_dc, 0)]
    elif decomp_mode == "pseudo_only":
        col_defs = [("RF pseudopotential", vm_ps, 2)]
    else:
        col_defs = [("Total potential", vm_tot, 3)]

    ncols = len(col_defs)
    suptitle_base = f"Potential in x–z plane at y = ±{abs(y0_um):.3g} μm"
    suptitle_full = suptitle_base + (f"\n{decomp_note}" if decomp_note else "")
    fig_w = 5 * ncols + 2
    mark_col = _col_for_total_min_marker(col_defs)

    if mode == "heatmap":
        fig, axes = plt.subplots(2, ncols, figsize=(fig_w, 9), squeeze=False)
        for row, (y_um, _ync), pack in zip(range(2), slices, layers_data):
            y_label = f"y = {y_um:+.3g} μm"
            V_tot_layer = pack[3]
            for col, (title_h, (vmin, vmax), idx) in enumerate(col_defs):
                ax = axes[row, col]
                data = pack[idx]
                im = ax.pcolormesh(
                    cc1_um,
                    cc2_um,
                    data,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                )
                cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
                cbar.set_label("V")
                ax.set_xlabel("z (μm)")
                ax.set_ylabel("x (μm)")
                ax.set_aspect("equal")
                ax.set_title(f"{title_h} ({y_label})")
                if mark_potential_min and col == mark_col:
                    _mark_total_min_2d_heatmap(ax, V_tot_layer, cc1_um, cc2_um, "z", "x")
        fig.suptitle(suptitle_full)
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _save_or_show(None, "_potentials", fig, out_path)
        plt.close(fig)

        if show_rf_amp:
            fig2, axes2 = plt.subplots(2, 1, figsize=(7, 10))
            for ax, (y_um, _ync), V_rf_2d in zip(
                axes2,
                slices,
                (V_rf_p, V_rf_n),
            ):
                im = ax.pcolormesh(
                    cc1_um,
                    cc2_um,
                    V_rf_2d,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=vm_rf[0],
                    vmax=vm_rf[1],
                )
                cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
                cbar.set_label("V")
                ax.set_xlabel("z (μm)")
                ax.set_ylabel("x (μm)")
                ax.set_aspect("equal")
                y_label = f"y = {y_um:+.3g} μm"
                ax.set_title(f"RF amplitude ({y_label})")
            fig2.suptitle(suptitle_full)
            plt.tight_layout()
            _save_or_show(None, "_rf_amp", fig2, out_path)
            plt.close(fig2)
    else:
        fig = plt.figure(figsize=(fig_w, 9))
        for row, (y_um, _ync), pack in zip(range(2), slices, layers_data):
            y_label = f"y = {y_um:+.3g} μm"
            V_tot_layer = pack[3]
            for col, (title_h, (vmin, vmax), idx) in enumerate(col_defs):
                ax = fig.add_subplot(2, ncols, row * ncols + col + 1, projection="3d")
                data = pack[idx]
                surf = ax.plot_surface(
                    cc1_um,
                    cc2_um,
                    data,
                    cmap="RdBu_r",
                    alpha=0.9,
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(surf, ax=ax, shrink=0.5, label="V")
                ax.set_xlabel("z (μm)")
                ax.set_ylabel("x (μm)")
                ax.set_zlabel("Potential (V)")
                ax.set_title(f"{title_h} ({y_label})")
                if mark_potential_min and col == mark_col:
                    _mark_total_min_2d_surface(ax, V_tot_layer, cc1_um, cc2_um, "z", "x")
        fig.suptitle(suptitle_full)
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _save_or_show(None, "_potentials", fig, out_path)
        plt.close(fig)

        if show_rf_amp:
            fig2 = plt.figure(figsize=(fig_w, 9))
            for row, (y_um, _ync), V_rf_2d in zip(
                range(2),
                slices,
                (V_rf_p, V_rf_n),
            ):
                ax = fig2.add_subplot(2, 1, row + 1, projection="3d")
                surf = ax.plot_surface(
                    cc1_um,
                    cc2_um,
                    V_rf_2d,
                    cmap="RdBu_r",
                    alpha=0.9,
                    vmin=vm_rf[0],
                    vmax=vm_rf[1],
                )
                fig2.colorbar(surf, ax=ax, shrink=0.5, label="V")
                ax.set_xlabel("z (μm)")
                ax.set_ylabel("x (μm)")
                ax.set_zlabel("Potential (V)")
                y_label = f"y = {y_um:+.3g} μm"
                ax.set_title(f"RF amplitude ({y_label})")
            fig2.suptitle(suptitle_full)
            plt.tight_layout()
            _save_or_show(None, "_rf_amp", fig2, out_path)
            plt.close(fig2)


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
    mark_potential_min: bool = False,
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

    decomp_mode = _classify_dc_rf_coverage(V_dc, V_pseudo)
    decomp_note = _note_for_dc_rf_mode(decomp_mode)

    other = next(c for c in "xyz" if c not in vary_axes)
    suptitle_base = f"Potential distribution ({other}={const_um:.1f} μm)"
    suptitle_full = (
        suptitle_base + (f"\n{decomp_note}" if decomp_note else "")
    )

    if mode == "heatmap":
        if decomp_mode == "both":
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
            if mark_potential_min:
                _mark_total_min_2d_heatmap(
                    axes_flat[2], V_total_2d, cc1_um, cc2_um, a1, a2
                )
        else:
            fig, ax = plt.subplots(figsize=(6, 5.5))
            if decomp_mode == "dc_only":
                data, title = V_dc_2d, "Static potential"
            elif decomp_mode == "pseudo_only":
                data, title = V_pseudo_2d, "RF pseudopotential"
            else:
                data, title = V_total_2d, "Total potential"
            im = ax.pcolormesh(cc1_um, cc2_um, data, shading="auto", cmap="RdBu_r")
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
            cbar.set_label("V")
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_aspect("equal")
            ax.set_title(title)
            if mark_potential_min:
                _mark_total_min_2d_heatmap(ax, V_total_2d, cc1_um, cc2_um, a1, a2)
        fig.suptitle(suptitle_full)
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
            fig2.suptitle(suptitle_full)
            plt.tight_layout()
            _save_or_show(None, "_rf_amp", fig2, out_path)
            plt.close(fig2)
    else:
        if decomp_mode == "both":
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
                if mark_potential_min and title == "Total potential":
                    _mark_total_min_2d_surface(ax, V_total_2d, cc1_um, cc2_um, a1, a2)
            fig.add_subplot(2, 2, 4).set_visible(False)
        else:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
            if decomp_mode == "dc_only":
                data, title = V_dc_2d, "Static potential"
            elif decomp_mode == "pseudo_only":
                data, title = V_pseudo_2d, "RF pseudopotential"
            else:
                data, title = V_total_2d, "Total potential"
            ax.plot_surface(cc1_um, cc2_um, data, cmap="RdBu_r", alpha=0.9)
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_zlabel("Potential (V)")
            ax.set_title(title)
            if mark_potential_min:
                _mark_total_min_2d_surface(ax, V_total_2d, cc1_um, cc2_um, a1, a2)
        fig.suptitle(suptitle_full)
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
            fig2.suptitle(suptitle_full)
            plt.tight_layout()
            _save_or_show(None, "_rf_amp", fig2, out_path)
            plt.close(fig2)
