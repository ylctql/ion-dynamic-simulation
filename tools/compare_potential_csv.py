#!/usr/bin/env python3
"""
比较两个电场 CSV 在给定电压配置下的总势场（静电 + RF 赝势）差异。

在统一的无量纲尺度（由主 config 的 RF 基准频率决定 dl、dV、Omega）下读取两个 CSV，
分别建立插值并计算总势，在同一采样网格上统计 |ΔV| 的最大值与平均值，并绘图。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from FieldConfiguration.constants import init_from_config
from FieldConfiguration.loader import field_settings_from_config, load_field_config
from FieldParser.calc_field import calc_field, calc_potential
from FieldParser.csv_reader import read as read_csv
from field_visualize.core import (
    AXIS_INDEX,
    apply_offset_min,
    build_grid_1d,
    build_grid_2d,
    compute_potentials,
    norm_to_um,
    um_to_norm,
)
from field_visualize.plots import CoordAxis

from Interface.cli import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH


def _resolve_path(arg: str, default_full: str, default_dir: str) -> str:
    if not arg:
        return str(_ROOT / default_full)
    p = Path(arg)
    if not p.is_absolute() and "/" not in arg and "\\" not in arg:
        return str(_ROOT / default_dir / arg)
    return str(_ROOT / arg) if not p.is_absolute() else arg


def _parse_range(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a.strip()), float(b.strip())


def _parse_const(s: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--const must be three numbers x,y,z (μm)")
    return parts[0], parts[1], parts[2]


def _parse_n_pts(s: str, dim: int):
    if not s.strip():
        return 500 if dim == 1 else (100, 100)
    parts = [p.strip() for p in s.split(",")]
    if dim == 1:
        if len(parts) != 1:
            raise ValueError("For 1D --n_pts must be a single integer (e.g. 500)")
        return int(parts[0])
    if len(parts) != 2:
        raise ValueError("For 2D --n_pts must be two integers (e.g. 100,100)")
    return int(parts[0]), int(parts[1])


def _ref_freq_mhz_from_config_dict(config: dict) -> float:
    from FieldConfiguration.constants import _get_ref_freq_from_config

    return _get_ref_freq_from_config(config)


def _apply_smooth(grid_coord, grid_voltage, raw_smooth: str, smooth_sg: str):
    if raw_smooth.strip().lower() == "none":
        return grid_voltage
    from field_visualize.core import apply_savgol_smooth

    axes_parts = [a.strip().lower() for a in raw_smooth.split(",") if a.strip()]
    valid_axes = [a for a in axes_parts if a in "xyz"]
    if not valid_axes:
        return grid_voltage
    try:
        sg_parts = [p.strip() for p in smooth_sg.split(",")]
        wl = int(sg_parts[0]) if sg_parts else 11
        poly = int(sg_parts[1]) if len(sg_parts) >= 2 else 3
    except (ValueError, IndexError):
        wl, poly = 11, 3
    return apply_savgol_smooth(
        grid_coord, grid_voltage, tuple(valid_axes), window_length=wl, polyorder=poly
    )


def _load_one_side(
    csv_path: str,
    config_path: str,
    cfg,
    *,
    smooth_axes: str,
    smooth_sg: str,
) -> tuple:
    """Returns grid_coord, grid_voltage, potential_interps, field_interps, voltage_list."""
    grid_coord, grid_voltage = read_csv(
        Path(csv_path), None, normalize=True, dl=cfg.dl, dV=cfg.dV
    )
    grid_voltage = _apply_smooth(grid_coord, grid_voltage, smooth_axes, smooth_sg)
    n_voltage = grid_voltage.shape[1]
    fs = field_settings_from_config(csv_path, config_path, n_voltage, cfg)
    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)
    return (
        grid_coord,
        grid_voltage,
        potential_interps,
        field_interps,
        fs.voltage_list,
    )


def _diff_stats(diff_v: np.ndarray) -> tuple[float, float, int]:
    valid = diff_v[np.isfinite(diff_v)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    return (
        float(np.max(np.abs(valid))),
        float(np.mean(np.abs(valid))),
        int(valid.size),
    )


def _plot_compare_1d(
    coord_um: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
    v_diff: np.ndarray,
    vary_axis: CoordAxis,
    title_suffix: str,
    out_path: str | None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), height_ratios=[2, 1])
    ax0, ax1 = axes
    ax0.plot(coord_um, v_a, label="Total (CSV A)", color="C0")
    ax0.plot(coord_um, v_b, label="Total (CSV B)", color="C1", alpha=0.85)
    ax0.set_ylabel("Potential (V)")
    ax0.set_title(f"Total potential vs {vary_axis}{title_suffix}")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)

    ax1.plot(coord_um, v_diff, label="Difference (A − B)", color="C2")
    ax1.set_xlabel(f"{vary_axis} (μm)")
    ax1.set_ylabel("ΔV (V)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def _plot_compare_2d(
    cc1_um: np.ndarray,
    cc2_um: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
    v_diff: np.ndarray,
    vary_axes: tuple[CoordAxis, CoordAxis],
    const_um: float,
    const_axis: str,
    mode: Literal["heatmap", "3d"],
    out_path: str | None,
) -> None:
    import matplotlib.pyplot as plt

    a1, a2 = vary_axes
    suptitle = f"Total potential & difference ({const_axis}={const_um:.1f} μm)"

    if mode == "heatmap":
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
        v_all = np.concatenate([v_a.ravel(), v_b.ravel()])
        v_all = v_all[np.isfinite(v_all)]
        if v_all.size:
            vmin, vmax = float(np.min(v_all)), float(np.max(v_all))
        else:
            vmin, vmax = 0.0, 1.0
        d_valid = v_diff[np.isfinite(v_diff)]
        if d_valid.size:
            dmax = float(np.max(np.abs(d_valid)))
        else:
            dmax = 1.0
        for ax, data, title, cmap, dvmin, dvmax in zip(
            axes,
            (v_a, v_b, v_diff),
            ("Total (CSV A)", "Total (CSV B)", "Difference (A − B)"),
            ("RdBu_r", "RdBu_r", "RdBu_r"),
            (vmin, vmin, -dmax),
            (vmax, vmax, dmax),
        ):
            im = ax.pcolormesh(
                cc1_um, cc2_um, data, shading="auto", cmap=cmap, vmin=dvmin, vmax=dvmax
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="7%", pad=0.28)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_label("V")
            cb.ax.tick_params(labelsize=8)
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_aspect("equal")
            ax.set_title(title)
        fig.suptitle(suptitle, y=1.02)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96], pad=0.6, w_pad=1.4)
    else:
        fig = plt.figure(figsize=(15, 5.8))
        v_all = np.concatenate([v_a.ravel(), v_b.ravel()])
        v_all = v_all[np.isfinite(v_all)]
        if v_all.size:
            vmin, vmax = float(np.min(v_all)), float(np.max(v_all))
        else:
            vmin, vmax = 0.0, 1.0
        d_valid = v_diff[np.isfinite(v_diff)]
        dmax = float(np.max(np.abs(d_valid))) if d_valid.size else 1.0
        for i, (data, title, tmin, tmax) in enumerate(
            (
                (v_a, "Total (CSV A)", vmin, vmax),
                (v_b, "Total (CSV B)", vmin, vmax),
                (v_diff, "Difference (A − B)", -dmax, dmax),
            )
        ):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            surf = ax.plot_surface(
                cc1_um, cc2_um, data, cmap="RdBu_r", alpha=0.9, vmin=tmin, vmax=tmax
            )
            cb = fig.colorbar(
                surf,
                ax=ax,
                orientation="horizontal",
                shrink=0.78,
                fraction=0.05,
                pad=0.12,
                aspect=22,
            )
            cb.set_label("V")
            cb.ax.tick_params(labelsize=8)
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_zlabel("Potential (V)")
            ax.set_title(title, pad=10)
        fig.suptitle(suptitle, y=0.98)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.14, wspace=0.28)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare total potential (static + RF pseudopotential) from two field CSVs "
            "using the same dimensionless scaling (primary --config)."
        )
    )
    parser.add_argument("--csv-a", type=str, required=True, help="First field CSV path")
    parser.add_argument("--csv-b", type=str, required=True, help="Second field CSV path")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Voltage JSON for CSV A (and CSV B if --config-b omitted); sets dl, dV, Omega",
    )
    parser.add_argument(
        "--config-b",
        type=str,
        default="",
        help="Optional separate voltage JSON for CSV B (same cfg scaling as --config)",
    )
    parser.add_argument(
        "--vary",
        type=str,
        default="x",
        help="1D: x/y/z; 2D: comma-separated pair (e.g. x,y). Second axis range uses --y_range.",
    )
    parser.add_argument(
        "--x_range",
        type=str,
        default="-100,100",
        help="First varying axis range (μm), comma-separated",
    )
    parser.add_argument(
        "--y_range",
        type=str,
        default="-100,100",
        help="2D: second varying axis range (μm), comma-separated",
    )
    parser.add_argument("--const", type=str, default="0,0,0", help="Fixed x,y,z (μm)")
    parser.add_argument(
        "--n_pts",
        type=str,
        default="",
        help="1D: single int; 2D: nx,ny (default 500 or 100,100)",
    )
    parser.add_argument("--mode", type=str, choices=["heatmap", "3d"], default="heatmap")
    parser.add_argument("--out", type=str, default=None, help="Output figure path")
    parser.add_argument(
        "--offset",
        action="store_true",
        help="Subtract min from DC and pseudopotential separately per CSV (same as field_visualize)",
    )
    parser.add_argument(
        "--smooth-axes",
        type=str,
        default="z",
        metavar="AXES",
        help="Savitzky–Golay smoothing axes (default z); 'none' to disable",
    )
    parser.add_argument(
        "--smooth-sg",
        type=str,
        default="11,3",
        metavar="WINDOW,POLY",
        help="SG filter window and poly order",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_DIR)
    config_b_path = (
        _resolve_path(args.config_b, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_DIR)
        if args.config_b.strip()
        else config_path
    )

    csv_a = Path(args.csv_a)
    csv_b = Path(args.csv_b)
    if not csv_a.is_absolute():
        csv_a = _ROOT / csv_a
    if not csv_b.is_absolute():
        csv_b = _ROOT / csv_b

    cfg, _ = init_from_config(config_path)

    if (
        config_b_path != config_path
        and Path(config_b_path).exists()
        and Path(config_path).exists()
    ):
        cfg_b_dict = load_field_config(config_b_path)
        f_a = _ref_freq_mhz_from_config_dict(load_field_config(config_path))
        f_b = _ref_freq_mhz_from_config_dict(cfg_b_dict)
        if abs(f_a - f_b) > 1e-6:
            print(
                f"Warning: reference RF frequency differs ({f_a} vs {f_b} MHz). "
                f"Using dl, dV, Omega from primary --config only; ensure both CSVs are comparable.",
                file=sys.stderr,
            )

    (
        _,
        _,
        pot_a,
        fld_a,
        volt_a,
    ) = _load_one_side(
        str(csv_a),
        config_path,
        cfg,
        smooth_axes=args.smooth_axes,
        smooth_sg=args.smooth_sg,
    )
    (
        _,
        _,
        pot_b,
        fld_b,
        volt_b,
    ) = _load_one_side(
        str(csv_b),
        config_b_path,
        cfg,
        smooth_axes=args.smooth_axes,
        smooth_sg=args.smooth_sg,
    )

    xc_um, yc_um, zc_um = _parse_const(args.const)
    xr_um = _parse_range(args.x_range)
    yr_um = _parse_range(args.y_range)
    dl = cfg.dl
    xc = um_to_norm(xc_um, dl)
    yc = um_to_norm(yc_um, dl)
    zc = um_to_norm(zc_um, dl)
    xr = (um_to_norm(xr_um[0], dl), um_to_norm(xr_um[1], dl))

    vary_parts = [p.strip().lower() for p in args.vary.split(",")]
    if len(vary_parts) == 1:
        vary = vary_parts[0]
        if vary not in "xyz":
            raise ValueError("--vary must be x, y, or z")
        n_pts = _parse_n_pts(args.n_pts, 1)
        r = build_grid_1d(vary, xr, xc, yc, zc, n_pts)
        v_dc_a, _, v_ps_a, v_tot_a = compute_potentials(pot_a, fld_a, volt_a, cfg, r)
        v_dc_b, _, v_ps_b, v_tot_b = compute_potentials(pot_b, fld_b, volt_b, cfg, r)
        if args.offset:
            _, _, v_tot_a = apply_offset_min(v_dc_a, v_ps_a, v_tot_a)
            _, _, v_tot_b = apply_offset_min(v_dc_b, v_ps_b, v_tot_b)
        v_diff = v_tot_a - v_tot_b
        max_abs, mean_abs, n_valid = _diff_stats(v_diff)
        print(
            f"Difference (A − B): max |ΔV| = {max_abs:.6g} V, "
            f"mean |ΔV| = {mean_abs:.6g} V ({n_valid} points)"
        )
        idx = AXIS_INDEX[vary]
        coord_um = norm_to_um(r[:, idx], dl)
        title_suffix = (
            f" (fixed x={xc_um:g}, y={yc_um:g}, z={zc_um:g} μm)"
        )
        _plot_compare_1d(coord_um, v_tot_a, v_tot_b, v_diff, vary, title_suffix, args.out)
    elif len(vary_parts) == 2:
        a1, a2 = vary_parts[0], vary_parts[1]
        if a1 not in "xyz" or a2 not in "xyz" or a1 == a2:
            raise ValueError("--vary must be two distinct coordinates (e.g. x,y)")
        yr = (um_to_norm(yr_um[0], dl), um_to_norm(yr_um[1], dl))
        n_pts = _parse_n_pts(args.n_pts, 2)
        const_idx = next(i for i, c in enumerate("xyz") if c not in (a1, a2))
        const_axis = "xyz"[const_idx]
        const_val = [xc, yc, zc][const_idx]
        const_um = [xc_um, yc_um, zc_um][const_idx]
        r, (cc1, cc2) = build_grid_2d((a1, a2), xr, yr, const_val, n_pts)
        v_dc_a, _, v_ps_a, v_tot_a = compute_potentials(pot_a, fld_a, volt_a, cfg, r)
        v_dc_b, _, v_ps_b, v_tot_b = compute_potentials(pot_b, fld_b, volt_b, cfg, r)
        if args.offset:
            _, _, v_tot_a = apply_offset_min(v_dc_a, v_ps_a, v_tot_a)
            _, _, v_tot_b = apply_offset_min(v_dc_b, v_ps_b, v_tot_b)
        v_diff = v_tot_a - v_tot_b
        max_abs, mean_abs, n_valid = _diff_stats(v_diff)
        print(
            f"Difference (A − B): max |ΔV| = {max_abs:.6g} V, "
            f"mean |ΔV| = {mean_abs:.6g} V ({n_valid} points)"
        )
        cc1_um = norm_to_um(cc1, dl)
        cc2_um = norm_to_um(cc2, dl)
        sh = cc1.shape
        _plot_compare_2d(
            cc1_um,
            cc2_um,
            v_tot_a.reshape(sh),
            v_tot_b.reshape(sh),
            v_diff.reshape(sh),
            (a1, a2),
            const_um,
            const_axis,
            args.mode,
            args.out,
        )
    else:
        raise ValueError("--vary must be one coordinate (x/y/z) or two (e.g. x,y)")


if __name__ == "__main__":
    main()