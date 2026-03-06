"""
阱频计算：单点阱频、1D/2D 阱频扫描
"""
from __future__ import annotations

import numpy as np

from FieldConfiguration.constants import m as ION_MASS
from FieldParser.potential_fit import (
    fit_potential_1d,
    get_center_and_k2,
    k2_to_trap_freq_MHz,
)

from .core import AXIS_INDEX, build_grid_1d, compute_potentials, um_to_norm

CoordAxis = str  # "x" | "y" | "z"


def compute_trap_freqs_at_point(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    xc_um: float,
    yc_um: float,
    zc_um: float,
    x_range_um: tuple[float, float],
    y_range_um: tuple[float, float],
    z_range_um: tuple[float, float],
    n_pts: int,
    fit_degree: int,
) -> dict[str, float]:
    """在给定点处沿 x,y,z 分别拟合总势，计算阱频 f_x, f_y, f_z (MHz)"""
    dl = cfg.dl
    xc = um_to_norm(xc_um, dl)
    yc = um_to_norm(yc_um, dl)
    zc = um_to_norm(zc_um, dl)
    xr = (um_to_norm(x_range_um[0], dl), um_to_norm(x_range_um[1], dl))
    yr = (um_to_norm(y_range_um[0], dl), um_to_norm(y_range_um[1], dl))
    zr = (um_to_norm(z_range_um[0], dl), um_to_norm(z_range_um[1], dl))

    result: dict[str, float] = {}
    for axis, vary_range, coord_um in [
        ("x", xr, np.linspace(x_range_um[0], x_range_um[1], n_pts)),
        ("y", yr, np.linspace(y_range_um[0], y_range_um[1], n_pts)),
        ("z", zr, np.linspace(z_range_um[0], z_range_um[1], n_pts)),
    ]:
        r = build_grid_1d(axis, vary_range, xc, yc, zc, n_pts)
        _, _, _, V_total = compute_potentials(
            potential_interps, field_interps, voltage_list, cfg, r
        )
        try:
            fit_result, _ = fit_potential_1d(coord_um, V_total, degree=fit_degree)
            _, k2 = get_center_and_k2(fit_result, fit_degree)
            result[f"f_{axis}"] = k2_to_trap_freq_MHz(k2, ION_MASS)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            result[f"f_{axis}"] = float("nan")
    return result


def compute_freq_scan_1d(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    scan_axis: CoordAxis,
    scan_range_um: tuple[float, float],
    xc_um: float,
    yc_um: float,
    zc_um: float,
    x_range_um: tuple[float, float],
    y_range_um: tuple[float, float],
    z_range_um: tuple[float, float],
    n_scan: int,
    n_fit_pts: int,
    fit_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """沿单轴扫描阱频，返回 (coord_um, f_x, f_y, f_z)。扫描方向阱频为常量"""
    coord_um = np.linspace(scan_range_um[0], scan_range_um[1], n_scan)
    f_x = np.full(n_scan, np.nan)
    f_y = np.full(n_scan, np.nan)
    f_z = np.full(n_scan, np.nan)
    for i, c_um in enumerate(coord_um):
        pt = [xc_um, yc_um, zc_um]
        pt[AXIS_INDEX[scan_axis]] = c_um
        freqs = compute_trap_freqs_at_point(
            potential_interps,
            field_interps,
            voltage_list,
            cfg,
            pt[0],
            pt[1],
            pt[2],
            x_range_um=x_range_um,
            y_range_um=y_range_um,
            z_range_um=z_range_um,
            n_pts=n_fit_pts,
            fit_degree=fit_degree,
        )
        f_x[i] = freqs["f_x"]
        f_y[i] = freqs["f_y"]
        f_z[i] = freqs["f_z"]
    return coord_um, f_x, f_y, f_z


def compute_freq_scan_2d(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    scan_axes: tuple[CoordAxis, CoordAxis],
    x_range_um: tuple[float, float],
    y_range_um: tuple[float, float],
    z_range_um: tuple[float, float],
    xc_um: float,
    yc_um: float,
    zc_um: float,
    n_scan: tuple[int, int],
    n_fit_pts: int,
    fit_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """沿两轴扫描阱频，返回 (cc1_um, cc2_um, f_x_2d, f_y_2d, f_z_2d)"""
    a1, a2 = scan_axes
    r1 = x_range_um if a1 == "x" else (y_range_um if a1 == "y" else z_range_um)
    r2 = x_range_um if a2 == "x" else (y_range_um if a2 == "y" else z_range_um)
    c1 = np.linspace(r1[0], r1[1], n_scan[0])
    c2 = np.linspace(r2[0], r2[1], n_scan[1])
    cc1, cc2 = np.meshgrid(c1, c2, indexing="ij")

    f_x_2d = np.full_like(cc1, np.nan)
    f_y_2d = np.full_like(cc1, np.nan)
    f_z_2d = np.full_like(cc1, np.nan)

    for i in range(n_scan[0]):
        for j in range(n_scan[1]):
            pt = [xc_um, yc_um, zc_um]
            pt[AXIS_INDEX[a1]] = cc1[i, j]
            pt[AXIS_INDEX[a2]] = cc2[i, j]
            freqs = compute_trap_freqs_at_point(
                potential_interps,
                field_interps,
                voltage_list,
                cfg,
                pt[0],
                pt[1],
                pt[2],
                x_range_um=x_range_um,
                y_range_um=y_range_um,
                z_range_um=z_range_um,
                n_pts=n_fit_pts,
                fit_degree=fit_degree,
            )
            f_x_2d[i, j] = freqs["f_x"]
            f_y_2d[i, j] = freqs["f_y"]
            f_z_2d[i, j] = freqs["f_z"]

    return cc1, cc2, f_x_2d, f_y_2d, f_z_2d
