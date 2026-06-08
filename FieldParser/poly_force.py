"""
基于 3D 多项式拟合的电场力函数构建

对每个电极基函数独立做 quartic 多项式拟合，用解析梯度替代格点插值，
消除数值噪声，获得全局光滑的力场。

复用 equilibrium/potential_fit_3d.py 的拟合基础设施。
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from equilibrium.potential_fit_3d import (
    FitResult3D,
    fit_potential_3d_quartic,
    grad_fit_3d,
)

logger = logging.getLogger(__name__)


def _extract_grid_axes(grid_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从格点坐标提取唯一轴坐标，并验证规则网格。"""
    x = np.unique(grid_coord[:, 0])
    y = np.unique(grid_coord[:, 1])
    z = np.unique(grid_coord[:, 2])
    nx, ny, nz = len(x), len(y), len(z)
    if grid_coord.shape[0] != nx * ny * nz:
        raise ValueError(
            f"格点非规则网格: 共 {grid_coord.shape[0]} 点，"
            f"但 unique(x,y,z) 得 {nx}×{ny}×{nz}={nx * ny * nz}"
        )
    return x, y, z


def _grid_range_um(
    grid_coord: np.ndarray,
    dl: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """从归一化格点坐标推导物理拟合范围 (µm)。"""
    # grid_coord 是归一化坐标，乘以 dl (m) 再乘 1e6 得 µm
    coord_um = grid_coord * dl * 1e6
    x_range = (float(coord_um[:, 0].min()), float(coord_um[:, 0].max()))
    y_range = (float(coord_um[:, 1].min()), float(coord_um[:, 1].max()))
    z_range = (float(coord_um[:, 2].min()), float(coord_um[:, 2].max()))
    return x_range, y_range, z_range


def _make_field_callable(fit: FitResult3D, dl: float, dV: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    从 FitResult3D 创建与 calc_field 兼容的电场 callable。

    单位转换链：
        r_norm ──×dl_um──→ r_µm ──grad_fit_3d──→ dV/dr (V/µm) ──×(dl_um/dV)──→ E_norm

    其中 dl_um = dl * 1e6 (m → µm)

    Parameters
    ----------
    fit : FitResult3D
        多项式拟合结果（µm / V 单位）
    dl : float
        特征长度 (m)，归一化坐标 → 物理坐标的缩放因子
    dV : float
        特征电压 (V)，归一化势 → 物理电压的缩放因子
    """
    dl_um = dl * 1e6       # m → µm，用于 r_norm → r_µm
    scale = dl_um / dV     # E_norm = -(dV/dr_µm) * (dl_um / dV)

    def field_at_r(r_norm: np.ndarray) -> np.ndarray:
        r_norm = np.atleast_2d(r_norm)
        r_um = r_norm * dl_um
        grad_V = grad_fit_3d(fit, r_um)  # (M, 3), V/µm
        E_norm = -grad_V * scale          # (M, 3), 归一化 E
        return E_norm

    return field_at_r


def calc_field_from_poly(
    grid_coord: np.ndarray,
    grid_voltage: np.ndarray,
    dl: float,
    dV: float,
    *,
    fit_mode: str = "quartic",
    n_pts_per_axis: int = 8,
    center_um: tuple[float, float, float] | None = None,
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> list[Callable[[np.ndarray], np.ndarray]]:
    """
    对每个电极基函数做 3D 多项式拟合，返回解析梯度 callable 列表。

    功能与 calc_field() 对等：输入归一化格点数据，返回的 callable 接受
    归一化坐标、返回归一化电场。

    Parameters
    ----------
    grid_coord : np.ndarray, shape (N, 3)
        归一化格点坐标（来自 csv_reader）
    grid_voltage : np.ndarray, shape (N, n_basis)
        归一化势场值，每列对应一个电极基
    dl : float
        特征长度 (m)
    dV : float
        特征电压 (V)
    fit_mode : str
        多项式拟合模式，默认 "quartic"（35 项，i+j+k≤4）
    n_pts_per_axis : int
        拟合采样每轴点数，默认 8
    center_um : tuple or None
        拟合中心 (µm)，默认为格点中心
    range_um : tuple or None
        拟合范围 ((x_min,x_max), ...) (µm)，默认为格点范围

    Returns
    -------
    list[Callable]
        每个电极基对应的电场 callable，签名 field_at_r(r_norm) -> E_norm (M, 3)
    """
    x_arr, y_arr, z_arr = _extract_grid_axes(grid_coord)
    nx, ny, nz = len(x_arr), len(y_arr), len(z_arr)

    # 自动推导拟合中心和范围
    if range_um is None:
        range_um = _grid_range_um(grid_coord, dl)
    if center_um is None:
        center_um = (
            0.5 * (range_um[0][0] + range_um[0][1]),
            0.5 * (range_um[1][0] + range_um[1][1]),
            0.5 * (range_um[2][0] + range_um[2][1]),
        )

    dl_um = dl * 1e6  # m → µm
    um_to_norm = lambda v: v / dl_um  # µm → 归一化

    n_basis = grid_voltage.shape[1]
    field_interps: list[Callable[[np.ndarray], np.ndarray]] = []

    for i in range(n_basis):
        # 对第 i 个电极基，从格点构建势插值器（仅用于拟合采样）
        V_3d = grid_voltage[:, i].reshape(nx, ny, nz, order="C")
        pot_interp = RegularGridInterpolator(
            (x_arr, y_arr, z_arr), V_3d,
            method="linear", bounds_error=False, fill_value=None,
        )

        # 包装为 fit_potential_3d_quartic 期望的接口
        def _make_compute_V(pot_interp_i: RegularGridInterpolator, dV_val: float):
            def compute_V_total(r_norm: np.ndarray) -> np.ndarray:
                return pot_interp_i(r_norm) * dV_val  # 归一化 → V
            return compute_V_total

        compute_V = _make_compute_V(pot_interp, dV)

        # 执行 3D 多项式拟合
        fit = fit_potential_3d_quartic(
            compute_V_total=compute_V,
            um_to_norm=um_to_norm,
            center_um=center_um,
            range_um=range_um,
            n_pts_per_axis=n_pts_per_axis,
            fit_mode=fit_mode,
        )

        logger.info(
            "电极 %d/%d 多项式拟合: R²=%.8f, fit_mode=%s",
            i + 1, n_basis, fit.r_squared, fit_mode,
        )

        # 创建力兼容 callable
        field_interps.append(_make_field_callable(fit, dl, dV))

    return field_interps
