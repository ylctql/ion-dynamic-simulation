"""
总势场 3D 四次多项式拟合

fit_potential_3d_quartic: 直接拟合 V(x,y,z) 为完整 3D 四次多项式
V(x,y,z) = Σ c_ijk x^i y^j z^k, i,j,k ≤ 4
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial import polynomial as P


DEGREE_QUARTIC = 4


@dataclass
class FitResult3D:
    """
    3D 四次多项式拟合结果

    势场模型: V_shifted(x,y,z) = Σ c_ijk u^i v^j w^k，其中 u=(x-x0)/L, v=(y-y0)/L, w=(z-z0)/L
    为数值稳定，拟合在缩放坐标 [-1,1] 上进行。L 为半跨度。
    其中 V_shifted = V_true - V_min_ref（将零点平移到参考最小势）。
    坐标单位: μm
    """

    coeffs: np.ndarray  # shape (5,5,5)，对应 (u,v,w) 的系数
    center_um: tuple[float, float, float]
    scale_um: float  # L，坐标缩放半跨度
    potential_offset_V: float  # 参考最小势 V_min_ref（被减去）
    r_squared: float


def fit_potential_3d_quartic(
    compute_V_total: Callable[[np.ndarray], np.ndarray],
    um_to_norm: Callable[[float], float],
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    n_pts_per_axis: int | tuple[int, int, int] = 8,
    potential_offset_V: float | None = None,
) -> FitResult3D:
    """
    直接对总势场做 3D 四次多项式拟合，并将势能零点平移到参考最小势。

    模型: V_shifted(x,y,z) = Σ c_ijk x^i y^j z^k, 0 ≤ i,j,k ≤ 4
    共 5×5×5 = 125 项，使用最小二乘拟合。

    Parameters
    ----------
    compute_V_total : callable
        接收 r (N, 3) 归一化坐标，返回 V_total (N,) 单位 V
    um_to_norm : callable
        μm → 归一化坐标的转换函数
    center_um : tuple
        参考中心 (x0, y0, z0) μm，用于报告
    range_um : tuple of tuples, optional
        各轴拟合范围 ((x_min, x_max), (y_min, y_max), (z_min, z_max)) μm
        默认 (-50, 50) 每轴
    n_pts_per_axis : int | tuple[int, int, int]
        采样点数。可传单个整数（x/y/z 相同），或传 (nx, ny, nz) 分别指定三轴采样点数。
        为保证 125 项拟合稳定，建议每轴 >= 6。
    potential_offset_V : float | None
        势能零点平移参考值 V_min_ref（单位 V）。若为 None，则退化为当前拟合采样点的最小势。

    Returns
    -------
    FitResult3D
        拟合结果
    """
    if range_um is None:
        range_um = ((-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0))

    (xr0, xr1), (yr0, yr1), (zr0, zr1) = range_um
    x0, y0, z0 = center_um
    # 缩放：u = (x-x0)/L，使 u,v,w ∈ [-1,1]，提高数值稳定性
    Lx = max(xr1 - x0, x0 - xr0, 1e-6)
    Ly = max(yr1 - y0, y0 - yr0, 1e-6)
    Lz = max(zr1 - z0, z0 - zr0, 1e-6)
    scale_um = max(Lx, Ly, Lz)

    xr_n = (um_to_norm(xr0), um_to_norm(xr1))
    yr_n = (um_to_norm(yr0), um_to_norm(yr1))
    zr_n = (um_to_norm(zr0), um_to_norm(zr1))

    # 3D 网格采样（支持三轴不同采样数）
    if isinstance(n_pts_per_axis, int):
        nx = ny = nz = int(n_pts_per_axis)
    else:
        nx, ny, nz = [int(v) for v in n_pts_per_axis]
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError(f"n_pts_per_axis 每轴至少为 2，当前为 ({nx}, {ny}, {nz})")

    x_um = np.linspace(xr0, xr1, nx)
    y_um = np.linspace(yr0, yr1, ny)
    z_um = np.linspace(zr0, zr1, nz)
    xx, yy, zz = np.meshgrid(x_um, y_um, z_um, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = zz.ravel()

    # 缩放坐标 u,v,w ∈ [-1,1]
    u_flat = (x_flat - x0) / scale_um
    v_flat = (y_flat - y0) / scale_um
    w_flat = (z_flat - z0) / scale_um

    # 归一化坐标供 compute_V_total 使用
    r_norm = np.column_stack([
        um_to_norm(x_flat),
        um_to_norm(y_flat),
        um_to_norm(z_flat),
    ])
    V_true = compute_V_total(r_norm)

    # 设计矩阵与最小二乘（在缩放坐标上）
    deg = [DEGREE_QUARTIC, DEGREE_QUARTIC, DEGREE_QUARTIC]
    V_mat = P.polyvander3d(u_flat, v_flat, w_flat, deg)
    valid = np.isfinite(V_true)
    if np.sum(valid) < 125:
        raise ValueError(
            f"有效采样点 {np.sum(valid)} 不足，至少需 125 点拟合 3D 四次多项式。"
            f"请增大 n_pts_per_axis（当前网格 {nx}x{ny}x{nz}）或检查势场范围。"
        )
    V_mat_valid = V_mat[valid]
    V_true_valid = V_true[valid]

    # 势能零点平移：默认减去拟合采样最小值；若给定参考值则统一使用该参考值
    if potential_offset_V is None:
        v_min_ref = float(np.min(V_true_valid))
    else:
        v_min_ref = float(potential_offset_V)
    V_shifted_valid = V_true_valid - v_min_ref

    coefs_flat, residuals, rank, s = np.linalg.lstsq(V_mat_valid, V_shifted_valid, rcond=None)
    coefs = np.zeros((5, 5, 5))
    for idx in range(125):
        i = idx // 25
        j = (idx % 25) // 5
        k = idx % 5
        coefs[i, j, k] = coefs_flat[idx]

    # R²
    V_pred = V_mat_valid @ coefs_flat
    ss_res = np.sum((V_shifted_valid - V_pred) ** 2)
    ss_tot = np.sum((V_shifted_valid - np.mean(V_shifted_valid)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return FitResult3D(
        coeffs=coefs,
        center_um=center_um,
        scale_um=scale_um,
        potential_offset_V=v_min_ref,
        r_squared=r_squared,
    )


def eval_fit_3d(fit: FitResult3D, r_um: np.ndarray) -> np.ndarray:
    """
    用 3D 拟合结果在给定坐标上求值。

    Parameters
    ----------
    fit : FitResult3D
        fit_potential_3d_quartic 的返回值
    r_um : np.ndarray, shape (N, 3)
        坐标 (x, y, z) 单位 μm

    Returns
    -------
    V : np.ndarray, shape (N,)
        拟合电势值，单位 V
    """
    r_um = np.atleast_2d(r_um)
    x0, y0, z0 = fit.center_um
    L = fit.scale_um
    u = (r_um[:, 0] - x0) / L
    v = (r_um[:, 1] - y0) / L
    w = (r_um[:, 2] - z0) / L
    return P.polyval3d(u, v, w, fit.coeffs)


def grad_fit_3d(fit: FitResult3D, r_um: np.ndarray) -> np.ndarray:
    """
    用 3D 拟合结果计算电势梯度 dV/dr，单位 V/μm。

    拟合在 (u,v,w) 上，u=(x-x0)/L，故 dV/dx = (dV/du) / L
    """
    r_um = np.atleast_2d(r_um)
    x0, y0, z0 = fit.center_um
    L = fit.scale_um
    u = (r_um[:, 0] - x0) / L
    v = (r_um[:, 1] - y0) / L
    w = (r_um[:, 2] - z0) / L
    c = fit.coeffs

    # dV/du, dV/dv, dV/dw（在缩放坐标下）
    c_du = np.zeros((4, 5, 5))
    for i in range(4):
        c_du[i, :, :] = c[i + 1, :, :] * (i + 1)
    dV_du = P.polyval3d(u, v, w, c_du)

    c_dv = np.zeros((5, 4, 5))
    for j in range(4):
        c_dv[:, j, :] = c[:, j + 1, :] * (j + 1)
    dV_dv = P.polyval3d(u, v, w, c_dv)

    c_dw = np.zeros((5, 5, 4))
    for k in range(4):
        c_dw[:, :, k] = c[:, :, k + 1] * (k + 1)
    dV_dw = P.polyval3d(u, v, w, c_dw)

    # 链式法则: dV/dx = (dV/du) * (du/dx) = (dV/du) / L
    return np.column_stack([dV_du / L, dV_dv / L, dV_dw / L])


def hessian_fit_3d(fit: FitResult3D, r_um: np.ndarray) -> np.ndarray:
    """
    用 3D 拟合结果计算电势 Hessian，单位 V/μm^2。

    Returns
    -------
    hess : np.ndarray, shape (N, 3, 3)
        每个点对应一个 3x3 对称 Hessian 矩阵，变量顺序为 (x, y, z)。
    """
    r_um = np.atleast_2d(r_um)
    x0, y0, z0 = fit.center_um
    L = fit.scale_um
    u = (r_um[:, 0] - x0) / L
    v = (r_um[:, 1] - y0) / L
    w = (r_um[:, 2] - z0) / L
    c = fit.coeffs

    # 二阶偏导在 (u,v,w) 坐标下
    c_duu = np.zeros((3, 5, 5))
    for i in range(3):
        c_duu[i, :, :] = c[i + 2, :, :] * (i + 2) * (i + 1)
    d2V_duu = P.polyval3d(u, v, w, c_duu)

    c_dvv = np.zeros((5, 3, 5))
    for j in range(3):
        c_dvv[:, j, :] = c[:, j + 2, :] * (j + 2) * (j + 1)
    d2V_dvv = P.polyval3d(u, v, w, c_dvv)

    c_dww = np.zeros((5, 5, 3))
    for k in range(3):
        c_dww[:, :, k] = c[:, :, k + 2] * (k + 2) * (k + 1)
    d2V_dww = P.polyval3d(u, v, w, c_dww)

    c_duv = np.zeros((4, 4, 5))
    for i in range(4):
        for j in range(4):
            c_duv[i, j, :] = c[i + 1, j + 1, :] * (i + 1) * (j + 1)
    d2V_duv = P.polyval3d(u, v, w, c_duv)

    c_duw = np.zeros((4, 5, 4))
    for i in range(4):
        for k in range(4):
            c_duw[i, :, k] = c[i + 1, :, k + 1] * (i + 1) * (k + 1)
    d2V_duw = P.polyval3d(u, v, w, c_duw)

    c_dvw = np.zeros((5, 4, 4))
    for j in range(4):
        for k in range(4):
            c_dvw[:, j, k] = c[:, j + 1, k + 1] * (j + 1) * (k + 1)
    d2V_dvw = P.polyval3d(u, v, w, c_dvw)

    # 链式法则：每个一阶导都会引入 1/L，因此二阶导统一为 1/L^2
    scale2 = L * L
    hxx = d2V_duu / scale2
    hyy = d2V_dvv / scale2
    hzz = d2V_dww / scale2
    hxy = d2V_duv / scale2
    hxz = d2V_duw / scale2
    hyz = d2V_dvw / scale2

    n = r_um.shape[0]
    hess = np.zeros((n, 3, 3), dtype=float)
    hess[:, 0, 0] = hxx
    hess[:, 1, 1] = hyy
    hess[:, 2, 2] = hzz
    hess[:, 0, 1] = hxy
    hess[:, 1, 0] = hxy
    hess[:, 0, 2] = hxz
    hess[:, 2, 0] = hxz
    hess[:, 1, 2] = hyz
    hess[:, 2, 1] = hyz
    return hess


