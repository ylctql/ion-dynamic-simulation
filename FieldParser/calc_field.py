"""
根据势场数据基求梯度得电场，并建立线性插值函数
参考 outline.md - 得到基的空间分布函数 List[Callable]
"""
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def calc_field(
    grid_coord: np.ndarray,
    grid_voltage: np.ndarray,
    *,
    method: str = "linear",
    bounds_error: bool = False,
    fill_value: float = np.nan,
) -> list[Callable[[np.ndarray], np.ndarray]]:
    """
    对每组电势基求梯度得电场，再建立三维线性插值函数

    Parameters
    ----------
    grid_coord : np.ndarray, shape (N, 3)
        格点坐标 (x, y, z)，已按 lexsort(z, y, x) 排序
    grid_voltage : np.ndarray, shape (N, n_basis)
        势场值，每列对应一组基
    method : str
        插值方法，默认 "linear"
    bounds_error : bool
        越界时是否抛错，默认 False
    fill_value : float
        越界时的填充值，默认 np.nan

    Returns
    -------
    field_interps : list[Callable]
        每组基对应的电场插值函数
        每个函数 f(r) 接受 r: (M, 3)，返回 E: (M, 3)，即该基在空间各点的电场 (Ex, Ey, Ez)
    """
    x = np.unique(grid_coord[:, 0])
    y = np.unique(grid_coord[:, 1])
    z = np.unique(grid_coord[:, 2])

    nx, ny, nz = len(x), len(y), len(z)
    if grid_coord.shape[0] != nx * ny * nz:
        raise ValueError(
            f"格点非规则网格: 共 {grid_coord.shape[0]} 点，"
            f"但 unique(x,y,z) 得 {nx}×{ny}×{nz}={nx*ny*nz}"
        )

    n_basis = grid_voltage.shape[1]
    field_interps: list[Callable[[np.ndarray], np.ndarray]] = []

    for i in range(n_basis):
        V = grid_voltage[:, i].reshape(nx, ny, nz, order="C")

        # E = -grad V
        Ex, Ey, Ez = np.gradient(-V, x, y, z, edge_order=2)

        interp_x = RegularGridInterpolator(
            (x, y, z), Ex, method=method, bounds_error=bounds_error, fill_value=fill_value
        )
        interp_y = RegularGridInterpolator(
            (x, y, z), Ey, method=method, bounds_error=bounds_error, fill_value=fill_value
        )
        interp_z = RegularGridInterpolator(
            (x, y, z), Ez, method=method, bounds_error=bounds_error, fill_value=fill_value
        )

        def make_field(ix, iy, iz):
            def field_at_r(r: np.ndarray) -> np.ndarray:
                r = np.atleast_2d(r)
                ex = ix(r)
                ey = iy(r)
                ez = iz(r)
                return np.column_stack((ex, ey, ez))

            return field_at_r

        field_interps.append(make_field(interp_x, interp_y, interp_z))

    return field_interps
