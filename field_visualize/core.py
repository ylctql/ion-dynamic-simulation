"""
电场可视化核心：单位换算、电势计算、网格构建、势场平滑
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy.signal import savgol_filter

from utils import Voltage

CoordAxis = Literal["x", "y", "z"]
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def apply_savgol_smooth(
    grid_coord: np.ndarray,
    grid_voltage: np.ndarray,
    axes: tuple[str, ...],
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    对势场格点数据沿指定方向应用 Savitzky-Golay 滤波平滑。

    Parameters
    ----------
    grid_coord : np.ndarray, shape (N, 3)
        格点坐标 (x, y, z)，已按 lexsort 排序
    grid_voltage : np.ndarray, shape (N, n_basis)
        势场值
    axes : tuple[str, ...]
        滤波方向，如 ("x",), ("x", "y"), ("x", "y", "z")
    window_length : int
        SG 滤波器窗口长度，须为奇数
    polyorder : int
        SG 滤波器多项式阶数

    Returns
    -------
    grid_voltage_smooth : np.ndarray
        平滑后的势场，形状与 grid_voltage 相同
    """
    if not axes:
        return grid_voltage.copy()

    x = np.unique(grid_coord[:, 0])
    y = np.unique(grid_coord[:, 1])
    z = np.unique(grid_coord[:, 2])
    nx, ny, nz = len(x), len(y), len(z)
    n_basis = grid_voltage.shape[1]

    axis_map = {"x": 0, "y": 1, "z": 2}
    sizes = [nx, ny, nz]

    # 确保 window_length 为奇数且合法
    if window_length % 2 == 0:
        window_length = max(3, window_length - 1)
    window_length = max(polyorder + 1, window_length)
    if window_length % 2 == 0:
        window_length += 1

    out = grid_voltage.copy()
    for i in range(n_basis):
        V = out[:, i].reshape(nx, ny, nz, order="C")
        for ax in axes:
            axis_idx = axis_map.get(ax)
            if axis_idx is None:
                continue
            size = sizes[axis_idx]
            w = min(window_length, size)
            if w % 2 == 0:
                w = max(polyorder + 1, w - 1)
            if w < polyorder + 1:
                continue
            try:
                V = savgol_filter(V, w, polyorder, axis=axis_idx, mode="nearest")
            except ValueError:
                pass
        out[:, i] = V.ravel(order="C")
    return out


def um_to_norm(val_um: float, dl: float) -> float:
    """μm → 归一化坐标"""
    return val_um * 1e-6 / dl


def norm_to_um(val_norm: float, dl: float) -> float:
    """归一化坐标 → μm"""
    return val_norm * dl / 1e-6


def is_rf(voltage: Voltage) -> bool:
    """判断是否为 RF 电极（V0 非零）"""
    return abs(voltage.V0) > 1e-12


def compute_potentials(
    potential_interps: list[Callable],
    field_interps: list[Callable],
    voltage_list: list[Voltage],
    cfg,
    r: np.ndarray,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算静电势、RF 赝势、总电势（SI 单位 V）

    potential_interps 输出 V_raw/dV（无量纲），需乘 dV 得 SI 电势
    field_interps 输出 (dl/dV)*E_si，即 E_si = field_interp * (dV/dl)
    """
    from FieldConfiguration.constants import m as ION_MASS

    r = np.atleast_2d(r)
    dl, dV = cfg.dl, cfg.dV
    V_dc = np.zeros(r.shape[0])
    V_rf_amp = np.zeros(r.shape[0])
    E_rf = np.zeros((r.shape[0], 3))

    for v_interp, e_interp, v in zip(potential_interps, field_interps, voltage_list):
        V_basis_norm = np.atleast_1d(v_interp(r)).ravel()
        E_basis = e_interp(r)
        V_dc += v.V_bias * V_basis_norm * dV
        if is_rf(v):
            V_rf_amp += v.V0 * V_basis_norm * dV
            E_rf += v.V0 * E_basis

    E_rf_si = E_rf * (dV / dl)
    from scipy.constants import e

    Omega = cfg.Omega
    q = e
    coeff = (q**2) / (4 * ION_MASS * Omega**2)
    E_rf_sq = np.sum(E_rf_si**2, axis=1)
    V_pseudo_J = coeff * E_rf_sq
    V_pseudo = V_pseudo_J / q

    V_total = V_dc + V_pseudo
    return V_dc, V_rf_amp, V_pseudo, V_total


def apply_offset_min(
    V_dc: np.ndarray, V_pseudo: np.ndarray, V_total: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """静电势与赝势各减去其最小值，总电势 = 偏置后静电势 + 偏置后赝势"""
    valid_dc = V_dc[~np.isnan(V_dc)]
    valid_pseudo = V_pseudo[~np.isnan(V_pseudo)]
    V_dc_off = V_dc - (np.min(valid_dc) if valid_dc.size else 0)
    V_pseudo_off = V_pseudo - (np.min(valid_pseudo) if valid_pseudo.size else 0)
    V_total_off = V_dc_off + V_pseudo_off
    return V_dc_off, V_pseudo_off, V_total_off


def set_ylim_from_data(ax, data: np.ndarray, pad_frac: float = 0.1) -> None:
    """根据数据范围设置 y 轴，避免数值过小时曲线不可见"""
    data = np.asarray(data)
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return
    y_min, y_max = np.min(valid), np.max(valid)
    span = y_max - y_min
    if span < 1e-30:
        ax.set_ylim(y_min - 0.1, y_min + 0.1)
    else:
        pad = max(span * pad_frac, span * 0.05)
        ax.set_ylim(y_min - pad, y_max + pad)


def build_grid_1d(
    vary_axis: CoordAxis,
    vary_range: tuple[float, float],
    x_const: float,
    y_const: float,
    z_const: float,
    n_pts: int,
) -> np.ndarray:
    """构建 1D 采样点，vary_range 为变化坐标的范围（归一化）"""
    coords = np.linspace(vary_range[0], vary_range[1], n_pts)
    r = np.zeros((n_pts, 3))
    r[:, 0] = x_const
    r[:, 1] = y_const
    r[:, 2] = z_const
    r[:, AXIS_INDEX[vary_axis]] = coords
    return r


def build_grid_2d(
    vary_axes: tuple[CoordAxis, CoordAxis],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    const_val: float,
    n_pts: tuple[int, int],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """构建 2D 采样网格，返回 (r, (cc1, cc2))，r 为 (n1*n2, 3)"""
    a1, a2 = vary_axes
    i1, i2 = AXIS_INDEX[a1], AXIS_INDEX[a2]
    c1 = np.linspace(x_range[0], x_range[1], n_pts[0])
    c2 = np.linspace(y_range[0], y_range[1], n_pts[1])
    cc1, cc2 = np.meshgrid(c1, c2, indexing="ij")
    n_tot = cc1.size
    r = np.full((n_tot, 3), const_val)
    r[:, i1] = cc1.ravel()
    r[:, i2] = cc2.ravel()
    return r, (cc1, cc2)
