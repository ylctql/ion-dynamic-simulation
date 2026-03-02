"""
根据 Voltage List 组合电场，结合耗散项得到 force 函数
供 ComputeKernel 实时计算使用
参考 outline.md - force.py

force 采用模块级函数 + 模块级变量，以便 multiprocessing 可 pickle（ism-cuda 模式）
"""
from typing import Callable

import numpy as np

from FieldConfiguration.field_settings import FieldSettings
from utils import Voltage

from .calc_field import calc_field


def _zero_force(r: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """无外力，供 multiprocessing 序列化（须为顶层函数）"""
    return np.zeros_like(r)


def build_force(
    field_settings: FieldSettings,
    grid_coord: np.ndarray,
    grid_voltage: np.ndarray,
    charge: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    根据 field_settings 构建 force 函数。
    若 voltage_list 为空则返回 _zero_force。

    Parameters
    ----------
    field_settings : FieldSettings
        电势场与耗散参数
    grid_coord : np.ndarray
        格点坐标，来自 csv_reader
    grid_voltage : np.ndarray
        势场值，来自 csv_reader
    charge : np.ndarray
        各离子电荷量

    Returns
    -------
    Callable[[r, v, t], F]
        force(r, v, t) -> (N, 3) 外力
    """
    if not field_settings.voltage_list:
        return _zero_force
    field_interps = calc_field(grid_coord, grid_voltage)
    gamma = field_settings.get_gamma()
    return make_force(
        field_interps,
        field_settings.voltage_list,
        gamma,
        charge,
        grid_coord,
    )

# 模块级变量，由 make_force 设置，供 force 使用（fork 子进程继承）
_field_interps: list | None = None
_voltage_list: list | None = None
_gamma: float | np.ndarray | None = None
_charge: np.ndarray | None = None
_bound_min: np.ndarray | None = None
_bound_max: np.ndarray | None = None


def force(r: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """
    模块级 force 函数，使用 _field_interps 等模块变量
    可被 pickle 序列化，供 multiprocessing 使用
    """
    r = np.asarray(r, dtype=float, order="C")
    v = np.asarray(v, dtype=float, order="C")
    if _bound_min is not None and _bound_max is not None:
        mask = (
            (r[:, 0] >= _bound_min[0]) & (r[:, 0] <= _bound_max[0])
            & (r[:, 1] >= _bound_min[1]) & (r[:, 1] <= _bound_max[1])
            & (r[:, 2] >= _bound_min[2]) & (r[:, 2] <= _bound_max[2])
        )
        if not np.all(mask):
            r = np.clip(r, _bound_min, _bound_max)
    E_tot = np.zeros_like(r)
    if _field_interps is not None and _voltage_list is not None:
        for field_interp, voltage in zip(_field_interps, _voltage_list):
            coef = voltage.V0 * voltage.f(t) + voltage.V_bias
            E_tot = E_tot + coef * field_interp(r)
    F = _charge.reshape(-1, 1) * E_tot - _gamma * v
    return F.astype(float, order="C")


def make_force(
    field_interps: list[Callable[[np.ndarray], np.ndarray]],
    voltage_list: list[Voltage],
    gamma: float | np.ndarray,
    charge: np.ndarray,
    grid_coord: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    构造包含外电场与耗散作用的力函数

    Parameters
    ----------
    field_interps : list[Callable]
        每组基的电场插值函数，来自 calc_field
    voltage_list : list[Voltage]
        电压组合，与 field_interps 顺序一致
    gamma : float | np.ndarray
        耗散强度，scalar 时为标量，vector 时为 (3,) 数组
    charge : np.ndarray, shape (N,)
        各离子电荷量（无量纲，单位元电荷）
    grid_coord : np.ndarray, shape (M, 3)
        格点坐标，用于确定插值有效区域边界

    Returns
    -------
    force : Callable[[r, v, t], F]
        r: (N, 3) 位置, v: (N, 3) 速度, t: float 时间
        F: (N, 3) 外力（不含库仑力，由 ComputeKernel 内部叠加）
    """
    if len(field_interps) != len(voltage_list):
        raise ValueError(
            f"field_interps 数量 {len(field_interps)} 与 voltage_list 数量 {len(voltage_list)} 不一致"
        )

    bound_min = np.array(
        [np.min(grid_coord[:, i]) + 1e-9 for i in range(3)], dtype=float
    )
    bound_max = np.array(
        [np.max(grid_coord[:, i]) - 1e-9 for i in range(3)], dtype=float
    )

    gamma_val = np.asarray(gamma, dtype=float)
    if gamma_val.ndim == 0:
        gamma_val = float(gamma_val)

    global _field_interps, _voltage_list, _gamma, _charge, _bound_min, _bound_max
    _field_interps = field_interps
    _voltage_list = voltage_list
    _gamma = gamma_val
    _charge = charge
    _bound_min = bound_min
    _bound_max = bound_max

    return force
