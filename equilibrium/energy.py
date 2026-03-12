"""
平衡构型求解所需的能量模型

- 外势能: U_trap = Σ q_i * V(r_i)
- 库伦势能: U_coul = Σ_{i<j} k*q_i*q_j/|r_i-r_j|
- 总势能: U_total = U_trap + U_coul
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import e as ELEMENTARY_CHARGE
from scipy.constants import epsilon_0, pi

from equilibrium.potential_fit_3d import FitResult3D, eval_fit_3d, grad_fit_3d


COULOMB_K = 1.0 / (4.0 * pi * epsilon_0)
UM_TO_M = 1e-6


@dataclass
class EnergyBreakdown:
    """能量分解（单位 eV）"""

    trap_eV: float
    coulomb_eV: float
    total_eV: float


def _as_charge_coulomb(charge_ec: np.ndarray) -> np.ndarray:
    """将以元电荷为单位的电荷数组转换为库仑"""
    q = np.asarray(charge_ec, dtype=float).ravel()
    return q * ELEMENTARY_CHARGE


def trap_energy_and_grad(
    fit: FitResult3D,
    r_um: np.ndarray,
    charge_ec: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    计算外势能及其梯度（对 r_um 的导数）。

    Returns
    -------
    energy_eV : float
        外势能，单位 eV
    grad_eV_per_um : np.ndarray, shape (N, 3)
        dU/dr，单位 eV/μm
    """
    r_um = np.asarray(r_um, dtype=float)
    q_c = _as_charge_coulomb(charge_ec)
    if r_um.ndim != 2 or r_um.shape[1] != 3:
        raise ValueError(f"r_um 应为 (N,3)，当前 {r_um.shape}")
    if q_c.shape[0] != r_um.shape[0]:
        raise ValueError(f"charge 长度 {q_c.shape[0]} 与 N={r_um.shape[0]} 不一致")

    V = eval_fit_3d(fit, r_um)  # V
    dV_dr = grad_fit_3d(fit, r_um)  # V/um
    energy_j = float(np.sum(q_c * V))
    grad_j_per_um = q_c[:, None] * dV_dr  # J/um
    energy = energy_j / ELEMENTARY_CHARGE
    grad = grad_j_per_um / ELEMENTARY_CHARGE  # eV/um
    return energy, grad


def coulomb_energy_and_grad(
    r_um: np.ndarray,
    charge_ec: np.ndarray,
    softening_um: float = 0.0,
) -> tuple[float, np.ndarray]:
    """
    计算库伦势能及其梯度（对 r_um 的导数，单位 eV 和 eV/μm）。

    softening_um>0 时使用 sqrt(r^2 + soft^2) 防止粒子重合导致奇点。
    """
    r_um = np.asarray(r_um, dtype=float)
    q_c = _as_charge_coulomb(charge_ec)
    n = r_um.shape[0]
    if r_um.ndim != 2 or r_um.shape[1] != 3:
        raise ValueError(f"r_um 应为 (N,3)，当前 {r_um.shape}")
    if q_c.shape[0] != n:
        raise ValueError(f"charge 长度 {q_c.shape[0]} 与 N={n} 不一致")

    r_m = r_um * UM_TO_M
    diff = r_m[:, None, :] - r_m[None, :, :]  # (N,N,3), m
    dist2 = np.sum(diff * diff, axis=2)  # m^2
    soft_m = float(max(softening_um, 0.0)) * UM_TO_M
    if soft_m > 0.0:
        dist2 = dist2 + soft_m * soft_m

    # 避免对角线除零
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)
    inv_dist = 1.0 / dist
    inv_dist3 = inv_dist / dist2

    q_prod = q_c[:, None] * q_c[None, :]

    # 势能：只取上三角
    energy_mat = COULOMB_K * q_prod * inv_dist
    energy_j = float(np.sum(np.triu(energy_mat, k=1)))

    # 梯度 wrt r_um:
    # dU/dri_um = -k * Σ_j q_i q_j * (ri-rj)_m / |ri-rj|^3 * (dr_m/dr_um)
    # dr_m/dr_um = 1e-6
    coeff = -COULOMB_K * q_prod * inv_dist3 * UM_TO_M  # J/(um*m)
    grad_j_per_um = np.sum(coeff[:, :, None] * diff, axis=1)  # J/um
    energy = energy_j / ELEMENTARY_CHARGE
    grad = grad_j_per_um / ELEMENTARY_CHARGE  # eV/um
    return energy, grad


def total_energy_and_grad(
    fit: FitResult3D,
    r_um: np.ndarray,
    charge_ec: np.ndarray,
    softening_um: float = 0.0,
) -> tuple[EnergyBreakdown, np.ndarray]:
    """计算总势能分解及总梯度（单位 eV 与 eV/μm）"""
    trap_e, trap_g = trap_energy_and_grad(fit, r_um, charge_ec)
    coul_e, coul_g = coulomb_energy_and_grad(r_um, charge_ec, softening_um=softening_um)
    total = trap_e + coul_e
    grad = trap_g + coul_g
    return EnergyBreakdown(trap_eV=trap_e, coulomb_eV=coul_e, total_eV=total), grad

