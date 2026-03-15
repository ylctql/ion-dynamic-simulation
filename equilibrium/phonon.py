"""
离子晶格在平衡位置附近的小振动（声子）分析。

核心流程：
1) 在平衡位置处构建总势能 Hessian（外势 + 库伦）
2) 进行质量加权得到动力学矩阵
3) 对角化提取本征频率与本征模
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import e as ELEMENTARY_CHARGE
from scipy.constants import physical_constants

from equilibrium.energy import COULOMB_K, UM_TO_M, _as_charge_coulomb
from equilibrium.potential_fit_3d import FitResult3D, hessian_fit_3d

ATOMIC_MASS_KG = physical_constants["atomic mass constant"][0]


@dataclass
class PhononResult:
    """声子模式分析结果。"""

    hessian_total_eV_per_um2: np.ndarray  # (3N, 3N)
    hessian_trap_eV_per_um2: np.ndarray  # (3N, 3N)
    hessian_coulomb_eV_per_um2: np.ndarray  # (3N, 3N)
    dynamical_matrix_s2: np.ndarray  # (3N, 3N)
    omega2_s2: np.ndarray  # (M,), 按频率降序排序
    freq_hz_signed: np.ndarray  # (3N,), 负值表示不稳定模（虚频）
    eigvec_mass_weighted: np.ndarray  # (M, M), D 的本征向量（列向量对应降序频率）
    eigvec_cartesian: np.ndarray  # (M, M), 转回笛卡尔坐标后的模态向量
    dof_indices: np.ndarray  # (M,), 子空间对应的全局自由度索引


def trap_hessian(
    fit: FitResult3D,
    r_um: np.ndarray,
    charge_ec: np.ndarray,
) -> np.ndarray:
    """
    外势总能 Hessian，单位 eV/μm^2，形状 (3N, 3N)。
    """
    r_um = np.asarray(r_um, dtype=float)
    q_e = np.asarray(charge_ec, dtype=float).ravel()
    n = r_um.shape[0]
    if r_um.ndim != 2 or r_um.shape[1] != 3:
        raise ValueError(f"r_um 应为 (N,3)，当前 {r_um.shape}")
    if q_e.shape[0] != n:
        raise ValueError(f"charge 长度 {q_e.shape[0]} 与 N={n} 不一致")

    # V 的 Hessian 单位 V/μm^2；U_trap[eV] = Σ q_e * V[V]
    h_v = hessian_fit_3d(fit, r_um)
    h = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        block = q_e[i] * h_v[i]
        s = slice(3 * i, 3 * i + 3)
        h[s, s] = block
    return h


def coulomb_hessian(
    r_um: np.ndarray,
    charge_ec: np.ndarray,
    softening_um: float = 0.0,
) -> np.ndarray:
    """
    库伦势能 Hessian，单位 eV/μm^2，形状 (3N, 3N)。
    """
    r_um = np.asarray(r_um, dtype=float)
    q_c = _as_charge_coulomb(charge_ec)
    n = r_um.shape[0]
    if r_um.ndim != 2 or r_um.shape[1] != 3:
        raise ValueError(f"r_um 应为 (N,3)，当前 {r_um.shape}")
    if q_c.shape[0] != n:
        raise ValueError(f"charge 长度 {q_c.shape[0]} 与 N={n} 不一致")

    r_m = r_um * UM_TO_M
    soft_m = float(max(softening_um, 0.0)) * UM_TO_M

    h_j_per_um2 = np.zeros((3 * n, 3 * n), dtype=float)
    eye3 = np.eye(3)
    for i in range(n):
        for j in range(i + 1, n):
            rij = r_m[i] - r_m[j]
            r2 = float(np.dot(rij, rij))
            s2 = r2 + soft_m * soft_m
            s = np.sqrt(s2)
            s3 = s2 * s
            s5 = s3 * s2

            pref = COULOMB_K * q_c[i] * q_c[j]
            # 对位移向量 rij 的二阶导（m 坐标）:
            # d2(1/s)/drdr = -I/s^3 + 3 rr^T/s^5
            d2_m = pref * (-eye3 / s3 + 3.0 * np.outer(rij, rij) / s5)  # J/m^2
            # m -> um 二次链式法则
            block = d2_m * (UM_TO_M * UM_TO_M)  # J/um^2

            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * j, 3 * j + 3)
            h_j_per_um2[si, si] += block
            h_j_per_um2[sj, sj] += block
            h_j_per_um2[si, sj] -= block
            h_j_per_um2[sj, si] -= block

    return h_j_per_um2 / ELEMENTARY_CHARGE


def total_hessian(
    fit: FitResult3D,
    r_um: np.ndarray,
    charge_ec: np.ndarray,
    softening_um: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算总 Hessian = H_trap + H_coulomb（单位 eV/μm^2）。
    """
    h_trap = trap_hessian(fit, r_um, charge_ec)
    h_coul = coulomb_hessian(r_um, charge_ec, softening_um=softening_um)
    h_total = h_trap + h_coul
    return h_total, h_trap, h_coul


def solve_phonon_modes(
    fit: FitResult3D,
    r_um: np.ndarray,
    charge_ec: np.ndarray,
    mass_amu: float | np.ndarray,
    softening_um: float = 0.0,
    dof_indices: np.ndarray | None = None,
) -> PhononResult:
    """
    在平衡位置附近线性化并求解声子模式。

    Parameters
    ----------
    mass_amu : float | np.ndarray
        离子质量（amu）。可传标量（全离子同质量）或长度 N 数组。
    dof_indices : np.ndarray | None
        自由度子空间索引（针对 3N 维坐标）。若传入，则仅在该子矩阵上求本征模。
    """
    r_um = np.asarray(r_um, dtype=float)
    n = r_um.shape[0]
    if r_um.ndim != 2 or r_um.shape[1] != 3:
        raise ValueError(f"r_um 应为 (N,3)，当前 {r_um.shape}")

    if np.ndim(mass_amu) == 0:
        m_amu = np.full(n, float(mass_amu), dtype=float)
    else:
        m_amu = np.asarray(mass_amu, dtype=float).ravel()
    if m_amu.shape[0] != n:
        raise ValueError(f"mass_amu 长度 {m_amu.shape[0]} 与 N={n} 不一致")
    if np.any(m_amu <= 0):
        raise ValueError("mass_amu 必须为正")

    h_total, h_trap, h_coul = total_hessian(
        fit=fit,
        r_um=r_um,
        charge_ec=charge_ec,
        softening_um=softening_um,
    )

    # eV/um^2 -> N/m
    k_si = h_total * ELEMENTARY_CHARGE / (UM_TO_M * UM_TO_M)
    m_kg = m_amu * ATOMIC_MASS_KG
    m_xyz = np.repeat(m_kg, 3)

    if dof_indices is None:
        idx = np.arange(3 * n, dtype=int)
    else:
        idx = np.asarray(dof_indices, dtype=int).ravel()
        if idx.size == 0:
            raise ValueError("dof_indices 不能为空")
        if np.any(idx < 0) or np.any(idx >= 3 * n):
            raise ValueError(f"dof_indices 超出范围 [0, {3*n - 1}]")
    if idx.size != np.unique(idx).size:
        raise ValueError("dof_indices 不能包含重复索引")

    h_total = h_total[np.ix_(idx, idx)]
    h_trap = h_trap[np.ix_(idx, idx)]
    h_coul = h_coul[np.ix_(idx, idx)]
    k_si = k_si[np.ix_(idx, idx)]
    m_xyz = m_xyz[idx]

    inv_sqrt_m = 1.0 / np.sqrt(m_xyz)

    # 动力学矩阵 D = M^{-1/2} K M^{-1/2}
    d_mat = (inv_sqrt_m[:, None] * k_si) * inv_sqrt_m[None, :]
    d_mat = 0.5 * (d_mat + d_mat.T)

    omega2, eig_mw = np.linalg.eigh(d_mat)
    omega2 = np.real_if_close(omega2)
    eig_mw = np.real_if_close(eig_mw)

    # 笛卡尔坐标模态向量 q = M^{-1/2} e
    eig_cart = inv_sqrt_m[:, None] * eig_mw
    # 统一每个模态的欧氏范数，便于比较形状
    norms = np.linalg.norm(eig_cart, axis=0)
    valid = norms > 0
    eig_cart[:, valid] /= norms[valid][None, :]

    freq_hz = np.sign(omega2) * np.sqrt(np.abs(omega2)) / (2.0 * np.pi)
    # 统一按频率从高到低排列，确保后续输出/可视化顺序一致
    order = np.argsort(freq_hz)[::-1]
    omega2 = omega2[order]
    freq_hz = freq_hz[order]
    eig_mw = eig_mw[:, order]
    eig_cart = eig_cart[:, order]

    return PhononResult(
        hessian_total_eV_per_um2=h_total,
        hessian_trap_eV_per_um2=h_trap,
        hessian_coulomb_eV_per_um2=h_coul,
        dynamical_matrix_s2=d_mat,
        omega2_s2=np.asarray(omega2, dtype=float),
        freq_hz_signed=np.asarray(freq_hz, dtype=float),
        eigvec_mass_weighted=np.asarray(eig_mw, dtype=float),
        eigvec_cartesian=np.asarray(eig_cart, dtype=float),
        dof_indices=np.asarray(idx, dtype=int),
    )

