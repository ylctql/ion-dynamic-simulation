"""radial_trap 平衡构型求解、解析 micromotion 与面内声子。

平衡求解复用 equilibrium.energy.total_energy_and_grad（L-BFGS-B）；
z 方向无势场依赖，用窄 bounds 钉在 z=0 平面（防最小化器数值漂移出平面）。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from equilibrium.energy import total_energy_and_grad
from equilibrium.phonon import PhononResult, solve_phonon_modes
from equilibrium.potential_fit_3d import FitResult3D

from radial_trap.potential import (
    get_species,
    mathieu_q_per_axis,
    rf_field_V_per_um,
    rf_omega_rad_per_s,
)
from radial_trap.types import EquilibriumResult, MicromotionResult, RadialTrapParams

# z 钉住半宽（µm）：z 无势场依赖，仅防最小化器漂移出平面
_Z_HALF_UM = 1e-6
# 判定离子贴边（疑似非囚禁或范围不足）的距离阈值（µm）
_BOUND_MARGIN_UM = 0.5
# |r_a| 低于此值（µm）视为零 → excess 比无定义，置 NaN
_R_ZERO_UM = 1e-9


def initial_positions(params: RadialTrapParams) -> np.ndarray:
    """按 seed 均匀撒在 x/y range 内（z=0），返回 (N,3) µm。"""
    rng = np.random.default_rng(params.seed)
    x0, x1 = params.x_range_um
    y0, y1 = params.y_range_um
    x = rng.uniform(x0, x1, params.n_ions)
    y = rng.uniform(y0, y1, params.n_ions)
    return np.column_stack([x, y, np.zeros(params.n_ions)])


def find_radial_equilibrium(
    fit: FitResult3D,
    params: RadialTrapParams,
    maxiter: int = 50000,
    tol: float = 1e-15,
) -> EquilibriumResult:
    """L-BFGS-B 最小化 外势+库伦 总能量，返回 EquilibriumResult。

    tol 映射到 L-BFGS-B ftol；gtol 固定 1e-12（eV/µm，保证间距收敛到
    远优于 1e-6 相对精度，供声子/解析验证使用）。
    """
    sp = get_species(params.species_name)
    n = params.n_ions
    charge = np.full(n, sp.charge_ec)
    r0 = initial_positions(params)

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        breakdown, grad = total_energy_and_grad(
            fit, x.reshape(n, 3), charge, softening_um=params.softening_um
        )
        return breakdown.total_eV, grad.ravel()

    x_lo, x_hi = params.x_range_um
    y_lo, y_hi = params.y_range_um
    bounds = [
        (x_lo, x_hi),
        (y_lo, y_hi),
        (-_Z_HALF_UM, _Z_HALF_UM),
    ] * n

    res = minimize(
        fun=lambda x: objective(x)[0],
        x0=r0.ravel(),
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": float(tol), "gtol": 1e-12},
    )

    r_eq = res.x.reshape(n, 3)
    breakdown, grad = total_energy_and_grad(
        fit, r_eq, charge, softening_um=params.softening_um
    )
    hit_bounds = bool(
        np.any(r_eq[:, 0] - x_lo < _BOUND_MARGIN_UM)
        or np.any(x_hi - r_eq[:, 0] < _BOUND_MARGIN_UM)
        or np.any(r_eq[:, 1] - y_lo < _BOUND_MARGIN_UM)
        or np.any(y_hi - r_eq[:, 1] < _BOUND_MARGIN_UM)
    )
    return EquilibriumResult(
        r_eq_um=r_eq,
        r_init_um=r0,
        total_eV=breakdown.total_eV,
        grad_norm_eV_per_um=float(np.linalg.norm(grad)),
        converged=bool(res.success),
        n_iter=int(res.nit),
        hit_bounds=hit_bounds,
        message=str(res.message),
    )


def micromotion_amplitude(
    r_eq_um: np.ndarray, params: RadialTrapParams
) -> MicromotionResult:
    """各离子一阶 RF micromotion 峰值幅度（矢量 + 模长 + excess 比）。

    a_mm = (Q_C·1e12/(m_kg·Ω²))·E_rf(r_eq) [µm]；
    四极极限恒等式 a_mm,a = (q_a/2)·r_a，故 excess 比
    ρ_a = a_mm,a/((q_a/2)·r_a) 偏离 1 的量即高阶 RF 项（B≠0）引起的
    过剩 micromotion。|r_a| < 1e-9 µm 或 q_a = 0 时无定义 → NaN。
    """
    sp = get_species(params.species_name)
    omega = rf_omega_rad_per_s(params.freq_rf_mhz)
    pref = sp.charge_C * 1e12 / (sp.mass_kg * omega**2)  # µm²/V

    r = np.asarray(r_eq_um, dtype=float)
    e_rf = rf_field_V_per_um(
        r[:, 0], r[:, 1], params.A_V_per_um2, params.B_V_per_um4
    )
    a_mm = np.zeros_like(r)
    a_mm[:, :2] = pref * e_rf
    mag = np.linalg.norm(a_mm, axis=1)

    q_x, q_y = mathieu_q_per_axis(params)
    half_q = np.zeros_like(r)
    half_q[:, 0] = 0.5 * q_x
    half_q[:, 1] = 0.5 * q_y
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(np.abs(r) < _R_ZERO_UM, np.nan, a_mm / (half_q * r))

    return MicromotionResult(a_mm_um=a_mm, mag_um=mag, excess_ratio=excess)


def solve_inplane_phonons(
    fit: FitResult3D, r_eq_um: np.ndarray, params: RadialTrapParams
) -> PhononResult:
    """面内 (x,y) 声子：dof_indices 取每离子 (3i, 3i+1)。"""
    sp = get_species(params.species_name)
    r = np.asarray(r_eq_um, dtype=float)
    n = r.shape[0]
    charge = np.full(n, sp.charge_ec)
    dof = np.array([ax for i in range(n) for ax in (3 * i, 3 * i + 1)], dtype=int)
    return solve_phonon_modes(
        fit,
        r,
        charge,
        sp.mass_amu,
        softening_um=params.softening_um,
        dof_indices=dof,
    )
