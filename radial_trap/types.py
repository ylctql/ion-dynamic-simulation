"""radial_trap 数据类型定义。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from equilibrium.phonon import PhononResult
from equilibrium.potential_fit_3d import FitResult3D


@dataclass(frozen=True)
class RadialTrapParams:
    """2D 径向多项式势阱参数（物理单位）。

    RF 势（零到峰幅度驱动）: φ_rf = A(x²−y²) + B(x⁴−6x²y²+y⁴)
    RF bias 势（静态）: D·φ_rf
    DC 势: E(x²−y²) + F(x⁴−6x²y²+y⁴)
    其中 x,y 单位 µm，A/E 单位 V/µm²，B/F 单位 V/µm⁴，D 无量纲。
    """

    A_V_per_um2: float
    B_V_per_um4: float = 0.0
    D_dimless: float = 0.0
    E_dc_V_per_um2: float = 0.0
    F_dc_V_per_um4: float = 0.0
    freq_rf_mhz: float = 35.28
    species_name: str = "Ba135+"
    n_ions: int = 10
    x_range_um: tuple[float, float] = (-10.0, 10.0)
    y_range_um: tuple[float, float] = (-80.0, 80.0)
    seed: int = 0
    softening_um: float = 1e-3
    scale_um: float = 100.0


@dataclass(frozen=True)
class LatticeConditions:
    """晶格成形条件评估结果（系数单位 V/µm^n，组合量见 radial_trap 文档）。

    条件编号对应设计文档：
    (1) 压制交叉项: DB+F → 0 且 B → 0
    (2) 压制 y 方向高次项: −16αAB+DB+F → 0 且 B → 0
    (3) y 方向束缚: c_y2 > 0
    (4) x 方向束缚: c_x2 > 0
    """

    alpha_eff_um2_per_V: float
    c_x2: float
    c_y2: float
    s_x4: float
    s_y4: float
    s_x2y2: float
    c_x6: float  # 16αB²（x⁶ 与 y⁶ 系数相同）
    c_x4y2: float  # 48αB²（x⁴y² 与 x²y⁴ 系数相同）
    db_plus_f: float
    y4_comb: float
    abs_b: float
    pass_cross: bool
    pass_y_high: bool
    pass_y_confine: bool
    pass_x_confine: bool
    f_trap_x_mhz: float
    f_trap_y_mhz: float
    q_mathieu_x: float
    q_mathieu_y: float
    eps4_x: float  # |s_x4|·L_ref²/c_x2，四次项相对重要性（L_ref=10 µm）
    eps4_y: float
    xy_degenerate: bool  # c_x2 ≈ c_y2（链取向简并，初始化随机性会决定结果）


@dataclass
class MicromotionResult:
    """各离子 RF micromotion 解析结果（一阶谐波峰值，单位 µm）。"""

    a_mm_um: np.ndarray  # (N,3) 幅度矢量（z 分量恒 0）
    mag_um: np.ndarray  # (N,) 模长
    excess_ratio: np.ndarray  # (N,3) ρ_a = a_mm,a/((q_a/2)·r_a)；r_a≈0 处为 NaN


@dataclass
class EquilibriumResult:
    """平衡构型求解结果。"""

    r_eq_um: np.ndarray  # (N,3)
    r_init_um: np.ndarray  # (N,3)
    total_eV: float
    grad_norm_eV_per_um: float
    converged: bool
    n_iter: int
    hit_bounds: bool  # 任一离子距 x/y 边界 < 0.5 µm（疑似非囚禁或范围不足）
    message: str


@dataclass
class RadialTrapResult:
    """一次完整评估的结果汇总。"""

    params: RadialTrapParams
    fit: FitResult3D
    conditions: LatticeConditions
    equilibrium: EquilibriumResult
    micromotion: MicromotionResult
    phonon: PhononResult | None = None
