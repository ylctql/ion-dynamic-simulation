"""2D 径向多项式势：RF 赝势 + RF bias + DC 的解析构建与物理量计算。

物理模型（与全仓约定一致：RF 幅度为零到峰，V0·cos(Ωt)）：

- RF 电势基（x,y 镜像对称的 2D Laplace 解）：
  φ_rf = A(x²−y²) + B(x⁴−6x²y²+y⁴)
- RF 场（本模块取 E_rf = +∇φ_rf，即设计文档的符号约定；物理电场为 −∇φ，
  仅分量符号相反，不影响幅度/模长与条件判断）：
  E_rf = (2Ax+4Bx³−12Bxy², −2Ay−12Bx²y+4By³)   [V/µm, 零到峰]
- 赝势（V 形式，能量 Ψ[J] = Q²|E|²/(4mΩ²) 除以 Q）：
  φ_pp[V] = α_eff·|E_rf[V/µm]|²，α_eff = Q·1e12/(4·m·Ω²) [µm²/V]
- 总势 φ_tot = φ_pp + D·φ_rf + [E(x²−y²) + F(x⁴−6x²y²+y⁴)]，展开为 9 个
  x,y 偶次单项式（最高 6 次，B² 部分 = 16αB²(x²+y²)³）。
- Mathieu q（trap_stability 约定）：q = 4Q·k2_RF/(mΩ²)，k2_RF 为 RF 幅值势
  的 x²/y² 系数（V/µm²）；轴阱频 ω = √(2Q·k2/m)。
- 每离子 micromotion 幅度（lattice.py）：a_mm = (Q/mΩ²)·E_rf(r_eq)，
  四极极限下 a_mm,a = (q_a/2)·r_a。
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.constants import e as ELEMENTARY_CHARGE

from equilibrium.potential_fit_3d import FitResult3D
from FieldConfiguration.ion_species import ION_SPECIES, Species
from FieldParser.potential_fit import k2_to_trap_freq_MHz

from radial_trap.types import LatticeConditions, RadialTrapParams

# 评估四次项相对重要性所用的参考长度（µm）
L_REF_UM = 10.0

# 总势单项式基底 (i, j)（x,y 均偶次，最高 6 次；x⁶/y⁶ 系数相同、x⁴y²/x²y⁴ 相同）
TOTAL_BASIS_2D: tuple[tuple[int, int], ...] = (
    (2, 0), (0, 2), (4, 0), (0, 4), (2, 2), (6, 0), (0, 6), (4, 2), (2, 4),
)


def get_species(name: str) -> Species:
    """按名称查 ION_SPECIES；未知名称抛 KeyError（CLI 捕获后报 parser.error）。"""
    sp = ION_SPECIES.get(name)
    if sp is None:
        raise KeyError(f"未知物种 {name!r}，可选: {sorted(ION_SPECIES)}")
    return sp


def rf_potential_V(x_um: np.ndarray, y_um: np.ndarray, A: float, B: float) -> np.ndarray:
    """RF 电势 φ_rf = A(x²−y²) + B(x⁴−6x²y²+y⁴) [V]，x/y 单位 µm。"""
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    return A * (x * x - y * y) + B * (x**4 - 6.0 * x * x * y * y + y**4)


def rf_field_V_per_um(x_um: np.ndarray, y_um: np.ndarray, A: float, B: float) -> np.ndarray:
    """RF 场 E_rf = +∇φ_rf [V/µm, 零到峰]，返回 (N,2)。"""
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    ex = 2.0 * A * x + 4.0 * B * x**3 - 12.0 * B * x * y * y
    ey = -2.0 * A * y - 12.0 * B * x * x * y + 4.0 * B * y**3
    return np.stack([ex, ey], axis=-1)


def rf_omega_rad_per_s(freq_rf_mhz: float) -> float:
    """RF 角频率 Ω = 2π·f（rad/s）。"""
    return 2.0 * np.pi * float(freq_rf_mhz) * 1e6


def alpha_eff_um2_per_V(freq_rf_mhz: float, species: Species) -> float:
    """赝势系数 α_eff [µm²/V]：φ_pp[V] = α_eff·|E_rf[V/µm]|²。

    α_eff = Q·1e12/(4·m·Ω²)，等价于 Ψ[J] = Q²|E|²/(4mΩ²) 除以 Q 的伏特形式
    （单电荷离子即除以元电荷 e；Note 的 α 为能量量纲，此处统一电势量纲 V）；
    注意使用物种自身质量（不同于 field_visualize 中硬编码 Ba135 的路径）。
    """
    omega = rf_omega_rad_per_s(freq_rf_mhz)
    return species.charge_C * 1e12 / (4.0 * species.mass_kg * omega**2)


def total_potential_coefficients(params: RadialTrapParams) -> dict[tuple[int, int], float]:
    """总势 9 个单项式的物理系数（V/µm^(i+j)），键为 (i, j) 指数。"""
    sp = get_species(params.species_name)
    alpha = alpha_eff_um2_per_V(params.freq_rf_mhz, sp)
    A = params.A_V_per_um2
    B = params.B_V_per_um4
    D = params.D_dimless
    E = params.E_dc_V_per_um2
    F = params.F_dc_V_per_um4
    return {
        (2, 0): 4.0 * alpha * A * A + D * A + E,
        (0, 2): 4.0 * alpha * A * A - D * A - E,
        (4, 0): 16.0 * alpha * A * B + D * B + F,
        (0, 4): -16.0 * alpha * A * B + D * B + F,
        (2, 2): -6.0 * (D * B + F),
        (6, 0): 16.0 * alpha * B * B,
        (0, 6): 16.0 * alpha * B * B,
        (4, 2): 48.0 * alpha * B * B,
        (2, 4): 48.0 * alpha * B * B,
    }


def build_total_potential_fit(params: RadialTrapParams) -> FitResult3D:
    """总势 → FitResult3D（(7,7,7) 缩放坐标张量，直接构造、无拟合）。

    物理系数 a_ij [V/µm^(i+j)] 按归一化约定存为 c_ij = a_ij·L^(i+j)
    （同 make_ideal_trap_fit）；z 方向无依赖（k 恒 0）。
    """
    coeff = total_potential_coefficients(params)
    if not any(v != 0.0 for v in coeff.values()):
        raise ValueError(
            "总势所有系数为 0（平坦势）：请检查 A/B/D/E/F 参数（A=B=E=F=0 时无囚禁）"
        )
    L = float(params.scale_um)
    coeffs = np.zeros((7, 7, 7), dtype=float)
    basis: list[tuple[int, int, int]] = []
    for (i, j), a_ij in coeff.items():
        coeffs[i, j, 0] = a_ij * L ** (i + j)
        basis.append((i, j, 0))
    basis.sort(key=lambda t: (sum(t), t))  # (总次数, 字典序) —— potential_fit_3d 约定
    return FitResult3D(
        coeffs=coeffs,
        center_um=(0.0, 0.0, 0.0),
        scale_um=L,
        potential_offset_V=0.0,
        r_squared=1.0,
        fit_mode=None,
        symmetry_axes=("x", "y"),
        basis_exps=tuple(basis),
    )


def augment_fit_with_axial_confinement(
    fit: FitResult3D, f_z_mhz: float, species_name: str
) -> FitResult3D:
    """给径向总势追加简谐轴向囚禁项 c_z2·z²（--export-fz 导出 3D 动力学用）。

    径向模块把 z 钉在 0 平面，总势本身无任何 z 项；导出的势直接交
    main.py --poly-potential 做 3D 演化时 z 方向无囚禁——库伦作用把离子
    沿 z 摊开成片、进而 y 分层（一维链在无轴势中只是鞍点，z 模全负）。
    本函数按目标轴向阱频 f_z 追加 c_z2·z²（c_z2 = m·ω_z²/(2Q)，复用
    trap_freq_MHz_to_k2），对应真实阱由端帽 DC 提供的轴向囚禁。

    N=20 经验：单排链要求 f_y/f_x 与 f_z/f_x 均 ≳ 8.5（有限 N zigzag
    边界，与横向方向无关），取 ~10 留裕量。
    """
    sp = get_species(species_name)
    c_z2 = trap_freq_MHz_to_k2(f_z_mhz, sp.mass_kg, sp.charge_C)
    coeffs = fit.coeffs.copy()
    coeffs[0, 0, 2] = c_z2 * fit.scale_um**2
    if (0, 0, 2) in fit.basis_exps:  # 幂等：已有 z² 项则只覆盖系数
        return replace(fit, coeffs=coeffs)
    basis = tuple(sorted(fit.basis_exps + ((0, 0, 2),), key=lambda t: (sum(t), t)))
    return replace(fit, coeffs=coeffs, basis_exps=basis, symmetry_axes=("x", "y", "z"))


def pseudopotential_V(
    x_um: np.ndarray, y_um: np.ndarray, params: RadialTrapParams
) -> np.ndarray:
    """RF 赝势 φ_pp = α_eff·|E_rf|² [V]（恒 ≥ 0，最小值在 RF null 处）。"""
    sp = get_species(params.species_name)
    alpha = alpha_eff_um2_per_V(params.freq_rf_mhz, sp)
    e_field = rf_field_V_per_um(x_um, y_um, params.A_V_per_um2, params.B_V_per_um4)
    return alpha * (e_field[..., 0] ** 2 + e_field[..., 1] ** 2)


def bias_potential_V(
    x_um: np.ndarray, y_um: np.ndarray, params: RadialTrapParams
) -> np.ndarray:
    """RF bias 静态势 D·φ_rf [V]（与 RF 电势同空间形状）。"""
    return params.D_dimless * rf_potential_V(
        x_um, y_um, params.A_V_per_um2, params.B_V_per_um4
    )


def dc_potential_V(
    x_um: np.ndarray, y_um: np.ndarray, params: RadialTrapParams
) -> np.ndarray:
    """DC 势 E(x²−y²) + F(x⁴−6x²y²+y⁴) [V]。"""
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    return (
        params.E_dc_V_per_um2 * (x * x - y * y)
        + params.F_dc_V_per_um4 * (x**4 - 6.0 * x * x * y * y + y**4)
    )


def total_potential_V(
    x_um: np.ndarray, y_um: np.ndarray, params: RadialTrapParams
) -> np.ndarray:
    """总势直接解析求值 = 赝势 + RF bias + DC 三组分之和（独立于 FitResult3D）。"""
    return (
        pseudopotential_V(x_um, y_um, params)
        + bias_potential_V(x_um, y_um, params)
        + dc_potential_V(x_um, y_um, params)
    )


def mathieu_q_per_axis(params: RadialTrapParams) -> tuple[float, float]:
    """Mathieu q（trap_stability 约定）：q = 4Q·k2_RF/(mΩ²)。

    k2_RF 为 RF 幅值势的 x²(y²) 系数：φ_rf 中 x² 系数 A、y² 系数 −A，
    故 q_y = −q_x。
    """
    sp = get_species(params.species_name)
    omega = rf_omega_rad_per_s(params.freq_rf_mhz)
    pref = 4.0 * sp.charge_C * 1e12 / (sp.mass_kg * omega**2)
    return (pref * params.A_V_per_um2, -pref * params.A_V_per_um2)


def trap_freq_MHz_to_k2(f_mhz: float, mass_kg: float, charge: float) -> float:
    """k2_to_trap_freq_MHz 的精确逆：阱频 f [MHz] → 二次系数 k2 [V/µm²]。

    ω = √(2Q·k2/m)（k2 以 V/µm² 计，含 1e12 µm²/m² 换算）→
    k2 = m·(2πf)²/(2Q) / 1e12。f ≤ 0 / NaN 抛 ValueError。
    """
    f = float(f_mhz)
    if not f > 0.0:  # 同时挡住 NaN
        raise ValueError(f"阱频需为有限正数 (MHz)，收到 {f_mhz}")
    omega = 2.0 * np.pi * f * 1e6
    return mass_kg * omega**2 / (2.0 * charge) / 1e12


def invert_trap_freqs_to_params(
    f_x_mhz: float,
    f_y_mhz: float,
    freq_rf_mhz: float,
    species_name: str,
    D: float = 0.0,
) -> tuple[float, float, float]:
    """由目标轴阱频 (f_x, f_y) 反算势系数 (A, E)，返回 (A, E, q_x)。

    利用和式恒等式 c_x2 + c_y2 = 8αA²（D、E 均从和式消去）：

    - c_a2 = trap_freq_MHz_to_k2(f_a)（目标二次系数，V/µm²）
    - A = √((c_x2+c_y2)/(8α))   （取正根 → q_x > 0）
    - E = (c_x2−c_y2)/2 − D·A   （D·A 项参与 c_x2/c_y2 的分配，需扣除）

    B/F 只影响四次及以上系数，反算仅确定二次部分（B/F 由调用方另行给定）。
    q_y = −q_x；调用方应检查 |q|（Mathieu 稳定性要求 |q| ≲ 0.9）。
    """
    sp = get_species(species_name)
    alpha = alpha_eff_um2_per_V(freq_rf_mhz, sp)
    c_x2 = trap_freq_MHz_to_k2(f_x_mhz, sp.mass_kg, sp.charge_C)
    c_y2 = trap_freq_MHz_to_k2(f_y_mhz, sp.mass_kg, sp.charge_C)
    a_sum = c_x2 + c_y2
    if not a_sum > 0.0:
        raise ValueError(f"c_x2+c_y2 = {a_sum} ≤ 0，无法反算 A")
    A = math.sqrt(a_sum / (8.0 * alpha))
    E = 0.5 * (c_x2 - c_y2) - D * A
    q_x, _ = mathieu_q_per_axis(
        RadialTrapParams(A_V_per_um2=A, freq_rf_mhz=freq_rf_mhz,
                         species_name=species_name)
    )
    return A, E, q_x


def evaluate_conditions(
    params: RadialTrapParams, cond_tol: float = 1e-12
) -> LatticeConditions:
    """评估设计文档的 4 条 2D 晶格成形条件并计算派生量。"""
    sp = get_species(params.species_name)
    alpha = alpha_eff_um2_per_V(params.freq_rf_mhz, sp)
    c = total_potential_coefficients(params)
    c_x2 = c[(2, 0)]
    c_y2 = c[(0, 2)]
    db_plus_f = params.D_dimless * params.B_V_per_um4 + params.F_dc_V_per_um4
    y4_comb = c[(0, 4)]  # = −16αAB + DB + F
    abs_b = abs(params.B_V_per_um4)

    f_x = k2_to_trap_freq_MHz(c_x2, sp.mass_kg, charge=sp.charge_C)
    f_y = k2_to_trap_freq_MHz(c_y2, sp.mass_kg, charge=sp.charge_C)
    q_x, q_y = mathieu_q_per_axis(params)

    def _eps4(s4: float, c2: float) -> float:
        return abs(s4) * L_REF_UM**2 / c2 if c2 > 0.0 else float("inf")

    xy_degenerate = (
        c_x2 > 0.0 and abs(c_x2 - c_y2) / c_x2 < 1e-6
    )
    return LatticeConditions(
        alpha_eff_um2_per_V=alpha,
        c_x2=c_x2,
        c_y2=c_y2,
        s_x4=c[(4, 0)],
        s_y4=c[(0, 4)],
        s_x2y2=c[(2, 2)],
        c_x6=c[(6, 0)],
        c_x4y2=c[(4, 2)],
        db_plus_f=db_plus_f,
        y4_comb=y4_comb,
        abs_b=abs_b,
        pass_cross=abs(db_plus_f) <= cond_tol and abs_b <= cond_tol,
        pass_y_high=abs(y4_comb) <= cond_tol and abs_b <= cond_tol,
        pass_y_confine=c_y2 > 0.0,
        pass_x_confine=c_x2 > 0.0,
        f_trap_x_mhz=f_x,
        f_trap_y_mhz=f_y,
        q_mathieu_x=q_x,
        q_mathieu_y=q_y,
        eps4_x=_eps4(c[(4, 0)], c_x2),
        eps4_y=_eps4(c[(0, 4)], c_y2),
        xy_degenerate=xy_degenerate,
    )
