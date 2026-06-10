"""
Mathieu 稳定性参数 (a, q) 核心计算

从实际场几何或理想四极阱参数计算 Mathieu a/q 参数、secural 频率、稳定性判断。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.constants import e as EC, pi

from FieldConfiguration.ion_species import Species


@dataclass(frozen=True)
class StabilityResult:
    """Mathieu 稳定性分析结果"""

    # Mathieu 参数（无量纲）
    a_x: float; a_y: float; a_z: float
    q_x: float; q_y: float; q_z: float
    # Secular 频率（绝热近似，MHz）
    f_sec_x: float; f_sec_y: float; f_sec_z: float
    # 总势阱频（直接拟合总势，MHz）
    f_trap_x: float; f_trap_y: float; f_trap_z: float
    # 稳定性判断
    is_stable: bool
    stability_note: str
    # 输入元信息
    species_name: str
    mass_amu: float
    omega_rf: float  # rad/s
    freq_rf_MHz: float
    # 曲率诊断（V/μm²）
    k2_dc: dict[str, float]
    k2_rf_amp: dict[str, float]


def check_stability_region(
    a_x: float, a_y: float, a_z: float,
    q_x: float, q_y: float, q_z: float,
) -> tuple[bool, str]:
    """
    判断 Mathieu 方程是否在第一稳定区内。

    第一稳定区近似判据：
      0 < |q| < 0.908
      a + q²/2 > 0 （下边界）
      a < 1 - q - q²/8 （上边界，仅当 q > 0 时有意义）

    Returns
    -------
    (is_stable, note)
    """
    unstable_axes: list[str] = []

    for axis, a, q in [("x", a_x, q_x), ("y", a_y, q_y), ("z", a_z, q_z)]:
        q_abs = abs(q)
        # q 超出第一稳定区
        if q_abs > 0.908:
            unstable_axes.append(f"{axis}: |q|={q_abs:.4f} > 0.908")
            continue
        # 下边界 a + q^2/2 > 0
        lower = a + q**2 / 2
        if lower <= 0:
            unstable_axes.append(f"{axis}: a+q^2/2 = {lower:.4f} <= 0")
            continue
        # 上边界 a < 1 - q - q²/8 (仅 q > 0 时检查)
        if q > 0:
            upper = 1 - q - q**2 / 8
            if a > upper:
                unstable_axes.append(f"{axis}: a={a:.4f} > {upper:.4f} (上边界)")

    if not unstable_axes:
        return True, "stable in first stability region"
    else:
        notes = "; ".join(unstable_axes)
        return False, f"unstable: {notes}"


def _secular_freq_mhz(a: float, q: float, Omega: float) -> float:
    """从 a, q 计算绝热近似 secular 频率 (MHz)"""
    arg = a + q**2 / 2
    if arg <= 0:
        return float("nan")
    omega_sec = (Omega / 2) * np.sqrt(arg)
    return omega_sec / (2 * pi * 1e6)


def compute_stability_direct(
    rf_freq_MHz: float,
    r0_um: float,
    V0: float,
    U: float = 0.0,
    species: Species | None = None,
    geometry: str = "linear",
) -> StabilityResult:
    """
    教科书公式计算 Mathieu a/q 参数（理想四极阱）。

    Parameters
    ----------
    rf_freq_MHz : float
        RF 驱动频率 (MHz)
    r0_um : float
        特征尺寸 r₀ (μm)
    V0 : float
        RF 零-峰值电压幅度 (V)
    U : float
        DC 电压 (V)，默认 0
    species : Species or None
        离子种类，None 时默认 Ba135+
    geometry : str
        "linear"（线性 Paul 阱）或 "3d"（三维 Paul 阱）
    """
    from FieldConfiguration.ion_species import BA_135

    if species is None:
        species = BA_135

    Omega = 2 * pi * rf_freq_MHz * 1e6  # rad/s
    r0 = r0_um * 1e-6  # m
    m = species.mass_kg

    coeff_a = 4 * EC / (m * r0**2 * Omega**2)
    coeff_q = 2 * EC / (m * r0**2 * Omega**2)

    if geometry == "linear":
        # 线性 Paul 阱：径向有 RF 提供赝势约束，轴向由 DC 端帽电压约束
        # q: RF 四极场（径向）, 符号取决于电场方向
        q_x = coeff_q * V0
        q_y = -q_x
        q_z = 0.0
        # a: DC 端帽电压 U > 0 → 轴向约束 (a_z > 0)，径向反约束
        a_z = coeff_a * U
        a_x = -a_z / 2
        a_y = a_x
    elif geometry == "3d":
        # 三维 Paul 阱 (Endcap)
        q_r = coeff_q * V0 / 2
        a_r = -coeff_a * U
        q_x = q_r
        q_y = q_r
        q_z = 0.0
        a_x = a_r
        a_y = a_r
        a_z = -2 * a_r
    else:
        raise ValueError(f"未知几何类型: {geometry}，支持 linear / 3d")

    # Secular 频率（绝热近似）
    f_sec_x = _secular_freq_mhz(a_x, q_x, Omega)
    f_sec_y = _secular_freq_mhz(a_y, q_y, Omega)
    f_sec_z = _secular_freq_mhz(a_z, q_z, Omega)

    is_stable, stability_note = check_stability_region(a_x, a_y, a_z, q_x, q_y, q_z)

    return StabilityResult(
        a_x=a_x, a_y=a_y, a_z=a_z,
        q_x=q_x, q_y=q_y, q_z=q_z,
        f_sec_x=f_sec_x, f_sec_y=f_sec_y, f_sec_z=f_sec_z,
        f_trap_x=f_sec_x, f_trap_y=f_sec_y, f_trap_z=f_sec_z,
        is_stable=is_stable,
        stability_note=stability_note,
        species_name=species.name,
        mass_amu=species.mass_amu,
        omega_rf=Omega,
        freq_rf_MHz=rf_freq_MHz,
        k2_dc={"x": 0.0, "y": 0.0, "z": 0.0},
        k2_rf_amp={"x": 0.0, "y": 0.0, "z": 0.0},
    )


def find_trap_center(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    grid_extent_um: tuple[float, float, float] = (100.0, 100.0, 300.0),
    n_coarse: int = 15,
) -> tuple[float, float, float]:
    """
    自动检测陷阱中心：在粗网格上找 V_total 最小值，再用 1D 拟合精化。

    Returns
    -------
    (xc_um, yc_um, zc_um) 陷阱中心坐标 (μm)
    """
    from field_visualize.core import compute_potentials, um_to_norm

    dl = cfg.dl
    # 构建粗 3D 网格
    xg = np.linspace(-grid_extent_um[0], grid_extent_um[0], n_coarse)
    yg = np.linspace(-grid_extent_um[1], grid_extent_um[1], n_coarse)
    zg = np.linspace(-grid_extent_um[2], grid_extent_um[2], n_coarse)
    xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
    r_um = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    r_norm = r_um * 1e-6 / dl

    _, _, _, V_total = compute_potentials(
        potential_interps, field_interps, voltage_list, cfg, r_norm
    )

    idx_min = np.nanargmin(V_total)
    xc, yc, zc = r_um[idx_min]

    # 1D 精化
    from FieldParser.potential_fit import fit_potential_1d, get_center_and_k2
    from field_visualize.core import build_grid_1d

    for axis_idx, (axis_name, coarse_val, extent) in enumerate([
        ("x", xc, grid_extent_um[0]),
        ("y", yc, grid_extent_um[1]),
        ("z", zc, grid_extent_um[2]),
    ]):
        axis = ("x", "y", "z")[axis_idx]
        fine_extent = extent * 0.3  # 在粗最小值附近缩小范围
        coord_um = np.linspace(coarse_val - fine_extent, coarse_val + fine_extent, 200)
        vary_range = (
            um_to_norm(coord_um[0], dl),
            um_to_norm(coord_um[-1], dl),
        )
        consts = [um_to_norm(xc, dl), um_to_norm(yc, dl), um_to_norm(zc, dl)]
        r_fine = build_grid_1d(axis, vary_range, consts[0], consts[1], consts[2], 200)
        _, _, _, V_fine = compute_potentials(
            potential_interps, field_interps, voltage_list, cfg, r_fine
        )
        try:
            fit_result, _ = fit_potential_1d(coord_um, V_fine, degree=2)
            refined = fit_result[0]  # center
            if axis_idx == 0:
                xc = refined
            elif axis_idx == 1:
                yc = refined
            else:
                zc = refined
        except (ValueError, np.linalg.LinAlgError):
            pass  # 保持粗值

    return xc, yc, zc


def compute_stability_from_field(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    species: Species,
    center_um: tuple[float, float, float],
    fit_range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    n_pts: int = 200,
    fit_degree: int = 2,
) -> StabilityResult:
    """
    从实际场几何计算 Mathieu a/q 参数。

    分别拟合 DC 势和 RF 幅值势沿各轴的曲率，转换为 a 和 q 参数。

    Parameters
    ----------
    potential_interps : list
        势场插值器列表（calc_potential 返回）
    field_interps : list
        电场插值器列表（calc_field 返回）
    voltage_list : list[Voltage]
        电压列表
    cfg : Config
        无量纲化常数
    species : Species
        离子种类
    center_um : tuple[float, float, float]
        陷阱中心 (xc, yc, zc) (μm)
    fit_range_um : tuple
        ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)) 各轴拟合范围 (μm)
    n_pts : int
        每轴采样点数
    fit_degree : int
        多项式拟合阶数 (2 或 4)
    """
    from field_visualize.core import (
        compute_potentials,
        build_grid_1d,
        um_to_norm,
    )
    from FieldParser.potential_fit import (
        fit_potential_1d,
        get_center_and_k2,
        k2_to_trap_freq_MHz,
    )
    from FieldConfiguration.constants import m as DEFAULT_ION_MASS

    dl = cfg.dl
    Omega = cfg.Omega
    xc, yc, zc = center_um
    mass_kg = species.mass_kg

    # 质量修正因子：compute_potentials 内部用 Ba135 质量计算赝势
    mass_correction = DEFAULT_ION_MASS / mass_kg

    axes_cfg = [
        ("x", fit_range_um[0]),
        ("y", fit_range_um[1]),
        ("z", fit_range_um[2]),
    ]

    k2_dc: dict[str, float] = {}
    k2_rf_amp: dict[str, float] = {}
    k2_total: dict[str, float] = {}
    a_dict: dict[str, float] = {}
    q_dict: dict[str, float] = {}
    f_trap: dict[str, float] = {}

    for axis, (lo, hi) in axes_cfg:
        coord_um = np.linspace(lo, hi, n_pts)
        vary_range = (um_to_norm(lo, dl), um_to_norm(hi, dl))
        xc_n = um_to_norm(xc, dl)
        yc_n = um_to_norm(yc, dl)
        zc_n = um_to_norm(zc, dl)

        r = build_grid_1d(axis, vary_range, xc_n, yc_n, zc_n, n_pts)
        V_dc, V_rf_amp, V_pseudo, V_total = compute_potentials(
            potential_interps, field_interps, voltage_list, cfg, r
        )

        # 修正赝势中的质量
        V_pseudo_corr = V_pseudo * mass_correction
        V_total_corr = V_dc + V_pseudo_corr

        # 拟合 DC 势
        try:
            fit_dc, _ = fit_potential_1d(coord_um, V_dc, degree=fit_degree)
            _, k2 = get_center_and_k2(fit_dc, fit_degree)
            k2_dc[axis] = k2
        except (ValueError, np.linalg.LinAlgError):
            k2_dc[axis] = 0.0

        # 拟合 RF 幅值势
        try:
            fit_rf, _ = fit_potential_1d(coord_um, V_rf_amp, degree=fit_degree)
            _, k2 = get_center_and_k2(fit_rf, fit_degree)
            k2_rf_amp[axis] = k2
        except (ValueError, np.linalg.LinAlgError):
            k2_rf_amp[axis] = 0.0

        # 拟合总势（修正后）
        try:
            fit_total, _ = fit_potential_1d(coord_um, V_total_corr, degree=fit_degree)
            _, k2 = get_center_and_k2(fit_total, fit_degree)
            k2_total[axis] = k2
            f_trap[axis] = k2_to_trap_freq_MHz(k2, mass_kg)
        except (ValueError, np.linalg.LinAlgError):
            k2_total[axis] = 0.0
            f_trap[axis] = float("nan")

        # 从 DC 曲率计算 a
        # k2 是 V''(center)/2 的系数，单位 V/μm²
        # V'' = 2*k2，转为 SI: V''_SI = 2*k2*1e12 (V/m²)
        k2_dc_si = k2_dc[axis] * 1e12 * 2  # V/m²
        a_dict[axis] = EC * k2_dc_si / (mass_kg * Omega**2 / 4)
        # 简化: a = 4*e*V''_SI/(m*Omega²) = 4*e*(2*k2*1e12)/(m*Omega²)

        # 从 RF 幅值曲率计算 q
        k2_rf_si = k2_rf_amp[axis] * 1e12 * 2  # V/m²
        # q = (2e/mOmega²) * V''_RF = 4e*k2_rf_si / (m*Omega²)
        q_dict[axis] = 4 * EC * k2_rf_si / (mass_kg * Omega**2)

    # Secular 频率（绝热近似）
    f_sec = {axis: _secular_freq_mhz(a_dict[axis], q_dict[axis], Omega)
             for axis in ("x", "y", "z")}

    is_stable, stability_note = check_stability_region(
        a_dict["x"], a_dict["y"], a_dict["z"],
        q_dict["x"], q_dict["y"], q_dict["z"],
    )

    return StabilityResult(
        a_x=a_dict["x"], a_y=a_dict["y"], a_z=a_dict["z"],
        q_x=q_dict["x"], q_y=q_dict["y"], q_z=q_dict["z"],
        f_sec_x=f_sec["x"], f_sec_y=f_sec["y"], f_sec_z=f_sec["z"],
        f_trap_x=f_trap["x"], f_trap_y=f_trap["y"], f_trap_z=f_trap["z"],
        is_stable=is_stable,
        stability_note=stability_note,
        species_name=species.name,
        mass_amu=species.mass_amu,
        omega_rf=Omega,
        freq_rf_MHz=cfg.freq_RF,
        k2_dc=k2_dc,
        k2_rf_amp=k2_rf_amp,
    )
