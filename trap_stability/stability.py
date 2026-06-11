"""
Mathieu 稳定性参数 (a, q) 核心计算

从实际场几何计算 Mathieu a/q 参数、secular 频率、非谐常数、稳定性判断。
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
    # 非谐常数（无量纲，Taylor 系数 c_{2k}·dl^{2k}/dV）
    # c_{2k} = Phi^{(2k)}(center) / (2k)!，含 1/(2k)! 因子
    anh4_dc: dict[str, float] | None = None
    anh4_rf: dict[str, float] | None = None
    anh6_dc: dict[str, float] | None = None
    anh6_rf: dict[str, float] | None = None


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
    fit_degree: int = 6,
) -> StabilityResult:
    """
    从实际场几何计算 Mathieu a/q 参数及非谐常数。

    分别拟合 DC 势和 RF 幅值势沿各轴的曲率，转换为 a 和 q 参数。
    同时进行多项式拟合（阶数由 fit_degree 控制），提取无量纲非谐常数。

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
        多项式拟合最高阶数 (2, 4 或 6)。
        - 2: 仅计算 a, q
        - 4: 额外输出 4 阶非谐常数
        - 6: 额外输出 4 阶和 6 阶非谐常数
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

    if fit_degree not in (2, 4, 6):
        raise ValueError(f"fit_degree 须为 2, 4 或 6，收到 {fit_degree}")

    dl = cfg.dl
    dV = cfg.dV
    Omega = cfg.Omega
    xc, yc, zc = center_um
    mass_kg = species.mass_kg

    # 质量修正因子：compute_potentials 内部用 Ba135 质量计算赝势
    mass_correction = DEFAULT_ION_MASS / mass_kg

    # dl 单位为 m，转换为 μm 用于与 V/μm^k 系数相乘
    dl_um = dl * 1e6

    axes_cfg = [
        ("x", fit_range_um[0]),
        ("y", fit_range_um[1]),
        ("z", fit_range_um[2]),
    ]

    k2_dc: dict[str, float] = {}
    k2_rf_amp: dict[str, float] = {}
    a_dict: dict[str, float] = {}
    q_dict: dict[str, float] = {}
    f_trap: dict[str, float] = {}
    # 仅在对应阶数可用时初始化为 dict，否则保持 None
    anh4_dc: dict[str, float] | None = {} if fit_degree >= 4 else None
    anh4_rf: dict[str, float] | None = {} if fit_degree >= 4 else None
    anh6_dc: dict[str, float] | None = {} if fit_degree >= 6 else None
    anh6_rf: dict[str, float] | None = {} if fit_degree >= 6 else None

    # 中心坐标归一化（循环外计算一次）
    xc_n = um_to_norm(xc, dl)
    yc_n = um_to_norm(yc, dl)
    zc_n = um_to_norm(zc, dl)

    def _fit_and_extract(
        potential: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        多项式拟合，提取 k2 及无量纲非谐常数。

        Returns
        -------
        (k2, anh4, anh6)
            k2 : 二次曲率系数 (V/μm²)，始终返回
            anh4 : 无量纲 4 阶非谐常数（fit_degree >= 4 时有效，否则 0.0）
            anh6 : 无量纲 6 阶非谐常数（fit_degree >= 6 时有效，否则 0.0）

        Notes
        -----
        Taylor 展开 V(x) = c₂x² + c₄x⁴ + c₆x⁶ + ...
        其中 c_{2k} = Φ^{(2k)}(center)/(2k)!，已含 1/(2k)! 因子。
        无量纲化：c_{2k} → c_{2k}·dl^{2k}/dV。
        """
        try:
            fit, _ = fit_potential_1d(coord_um, potential, degree=fit_degree)
        except (ValueError, np.linalg.LinAlgError):
            # 拟合失败，退回 2 阶
            try:
                fit2, _ = fit_potential_1d(coord_um, potential, degree=2)
                _, k2 = get_center_and_k2(fit2, 2)
                return k2, 0.0, 0.0
            except (ValueError, np.linalg.LinAlgError):
                return 0.0, 0.0, 0.0

        center = float(fit[0])
        coefs = fit[1:]  # [a0, a1, ..., a_d]
        d = len(coefs) - 1  # 实际拟合阶数
        c = center

        # k2 = V''(c)/2! = Σ_{k≥2} k(k-1)/2 · a_k · c^{k-2}
        k2 = sum(k * (k - 1) * coefs[k] * c ** (k - 2)
                 for k in range(2, d + 1)) / 2

        # 4 阶非谐常数
        anh4_val = 0.0
        if fit_degree >= 4 and d >= 4:
            # c4 = V^{(4)}(c)/4! = Σ_{k≥4} k!/(k-4)!/24 · a_k · c^{k-4}
            c4 = sum(
                k * (k - 1) * (k - 2) * (k - 3) * coefs[k] * c ** (k - 4)
                for k in range(4, d + 1)
            ) / 24
            anh4_val = c4 * dl_um ** 4 / dV

        # 6 阶非谐常数
        anh6_val = 0.0
        if fit_degree >= 6 and d >= 6:
            c6 = coefs[6]  # V^{(6)}/6! = a6
            anh6_val = c6 * dl_um ** 6 / dV

        return k2, anh4_val, anh6_val

    # 公共分母
    denom = mass_kg * Omega**2

    for axis, (lo, hi) in axes_cfg:
        coord_um = np.linspace(lo, hi, n_pts)
        vary_range = (um_to_norm(lo, dl), um_to_norm(hi, dl))

        r = build_grid_1d(axis, vary_range, xc_n, yc_n, zc_n, n_pts)
        V_dc, V_rf_amp, V_pseudo, _ = compute_potentials(
            potential_interps, field_interps, voltage_list, cfg, r
        )

        # 修正赝势中的质量
        V_total_corr = V_dc + V_pseudo * mass_correction

        # 拟合曲率 + 非谐常数
        k2_dc_val, a4, a6 = _fit_and_extract(V_dc)
        k2_rf_val, q4, q6 = _fit_and_extract(V_rf_amp)
        k2_total, _, _ = _fit_and_extract(V_total_corr)

        k2_dc[axis] = k2_dc_val
        k2_rf_amp[axis] = k2_rf_val
        if anh4_dc is not None:
            anh4_dc[axis] = a4
            anh4_rf[axis] = q4
        if anh6_dc is not None:
            anh6_dc[axis] = a6
            anh6_rf[axis] = q6

        f_trap[axis] = k2_to_trap_freq_MHz(k2_total, mass_kg) if k2_total != 0.0 else float("nan")

        # a = 4eV''/(mOmega^2)，其中 V'' = 2k2（k2 为 x^2 系数）
        a_dict[axis] = 8 * EC * k2_dc[axis] * 1e12 / denom

        # q = 4ek2/(mOmega^2)，其中 k2 为 RF 幅值势的 x^2 系数
        q_dict[axis] = 4 * EC * k2_rf_amp[axis] * 1e12 / denom

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
        anh4_dc=anh4_dc,
        anh4_rf=anh4_rf,
        anh6_dc=anh6_dc,
        anh6_rf=anh6_rf,
    )
