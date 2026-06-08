"""
目标函数与快速预计算：频率误差 + 对称性惩罚

核心思路：势场是基函数的线性叠加，插值器只构建一次。
预计算基函数在采样点的值，优化循环中仅做矩阵-向量乘法。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import e as ELEMENTARY_CHARGE

from FieldConfiguration.constants import m as ION_MASS
from FieldParser.potential_fit import (
    fit_potential_1d,
    get_center_and_k2,
    k2_to_trap_freq_MHz,
)
from field_visualize.core import (
    AXIS_INDEX,
    build_grid_1d,
    is_rf,
    um_to_norm,
)

from .types import OptimizationConfig


# ---------------------------------------------------------------------------
# 预计算数据结构
# ---------------------------------------------------------------------------
@dataclass
class FastEvaluator:
    """优化循环中用到的预计算数据"""

    # 1D 采样（频率计算用）
    # axis ("x"/"y"/"z") → (n_pts, n_electrodes)
    phi_1d: dict[str, np.ndarray]
    # axis → (n_pts, 3, n_electrodes)
    e_1d: dict[str, np.ndarray]
    # axis → (n_pts,) 坐标 µm
    coord_um_1d: dict[str, np.ndarray]

    # RF 赝势常量（DC-only 模式下 V0 固定，axis → (n_pts,)）
    v_pseudo_const: dict[str, np.ndarray] | None

    # 3D 采样（对称性惩罚用），若不需要则为 None
    phi_3d: np.ndarray | None      # (n_total, n_electrodes)
    e_3d: np.ndarray | None        # (n_total, 3, n_electrodes)
    v_pseudo_3d_const: np.ndarray | None  # (n_total,)
    # 3D 网格坐标 µm，用于 fit_potential_3d_quartic
    grid_3d_um: np.ndarray | None  # (n_total, 3)

    # 电极分类
    dc_indices: list[int]
    rf_indices: list[int]
    n_electrodes: int
    initial_V_biases: np.ndarray   # (n_electrodes,) 全部初始 V_bias
    initial_V0_rf: float | None    # RF 电极的 V0（无 RF 则为 None）

    # 物理常量
    dl: float
    dV: float
    pseudo_coeff: float  # q² / (4 m Ω²) × (dV/dl)² / q = e / (4 m Ω²) × (dV/dl)²


# ---------------------------------------------------------------------------
# 预计算构建
# ---------------------------------------------------------------------------
def build_fast_evaluator(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    opt_config: OptimizationConfig,
) -> FastEvaluator:
    """
    构建快速评估器：预计算所有基函数在采样点的值。

    Parameters
    ----------
    potential_interps, field_interps : 插值器列表（calc_potential / calc_field 输出）
    voltage_list : 初始电压列表
    cfg : Config (dl, dV, Omega, ...)
    opt_config : 优化配置
    """
    dl, dV = cfg.dl, cfg.dV
    Omega = cfg.Omega

    n_elec = len(potential_interps)

    # 电极分类
    dc_indices: list[int] = []
    rf_indices: list[int] = []
    initial_V_biases = np.zeros(n_elec)
    initial_V0_rf: float | None = None

    for i, v in enumerate(voltage_list):
        initial_V_biases[i] = v.V_bias
        if is_rf(v):
            rf_indices.append(i)
            initial_V0_rf = v.V0
        else:
            dc_indices.append(i)

    # 赝势系数: V_pseudo = pseudo_coeff * |E_rf_si|²
    # E_rf_si = (Σ V0_i × E_basis_i) × (dV/dl)
    # coeff_J = e² / (4 m Ω²),  V_pseudo(V) = coeff_J / e × |E_rf_si|²
    #  = e / (4 m Ω²) × (dV/dl)² × |Σ V0_i × E_basis_i|²
    pseudo_coeff = ELEMENTARY_CHARGE / (4 * ION_MASS * Omega**2) * (dV / dl) ** 2

    # 中心点（归一化）
    center_um = opt_config.center_um
    xc = um_to_norm(center_um[0], dl)
    yc = um_to_norm(center_um[1], dl)
    zc = um_to_norm(center_um[2], dl)

    n_pts = opt_config.n_fit_pts

    # ---- 1D 预计算 ----
    phi_1d: dict[str, np.ndarray] = {}
    e_1d: dict[str, np.ndarray] = {}
    coord_um_1d: dict[str, np.ndarray] = {}
    v_pseudo_const: dict[str, np.ndarray] | None = None

    if not opt_config.optimize_rf_v0 and rf_indices:
        v_pseudo_const = {}

    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        r_range_um = opt_config.fit_range_um[axis_idx]
        r_range_norm = (
            um_to_norm(r_range_um[0], dl),
            um_to_norm(r_range_um[1], dl),
        )
        # 归一化采样网格
        r_norm = build_grid_1d(axis_name, r_range_norm, xc, yc, zc, n_pts)
        # µm 坐标
        coord_um = np.linspace(r_range_um[0], r_range_um[1], n_pts)

        # 评估所有基函数
        phi = np.zeros((n_pts, n_elec))
        e_basis = np.zeros((n_pts, 3, n_elec))
        for i in range(n_elec):
            phi[:, i] = np.atleast_1d(potential_interps[i](r_norm)).ravel() * dV
            e_basis[:, :, i] = field_interps[i](r_norm)

        phi_1d[axis_name] = phi
        e_1d[axis_name] = e_basis
        coord_um_1d[axis_name] = coord_um

        # 计算固定 RF 赝势
        if v_pseudo_const is not None and rf_indices:
            E_rf = np.zeros((n_pts, 3))
            for ri in rf_indices:
                E_rf += voltage_list[ri].V0 * e_basis[:, :, ri]
            v_pseudo_const[axis_name] = pseudo_coeff * np.sum(E_rf**2, axis=1)

    # ---- 3D 预计算（对称性惩罚用）----
    phi_3d: np.ndarray | None = None
    e_3d: np.ndarray | None = None
    v_pseudo_3d_const: np.ndarray | None = None
    grid_3d_um: np.ndarray | None = None

    need_3d = opt_config.w_parity > 0 or opt_config.w_offdiag > 0
    if need_3d:
        sn = opt_config.symmetry_n_pts
        x_lin = np.linspace(opt_config.fit_range_um[0][0], opt_config.fit_range_um[0][1], sn)
        y_lin = np.linspace(opt_config.fit_range_um[1][0], opt_config.fit_range_um[1][1], sn)
        z_lin = np.linspace(opt_config.fit_range_um[2][0], opt_config.fit_range_um[2][1], sn)
        xx, yy, zz = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
        n_total = xx.size
        grid_3d_um_arr = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # 转归一化坐标
        grid_3d_norm = grid_3d_um_arr * 1e-6 / dl  # (n_total, 3)

        phi_3d_arr = np.zeros((n_total, n_elec))
        e_3d_arr = np.zeros((n_total, 3, n_elec))
        for i in range(n_elec):
            phi_3d_arr[:, i] = np.atleast_1d(potential_interps[i](grid_3d_norm)).ravel() * dV
            e_3d_arr[:, :, i] = field_interps[i](grid_3d_norm)

        phi_3d = phi_3d_arr
        e_3d = e_3d_arr
        grid_3d_um = grid_3d_um_arr

        if not opt_config.optimize_rf_v0 and rf_indices:
            E_rf_3d = np.zeros((n_total, 3))
            for ri in rf_indices:
                E_rf_3d += voltage_list[ri].V0 * e_3d_arr[:, :, ri]
            v_pseudo_3d_const = pseudo_coeff * np.sum(E_rf_3d**2, axis=1)

    return FastEvaluator(
        phi_1d=phi_1d,
        e_1d=e_1d,
        coord_um_1d=coord_um_1d,
        v_pseudo_const=v_pseudo_const,
        phi_3d=phi_3d,
        e_3d=e_3d,
        v_pseudo_3d_const=v_pseudo_3d_const,
        grid_3d_um=grid_3d_um,
        dc_indices=dc_indices,
        rf_indices=rf_indices,
        n_electrodes=n_elec,
        initial_V_biases=initial_V_biases,
        initial_V0_rf=initial_V0_rf,
        dl=dl,
        dV=dV,
        pseudo_coeff=pseudo_coeff,
    )


# ---------------------------------------------------------------------------
# 目标函数
# ---------------------------------------------------------------------------
def _compute_freqs_fast(
    x: np.ndarray,
    evaluator: FastEvaluator,
    opt_config: OptimizationConfig,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    从参数向量快速计算阱频。

    Returns
    -------
    freqs : {f_x, f_y, f_z} MHz
    v_total_1d : {axis: V_total array}  用于诊断
    """
    # 构建 V_bias 完整向量
    V_bias_full = evaluator.initial_V_biases.copy()
    for idx_local, idx_global in enumerate(evaluator.dc_indices):
        V_bias_full[idx_global] = x[idx_local]

    # RF V0（如果在优化中）
    V0_rf = evaluator.initial_V0_rf
    if opt_config.optimize_rf_v0 and evaluator.rf_indices:
        V0_rf = x[len(evaluator.dc_indices)]

    freqs: dict[str, float] = {}
    v_total_1d: dict[str, np.ndarray] = {}

    for axis_name in ["x", "y", "z"]:
        phi = evaluator.phi_1d[axis_name]
        coord_um = evaluator.coord_um_1d[axis_name]

        # DC 势
        V_dc = phi @ V_bias_full

        # RF 赝势
        if not opt_config.optimize_rf_v0:
            V_pseudo = (
                evaluator.v_pseudo_const[axis_name]
                if evaluator.v_pseudo_const is not None
                else np.zeros_like(V_dc)
            )
        else:
            if evaluator.rf_indices:
                e_basis = evaluator.e_1d[axis_name]
                E_rf = np.zeros_like(e_basis[:, :, 0])
                for ri in evaluator.rf_indices:
                    E_rf += V0_rf * e_basis[:, :, ri]
                V_pseudo = evaluator.pseudo_coeff * np.sum(E_rf**2, axis=1)
            else:
                V_pseudo = np.zeros_like(V_dc)

        V_total = V_dc + V_pseudo
        v_total_1d[axis_name] = V_total

        # 1D 多项式拟合 → 阱频
        try:
            fit_result, _ = fit_potential_1d(coord_um, V_total, degree=opt_config.fit_degree)
            _, k2 = get_center_and_k2(fit_result, opt_config.fit_degree)
            freqs[f"f_{axis_name}"] = k2_to_trap_freq_MHz(k2, ION_MASS)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            freqs[f"f_{axis_name}"] = float("nan")

    return freqs, v_total_1d


def _compute_symmetry_penalty(
    x: np.ndarray,
    evaluator: FastEvaluator,
    opt_config: OptimizationConfig,
) -> tuple[float, dict[str, float]]:
    """
    计算对称性惩罚项（多项式奇偶性 + Hessian 离轴比）。

    Returns
    -------
    penalty : 标量惩罚值
    metrics : 详细度量字典
    """
    if evaluator.phi_3d is None or evaluator.grid_3d_um is None:
        return 0.0, {}

    from equilibrium.potential_fit_3d import fit_potential_3d_quartic, hessian_fit_3d

    # 构建 V_bias 完整向量
    V_bias_full = evaluator.initial_V_biases.copy()
    for idx_local, idx_global in enumerate(evaluator.dc_indices):
        V_bias_full[idx_global] = x[idx_local]

    V0_rf = evaluator.initial_V0_rf
    if opt_config.optimize_rf_v0 and evaluator.rf_indices:
        V0_rf = x[len(evaluator.dc_indices)]

    # 计算 3D 总势
    V_dc_3d = evaluator.phi_3d @ V_bias_full
    if not opt_config.optimize_rf_v0:
        V_pseudo_3d = (
            evaluator.v_pseudo_3d_const
            if evaluator.v_pseudo_3d_const is not None
            else np.zeros_like(V_dc_3d)
        )
    else:
        if evaluator.rf_indices:
            E_rf_3d = np.zeros_like(evaluator.e_3d[:, :, 0])  # type: ignore[index]
            for ri in evaluator.rf_indices:
                E_rf_3d += V0_rf * evaluator.e_3d[:, :, ri]  # type: ignore[index]
            V_pseudo_3d = evaluator.pseudo_coeff * np.sum(E_rf_3d**2, axis=1)
        else:
            V_pseudo_3d = np.zeros_like(V_dc_3d)

    V_total_3d = V_dc_3d + V_pseudo_3d

    # 包装为 compute_V_total 形式的 callable（归一化坐标输入）
    # fit_potential_3d_quartic 需要 compute_V_total(r_norm) -> V
    # 这里我们直接在 um 网格上做，所以传一个从 um 网格映射的函数
    grid_um = evaluator.grid_3d_um
    n_total = grid_um.shape[0]
    _v_total_ref = V_total_3d  # 闭包捕获

    # 由于网格是固定的，直接用预计算的势值
    # fit_potential_3d_quartic 内部会再次采样 compute_V_total
    # 为了避免重复插值，我们直接用网格数据
    # 但 fit_potential_3d_quartic 接受 compute_V_total(r_norm) -> V
    # 我们构建一个简单的查表函数
    from scipy.interpolate import RegularGridInterpolator

    sn = opt_config.symmetry_n_pts
    x_lin = np.linspace(opt_config.fit_range_um[0][0], opt_config.fit_range_um[0][1], sn)
    y_lin = np.linspace(opt_config.fit_range_um[1][0], opt_config.fit_range_um[1][1], sn)
    z_lin = np.linspace(opt_config.fit_range_um[2][0], opt_config.fit_range_um[2][1], sn)
    V_3d_grid = V_total_3d.reshape(sn, sn, sn, order="C")
    # 插值器接受 um 坐标，输出 V
    interp_3d = RegularGridInterpolator(
        (x_lin, y_lin, z_lin), V_3d_grid, method="linear", bounds_error=False, fill_value=None
    )

    dl = evaluator.dl

    def compute_V_for_fit(r_norm: np.ndarray) -> np.ndarray:
        r_um = r_norm * dl / 1e-6  # 归一化 → µm
        return interp_3d(r_um)

    try:
        fit = fit_potential_3d_quartic(
            compute_V_total=compute_V_for_fit,
            um_to_norm=lambda v: um_to_norm(v, dl),
            center_um=opt_config.center_um,
            range_um=opt_config.fit_range_um,
            n_pts_per_axis=sn,
            fit_mode=opt_config.symmetry_fit_mode,
        )
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return 0.0, {}

    coeffs = fit.coeffs  # (5, 5, 5)
    metrics: dict[str, float] = {}

    # 奇偶性惩罚
    parity_penalty = 0.0
    if opt_config.w_parity > 0:
        from field_visualize.symmetry import _parity_coefficient

        s_yz = _parity_coefficient(coeffs, 0)
        s_xz = _parity_coefficient(coeffs, 1)
        s_xy = _parity_coefficient(coeffs, 2)
        parity_penalty = (1 - s_yz) + (1 - s_xz) + (1 - s_xy)
        metrics["s_parity_yz"] = s_yz
        metrics["s_parity_xz"] = s_xz
        metrics["s_parity_xy"] = s_xy

    # Hessian 离轴比惩罚
    offdiag_penalty = 0.0
    if opt_config.w_offdiag > 0:
        center_arr = np.array([opt_config.center_um])
        hess = hessian_fit_3d(fit, center_arr)[0]  # (3, 3) V/µm²
        diag = np.abs(np.array([hess[0, 0], hess[1, 1], hess[2, 2]]))
        offdiag = np.abs(np.array([hess[0, 1], hess[0, 2], hess[1, 2]]))
        mean_diag = np.mean(diag) + 1e-30
        offdiag_ratio = float(np.max(offdiag) / mean_diag)
        offdiag_penalty = offdiag_ratio
        metrics["offdiag_ratio"] = offdiag_ratio
        metrics["kappa_xx"] = float(hess[0, 0])
        metrics["kappa_yy"] = float(hess[1, 1])
        metrics["kappa_zz"] = float(hess[2, 2])

    penalty = opt_config.w_parity * parity_penalty + opt_config.w_offdiag * offdiag_penalty
    return penalty, metrics


def compute_objective(
    x: np.ndarray,
    evaluator: FastEvaluator,
    opt_config: OptimizationConfig,
) -> float:
    """
    目标函数：频率误差 + 对称性惩罚。

    Parameters
    ----------
    x : 参数向量 [V_bias_1, ..., V_bias_N_dc, (V0_rf)]
    evaluator : 预计算数据
    opt_config : 优化配置
    """
    # 频率计算
    freqs, _ = _compute_freqs_fast(x, evaluator, opt_config)

    # NaN 保护：anti-trapping 返回大惩罚
    f_vals = [freqs.get(f"f_{a}", float("nan")) for a in "xyz"]
    if any(np.isnan(f) for f in f_vals):
        return 1e6 + float(np.sum(np.abs(x)))

    # 频率损失（相对误差平方和）
    target = opt_config.target_freq_MHz
    freq_loss = sum(
        ((f_vals[i] - target[i]) / (target[i] + 1e-30)) ** 2 for i in range(3)
    )

    # 对称性惩罚
    sym_penalty = 0.0
    if opt_config.w_parity > 0 or opt_config.w_offdiag > 0:
        sym_penalty, _ = _compute_symmetry_penalty(x, evaluator, opt_config)

    return opt_config.w_freq * freq_loss + sym_penalty
