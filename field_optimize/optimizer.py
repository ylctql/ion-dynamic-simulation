"""
核心优化循环：scipy.optimize.minimize 包装

从初始电压配置出发，搜索使阱频匹配目标值的电极电压。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize

from FieldConfiguration.field_settings import voltage_dc, voltage_rf
from field_visualize.core import is_rf
from field_visualize.symmetry import compute_symmetry_report
from field_visualize.trap_freq import compute_trap_freqs_at_point

from .objective import FastEvaluator, build_fast_evaluator, compute_objective
from .types import OptimizationConfig, OptimizationResult

logger = logging.getLogger(__name__)


def _voltage_to_dict(v) -> dict[str, Any]:
    """Voltage 对象序列化为 dict"""
    return {
        "name": v.name,
        "type": "rf" if is_rf(v) else "dc",
        "V_bias": v.V_bias,
        "V0": v.V0,
    }


def _build_result_voltages(
    voltage_list: list,
    evaluator: FastEvaluator,
    x_opt: np.ndarray,
    opt_config: OptimizationConfig,
) -> list:
    """从优化结果参数向量重构 Voltage 列表"""
    from utils import Voltage

    # 构建 V_bias 完整向量
    V_bias_full = evaluator.initial_V_biases.copy()
    for idx_local, idx_global in enumerate(evaluator.dc_indices):
        V_bias_full[idx_global] = x_opt[idx_local]

    # RF V0
    V0_rf = evaluator.initial_V0_rf
    if opt_config.optimize_rf_v0 and evaluator.rf_indices:
        V0_rf = x_opt[len(evaluator.dc_indices)]

    new_list: list[Voltage] = []
    for i, v_orig in enumerate(voltage_list):
        if is_rf(v_orig):
            new_v0 = V0_rf if V0_rf is not None else v_orig.V0
            # 保持原有频率函数
            new_list.append(Voltage(v_orig.name, new_v0, v_orig.f, V_bias_full[i]))
        else:
            new_list.append(Voltage(v_orig.name, 0.0, v_orig.f, V_bias_full[i]))

    return new_list


def _compute_initial_final_symmetry(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    opt_config: OptimizationConfig,
) -> dict[str, Any] | None:
    """用完整 pipeline 计算对称性度量"""
    try:
        report = compute_symmetry_report(
            potential_interps,
            field_interps,
            voltage_list,
            cfg,
            center_um=opt_config.center_um,
            range_um=opt_config.fit_range_um,
            which=frozenset("ph"),
            n_fit_pts=opt_config.symmetry_n_pts,
            fit_mode=opt_config.symmetry_fit_mode,
        )
        result: dict[str, Any] = {}
        total = report.total
        if total.polynomial is not None:
            result["s_parity_yz"] = total.polynomial.s_parity_yz
            result["s_parity_xz"] = total.polynomial.s_parity_xz
            result["s_parity_xy"] = total.polynomial.s_parity_xy
        if total.hessian is not None:
            result["offdiag_ratio"] = total.hessian.offdiag_ratio
            result["kappa_xx"] = total.hessian.kappa_xx
            result["kappa_yy"] = total.hessian.kappa_yy
            result["kappa_zz"] = total.hessian.kappa_zz
        return result
    except Exception as e:
        logger.warning("对称性计算失败: %s", e)
        return None


def optimize_voltages(
    potential_interps: list,
    field_interps: list,
    cfg,
    voltage_list: list,
    opt_config: OptimizationConfig,
) -> OptimizationResult:
    """
    优化电极电压以匹配目标阱频。

    Parameters
    ----------
    potential_interps, field_interps : 插值器列表
    cfg : Config
    voltage_list : 初始电压列表
    opt_config : 优化配置（含目标频率、权重等）

    Returns
    -------
    OptimizationResult
    """
    # 1. 构建快速评估器
    evaluator = build_fast_evaluator(
        potential_interps, field_interps, voltage_list, cfg, opt_config
    )

    # 2. 初始频率（用完整 pipeline 计算，用于报告）
    center = opt_config.center_um
    fit_range = opt_config.fit_range_um
    initial_freqs = compute_trap_freqs_at_point(
        potential_interps,
        field_interps,
        voltage_list,
        cfg,
        xc_um=center[0],
        yc_um=center[1],
        zc_um=center[2],
        x_range_um=fit_range[0],
        y_range_um=fit_range[1],
        z_range_um=fit_range[2],
        n_pts=opt_config.n_fit_pts,
        fit_degree=opt_config.fit_degree,
    )

    # 3. 初始目标函数值
    x0_dc = np.array([voltage_list[i].V_bias for i in evaluator.dc_indices])
    x0_list = [x0_dc]
    if opt_config.optimize_rf_v0 and evaluator.rf_indices:
        x0_list.append(np.array([evaluator.initial_V0_rf]))
    x0 = np.concatenate(x0_list)

    initial_obj = compute_objective(x0, evaluator, opt_config)

    # 4. 构建边界
    bounds = [opt_config.v_bias_bounds] * len(evaluator.dc_indices)
    if opt_config.optimize_rf_v0 and evaluator.rf_indices:
        bounds.append(opt_config.v0_rf_bounds)

    # Nelder-Mead 不支持 bounds
    method = opt_config.method
    if method == "Nelder-Mead":
        bounds = None  # type: ignore[assignment]

    # 5. 优化
    logger.info(
        "开始优化: %d DC 变量%s, 目标频率 %.3f / %.3f / %.3f MHz",
        len(evaluator.dc_indices),
        " + RF V0" if opt_config.optimize_rf_v0 and evaluator.rf_indices else "",
        opt_config.target_freq_MHz[0],
        opt_config.target_freq_MHz[1],
        opt_config.target_freq_MHz[2],
    )

    res = minimize(
        fun=compute_objective,
        x0=x0,
        args=(evaluator, opt_config),
        method=method,
        bounds=bounds,
        options={"maxiter": opt_config.maxiter, "ftol": opt_config.tol},
    )

    logger.info("优化完成: %s (nit=%d, nfev=%d)", res.message, res.nit, res.nfev)

    # 6. 重构优化后的 voltage_list
    optimized_vlist = _build_result_voltages(voltage_list, evaluator, res.x, opt_config)

    # 7. 最终频率
    final_freqs = compute_trap_freqs_at_point(
        potential_interps,
        field_interps,
        optimized_vlist,
        cfg,
        xc_um=center[0],
        yc_um=center[1],
        zc_um=center[2],
        x_range_um=fit_range[0],
        y_range_um=fit_range[1],
        z_range_um=fit_range[2],
        n_pts=opt_config.n_fit_pts,
        fit_degree=opt_config.fit_degree,
    )

    # 8. 对称性度量
    need_sym = opt_config.w_parity > 0 or opt_config.w_offdiag > 0
    initial_symmetry = None
    optimized_symmetry = None
    if need_sym:
        initial_symmetry = _compute_initial_final_symmetry(
            potential_interps, field_interps, voltage_list, cfg, opt_config
        )
        optimized_symmetry = _compute_initial_final_symmetry(
            potential_interps, field_interps, optimized_vlist, cfg, opt_config
        )

    # 9. 最终目标函数值
    final_obj = float(res.fun)

    return OptimizationResult(
        success=res.success,
        message=str(res.message),
        n_iterations=res.nit,
        n_evaluations=res.nfev,
        initial_voltages=[_voltage_to_dict(v) for v in voltage_list],
        optimized_voltages=[_voltage_to_dict(v) for v in optimized_vlist],
        initial_freqs_MHz=initial_freqs,
        optimized_freqs_MHz=final_freqs,
        target_freqs_MHz={
            "f_x": opt_config.target_freq_MHz[0],
            "f_y": opt_config.target_freq_MHz[1],
            "f_z": opt_config.target_freq_MHz[2],
        },
        initial_objective=float(initial_obj),
        final_objective=final_obj,
        initial_symmetry=initial_symmetry,
        optimized_symmetry=optimized_symmetry,
    )
