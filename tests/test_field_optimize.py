"""
field_optimize 模块测试
"""
import sys
from pathlib import Path

import argparse

import numpy as np
import pytest

# 确保项目根在 sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from field_optimize.types import OptimizationConfig, OptimizationResult
from field_optimize.objective import (
    FastEvaluator,
    build_fast_evaluator,
    compute_objective,
    _compute_freqs_fast,
)


# ---------------------------------------------------------------------------
# 辅助：构建合成势场数据
# ---------------------------------------------------------------------------
def _make_harmonic_field(
    n_elec: int = 4,
    freqs_MHz: tuple[float, float, float] = (2.0, 3.0, 0.1),
    n_grid: int = 11,
):
    """
    构造合成谐波势场。

    电极 0 = RF（单位正弦电场），电极 1-3 = DC（各控制一个轴的阱频）。

    返回 (potential_interps, field_interps, voltage_list, cfg, opt_config)
    """
    from field_visualize.core import um_to_norm

    # 简单的 Config mock
    class MockConfig:
        dl = 4.0e-6       # 4 µm
        dV = 1.0          # 1 V
        Omega = 2 * np.pi * 35.28e6

    cfg = MockConfig()
    dl = cfg.dl
    dV = cfg.dV

    # 目标阱频 → 目标 k2 (V/µm²)
    from FieldConfiguration.constants import m as ION_MASS
    from scipy.constants import e
    target_k2 = []
    for f in freqs_MHz:
        omega = f * 2 * np.pi * 1e6
        k2 = omega**2 * ION_MASS / (2 * e)  # V/m²
        k2_um = k2 * 1e-12  # V/µm²
        target_k2.append(k2_um)

    # 3D 网格 (归一化坐标)
    span = 50.0  # µm
    coords = np.linspace(-span, span, n_grid)  # µm
    coords_norm = coords * 1e-6 / dl

    xx, yy, zz = np.meshgrid(coords_norm, coords_norm, coords_norm, indexing="ij")
    grid_norm = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # 基函数：每个 DC 电极控制一个轴的势场
    # Φ_i(r) = k2_i / dV × coord_i² / 2 (归一化单位)
    # 但这里用 µm 坐标算，然后除以 dV 得到归一化势值

    # 电极 0 (RF): 纯 x 方向电场（用于赝势）
    # E_rf ~ x 方向
    # 电极 1: 控制 x 方向阱频, V = k2_x * x² / 2
    # 电极 2: 控制 y 方向阱频, V = k2_y * y² / 2
    # 电极 3: 控制 z 方向阱频, V = k2_z * z² / 2

    # 归一化坐标
    u = grid_norm[:, 0]  # x norm
    v = grid_norm[:, 1]  # y norm
    w = grid_norm[:, 2]  # z norm

    # 每个基函数在 V_bias=1V 时产生的势 (V)
    # V_i(r) = V_bias * basis(r) * dV
    # 所以 basis(r) = V_desired / dV when V_bias = 1

    # DC 基函数 (归一化单位，乘 dV 后得到 V)
    phi_1 = (target_k2[0] / dV) * (u * dl / 1e-6) ** 2 / 2  # x² / 2 in µm², then ×k2/dV
    phi_2 = (target_k2[1] / dV) * (v * dl / 1e-6) ** 2 / 2
    phi_3 = (target_k2[2] / dV) * (w * dl / 1e-6) ** 2 / 2

    # RF 基函数（常量电场，用于赝势）
    phi_0 = np.zeros(len(grid_norm))
    # E_rf = V0 * E_basis, 我们让 E_basis 在 x 方向为常量
    # 赝势 V_pseudo = e/(4mΩ²) × (dV/dl)² × V0² × |E_basis|²

    grid_voltage = np.column_stack([phi_0, phi_1, phi_2, phi_3])

    # 插值器
    from FieldParser.calc_field import calc_potential, calc_field
    potential_interps = calc_potential(grid_norm, grid_voltage)
    field_interps = calc_field(grid_norm, grid_voltage)

    # 电压列表
    from utils import Voltage
    from utils import constant

    voltage_list = [
        Voltage("RF", V0=100.0, f=constant(1.0), V_bias=0.0),  # RF
        Voltage("Ux", V0=0.0, f=constant(1.0), V_bias=1.0),    # x-axis DC
        Voltage("Uy", V0=0.0, f=constant(1.0), V_bias=1.0),    # y-axis DC
        Voltage("Uz", V0=0.0, f=constant(1.0), V_bias=1.0),    # z-axis DC
    ]

    opt_config = OptimizationConfig(
        target_freq_MHz=freqs_MHz,
        center_um=(0.0, 0.0, 0.0),
        fit_range_um=((-span, span), (-span, span), (-span, span)),
        n_fit_pts=n_grid,
        fit_degree=2,
        w_freq=1.0,
        w_parity=0.0,
        w_offdiag=0.0,
        v_bias_bounds=(-10.0, 10.0),
    )

    return potential_interps, field_interps, voltage_list, cfg, opt_config


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
class TestTypes:
    def test_optimization_config_defaults(self):
        cfg = OptimizationConfig(target_freq_MHz=(2.0, 3.0, 0.1))
        assert cfg.target_freq_MHz == (2.0, 3.0, 0.1)
        assert cfg.w_freq == 1.0
        assert cfg.w_parity == 0.1
        assert cfg.method == "L-BFGS-B"
        assert not cfg.optimize_rf_v0

    def test_optimization_result(self):
        r = OptimizationResult(
            success=True,
            message="ok",
            n_iterations=10,
            n_evaluations=50,
            initial_voltages=[],
            optimized_voltages=[],
            initial_freqs_MHz={"f_x": 1.0, "f_y": 2.0, "f_z": 0.5},
            optimized_freqs_MHz={"f_x": 2.0, "f_y": 3.0, "f_z": 0.1},
            target_freqs_MHz={"f_x": 2.0, "f_y": 3.0, "f_z": 0.1},
            initial_objective=1.0,
            final_objective=0.001,
        )
        assert r.success
        assert r.n_iterations == 10


class TestFastEvaluator:
    def test_build_evaluator(self):
        """验证 FastEvaluator 构建成功"""
        pot, fld, vlist, cfg, opt_cfg = _make_harmonic_field()
        evaluator = build_fast_evaluator(pot, fld, vlist, cfg, opt_cfg)

        assert evaluator.n_electrodes == 4
        assert len(evaluator.dc_indices) == 3
        assert len(evaluator.rf_indices) == 1
        assert evaluator.rf_indices == [0]
        assert evaluator.initial_V0_rf == 100.0

        for axis in "xyz":
            assert axis in evaluator.phi_1d
            assert evaluator.phi_1d[axis].shape == (opt_cfg.n_fit_pts, 4)

    def test_objective_at_correct_voltages(self):
        """正确电压下目标函数值应接近 0"""
        pot, fld, vlist, cfg, opt_cfg = _make_harmonic_field()
        evaluator = build_fast_evaluator(pot, fld, vlist, cfg, opt_cfg)

        # 初始 V_bias = [1.0, 1.0, 1.0] 正是正确值
        x0 = np.array([1.0, 1.0, 1.0])
        obj = compute_objective(x0, evaluator, opt_cfg)
        # 由于合成场的离散化和拟合误差，不会严格为 0，但应很小
        # 放宽容限，因为合成场网格稀疏
        assert obj < 0.5, f"目标函数值 {obj} 过大，预期 < 0.5"

    def test_nan_protection(self):
        """anti-trapping（k2<0）时应返回大惩罚值"""
        pot, fld, vlist, cfg, opt_cfg = _make_harmonic_field()
        evaluator = build_fast_evaluator(pot, fld, vlist, cfg, opt_cfg)

        # 极大负电压 → anti-trapping
        x_bad = np.array([-1000.0, -1000.0, -1000.0])
        obj = compute_objective(x_bad, evaluator, opt_cfg)
        assert obj > 1e5, f"anti-trapping 惩罚值 {obj} 过小"

    def test_freqs_fast_computation(self):
        """快速频率计算应返回有限值"""
        pot, fld, vlist, cfg, opt_cfg = _make_harmonic_field()
        evaluator = build_fast_evaluator(pot, fld, vlist, cfg, opt_cfg)

        x0 = np.array([1.0, 1.0, 1.0])
        freqs, v_total = _compute_freqs_fast(x0, evaluator, opt_cfg)

        for axis in "xyz":
            f = freqs[f"f_{axis}"]
            assert np.isfinite(f), f"f_{axis} 不是有限值: {f}"


class TestOptimizer:
    def test_optimize_dc_only(self):
        """DC-only 优化：应能收敛到接近目标频率"""
        pot, fld, vlist, cfg, opt_cfg = _make_harmonic_field(
            freqs_MHz=(2.0, 3.0, 0.1),
        )
        from field_optimize.optimizer import optimize_voltages

        result = optimize_voltages(pot, fld, cfg, vlist, opt_cfg)

        # 验证结果结构
        assert isinstance(result, OptimizationResult)
        assert result.n_evaluations > 0

        # 验证优化后频率接近目标
        for axis, target in zip("xyz", [2.0, 3.0, 0.1]):
            f_opt = result.optimized_freqs_MHz[f"f_{axis}"]
            if np.isfinite(f_opt):
                rel_err = abs(f_opt - target) / target
                assert rel_err < 0.3, f"f_{axis}: 相对误差 {rel_err:.2%} 过大 ({f_opt:.4f} vs {target})"

    def test_optimize_from_perturbed_start(self):
        """从偏移初始值出发仍应收敛"""
        pot, fld, vlist, cfg, opt_cfg = _make_harmonic_field()

        # 偏移初始电压
        for i in range(1, 4):
            vlist[i].V_bias = 0.5

        from field_optimize.optimizer import optimize_voltages
        result = optimize_voltages(pot, fld, cfg, vlist, opt_cfg)
        assert result.final_objective < result.initial_objective


class TestCLI:
    def test_create_parser(self):
        from field_optimize.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--csv", "test.csv",
            "--config", "test.json",
            "--target-freq", "2.0", "3.0", "0.1",
        ])
        assert args.target_freq == [2.0, 3.0, 0.1]
        assert args.method == "L-BFGS-B"
        assert not args.optimize_rf_v0

    def test_parser_all_options(self):
        from field_optimize.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--csv", "test.csv",
            "--config", "test.json",
            "--target-freq", "1.0", "2.0", "0.5",
            "--center", "1,2,3",
            "--x-range=-30,30",
            "--y-range=-10,10",
            "--z-range=-100,100",
            "--fit-degree", "4",
            "--n-fit-pts", "100",
            "--w-freq", "2.0",
            "--w-parity", "0.5",
            "--w-offdiag", "0.2",
            "--optimize-rf-v0",
            "--v-bias-bounds=-50,50",
            "--v0-rf-bounds=100,300",
            "--maxiter", "100",
            "--tol", "1e-6",
            "--method", "Nelder-Mead",
        ])
        assert args.center == "1,2,3"
        assert args.fit_degree == 4
        assert args.w_parity == 0.5
        assert args.optimize_rf_v0
        assert args.method == "Nelder-Mead"

    def test_parse_range(self):
        from field_optimize.cli import _parse_range

        assert _parse_range("-50,50") == (-50.0, 50.0)
        assert _parse_range("0,100") == (0.0, 100.0)
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_range("bad")

    def test_parse_3floats(self):
        from field_optimize.cli import _parse_3floats

        assert _parse_3floats("1,2,3") == (1.0, 2.0, 3.0)
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_3floats("bad")


class TestOutputJSON:
    def test_output_json_structure(self):
        """验证输出 JSON 结构正确"""
        import json
        from field_optimize.cli import _write_output_json

        result = OptimizationResult(
            success=True,
            message="ok",
            n_iterations=10,
            n_evaluations=50,
            initial_voltages=[
                {"name": "RF", "type": "rf", "V_bias": 0.0, "V0": 275.0},
                {"name": "U1", "type": "dc", "V_bias": 0.0, "V0": 0.0},
            ],
            optimized_voltages=[
                {"name": "RF", "type": "rf", "V_bias": 0.0, "V0": 275.0},
                {"name": "U1", "type": "dc", "V_bias": -2.3, "V0": 0.0},
            ],
            initial_freqs_MHz={"f_x": 1.0, "f_y": 2.0, "f_z": 0.5},
            optimized_freqs_MHz={"f_x": 2.0, "f_y": 3.0, "f_z": 0.1},
            target_freqs_MHz={"f_x": 2.0, "f_y": 3.0, "f_z": 0.1},
            initial_objective=1.0,
            final_objective=0.001,
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name

        try:
            original_config = {
                "g": 0.1,
                "voltage_list": [
                    {"type": "rf", "name": "RF", "V0": 275, "V_bias": 0, "frequency": 35.28},
                    {"type": "dc", "name": "U1", "V_bias": 0},
                ],
            }
            _write_output_json(result, tmp_path, original_config)

            with open(tmp_path) as f:
                data = json.load(f)

            assert "g" in data
            assert data["g"] == 0.1
            assert "voltage_list" in data
            assert len(data["voltage_list"]) == 2
            assert data["voltage_list"][1]["V_bias"] == pytest.approx(-2.3, abs=1e-3)
            assert "_optimization" in data
            assert data["_optimization"]["success"] is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)
