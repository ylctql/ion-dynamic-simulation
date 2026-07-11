"""poly-potential 可视化测试：load_poly_field_bundle 接口契约与 compute_potentials 回收。

纯 numpy（不需 ionsim），conda base 可跑。核心不变量：
- poly 伪装为单电极 DC、V_bias=1 → V_dc=V_poly, V_pseudo=0, V_total=V_poly（dc_only 分支）
- dl/dV 在 µm↔归一化往返中抵消（dl 无关性）
"""
import matplotlib

matplotlib.use("Agg")  # 无显示环境；须在 pyplot 首次 import 前

import math

import numpy as np

from FieldConfiguration.constants import BA135_MASS_AMU
from equilibrium.potential_fit_3d import (
    eval_fit_3d,
    make_ideal_trap_fit,
    write_potential_fit_coeff_json,
)
from field_visualize.core import (
    FieldBundle,
    compute_potentials,
    load_poly_field_bundle,
    um_to_norm,
)

EXAMPLE = "FieldConfiguration/configs/poly_potential/example.json"


def _ideal_bundle(tmp_path, freqs=(2.0, 2.5, 0.5), scale_um=100.0):
    """用 make_ideal_trap_fit 构造已知二次势，导出 JSON 再 load_poly_field_bundle。"""
    fit = make_ideal_trap_fit(freqs, mass_amu=BA135_MASS_AMU, scale_um=scale_um)
    p = tmp_path / "ideal_poly.json"
    write_potential_fit_coeff_json(fit, p)
    bundle = load_poly_field_bundle(str(p))
    return fit, bundle


def _V_total_at(bundle, r_um):
    """在物理坐标 r_um (µm) 处算 V_total（自动按 bundle.cfg.dl 归一化）。"""
    r_norm = np.asarray(r_um) * 1e-6 / bundle.cfg.dl
    _, _, _, v_total = compute_potentials(
        bundle.potential_interps,
        bundle.field_interps,
        bundle.voltage_list,
        bundle.cfg,
        r_norm,
    )
    return v_total


# A. 接口契约
def test_poly_bundle_structure():
    bundle = load_poly_field_bundle(EXAMPLE)
    assert isinstance(bundle, FieldBundle)
    assert len(bundle.potential_interps) == 1
    assert len(bundle.field_interps) == 1
    assert len(bundle.voltage_list) == 1
    v = bundle.voltage_list[0]
    assert v.V0 == 0.0           # DC only，非 RF
    assert v.V_bias == 1.0       # 单位缩放
    assert bundle.grid_coord is None  # poly 无网格


# B. compute_potentials 回收 V_poly
def test_compute_potentials_returns_poly(tmp_path):
    fit, bundle = _ideal_bundle(tmp_path)
    r_um = np.array([[10.0, 5.0, 2.0], [0.0, 0.0, 0.0], [-10.0, -5.0, -2.0]])
    r_norm = r_um * 1e-6 / bundle.cfg.dl
    V_dc, V_rf, V_pseudo, V_total = compute_potentials(
        bundle.potential_interps, bundle.field_interps,
        bundle.voltage_list, bundle.cfg, r_norm,
    )
    V_expected = eval_fit_3d(fit, r_um)
    np.testing.assert_allclose(V_dc, V_expected, rtol=1e-10)
    np.testing.assert_allclose(V_rf, 0.0, atol=1e-30)
    np.testing.assert_allclose(V_pseudo, 0.0, atol=1e-30)
    np.testing.assert_allclose(V_total, V_expected, rtol=1e-10)


# C. poly 走 total 分支（总势标注，非 dc_only）
def test_poly_decomp_total_mode(tmp_path):
    from field_visualize.plots import _classify_dc_rf_coverage, _decomp_mode_and_note

    _, bundle = _ideal_bundle(tmp_path)
    r_um = np.array([[10.0, 5.0, 2.0], [-10.0, -5.0, -2.0]])
    r_norm = r_um * 1e-6 / bundle.cfg.dl
    V_dc, _, V_pseudo, _ = compute_potentials(
        bundle.potential_interps, bundle.field_interps,
        bundle.voltage_list, bundle.cfg, r_norm,
    )
    # 底层分类器按 V_dc/V_pseudo 判定为 dc_only（V_pseudo=0）
    assert _classify_dc_rf_coverage(V_dc, V_pseudo) == "dc_only"
    # poly 可视化传 source_label → 强制 total 分支（both_zero），note 用 source_label
    mode, note = _decomp_mode_and_note(V_dc, V_pseudo, "Polynomial total potential")
    assert mode == "both_zero"
    assert note == "Polynomial total potential"


# D. dl 无关性（同一物理坐标 r_um 在不同 cfg.dl 下 V_total 一致）
def test_dl_independence(tmp_path):
    import json

    _, b_default = _ideal_bundle(tmp_path)
    # 写一个不同 RF 频率的 config，使 cfg.dl 与合成默认（35.28 MHz）不同
    cfg_other = tmp_path / "other_config.json"
    cfg_other.write_text(
        json.dumps({"voltage_list": [
            {"type": "rf", "name": "RF", "V0": 275, "V_bias": -8, "frequency": 10.0}
        ]})
    )
    b_cfg = load_poly_field_bundle(
        str(tmp_path / "ideal_poly.json"), config_path=str(cfg_other),
    )
    assert b_default.cfg.dl != b_cfg.cfg.dl  # 确实用了不同 dl
    r_um = np.array([[10.0, 5.0, 2.0], [-8.0, 4.0, -1.0]])
    v_default = _V_total_at(b_default, r_um)
    v_cfg = _V_total_at(b_cfg, r_um)
    np.testing.assert_allclose(v_default, v_cfg, rtol=1e-12)


# E. 1D/2D 冒烟
def test_plot_smoke(tmp_path):
    import matplotlib.pyplot as plt
    from field_visualize.plots import plot_1d, plot_2d

    _, bundle = _ideal_bundle(tmp_path)
    dl = bundle.cfg.dl
    xr = (um_to_norm(-20, dl), um_to_norm(20, dl))
    yr = (um_to_norm(-20, dl), um_to_norm(20, dl))
    plot_1d(
        bundle.potential_interps, bundle.field_interps, bundle.voltage_list, bundle.cfg,
        vary_axis="x", vary_range=xr, n_pts=50, source_label="Total potential",
    )
    plt.close("all")
    plot_2d(
        bundle.potential_interps, bundle.field_interps, bundle.voltage_list, bundle.cfg,
        vary_axes=("x", "y"), x_range=xr, y_range=yr, n_pts=(30, 30),
        source_label="Total potential",
    )
    plt.close("all")


# F. 阱频回路（make_ideal_trap_fit 已知频率 → compute_trap_freqs_at_point 回收）
def test_trap_freq_roundtrip(tmp_path):
    from field_visualize.trap_freq import compute_trap_freqs_at_point

    _, bundle = _ideal_bundle(tmp_path, freqs=(2.0, 2.5, 0.5))
    freqs = compute_trap_freqs_at_point(
        bundle.potential_interps, bundle.field_interps, bundle.voltage_list, bundle.cfg,
        0.0, 0.0, 0.0,
        x_range_um=(-20.0, 20.0), y_range_um=(-20.0, 20.0), z_range_um=(-20.0, 20.0),
        n_pts=200, fit_degree=2,
    )
    assert math.isclose(freqs["f_x"], 2.0, rel_tol=1e-3)
    assert math.isclose(freqs["f_y"], 2.5, rel_tol=1e-3)
    assert math.isclose(freqs["f_z"], 0.5, rel_tol=1e-3)
