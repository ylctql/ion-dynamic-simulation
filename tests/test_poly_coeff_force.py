"""显式多项式系数势（order-free，term-label 显式指定）的解析、装载与动力学力测试。"""
from types import SimpleNamespace

import numpy as np
import pytest

from equilibrium.potential_fit_3d import (
    eval_fit_3d,
    fit_result_from_coeff_map,
    grad_fit_3d,
    load_poly_potential_json,
    make_ideal_trap_fit,
    parse_term_label,
    quartic_3d_term_label,
    write_potential_fit_coeff_json,
)
from FieldParser.force import build_poly_potential_force


# ---------------------------------------------------------------------------
# parse_term_label
# ---------------------------------------------------------------------------

def test_parse_term_label_roundtrip():
    """quartic_3d_term_label 的反函数对所有 (i,j,k) ≤4 往返一致。"""
    for i in range(5):
        for j in range(5):
            for k in range(5):
                label = quartic_3d_term_label(i, j, k)
                assert parse_term_label(label) == (i, j, k), f"{label} -> ?"


def test_parse_term_label_cases():
    assert parse_term_label("1") == (0, 0, 0)
    assert parse_term_label("x") == (1, 0, 0)
    assert parse_term_label("z^4") == (0, 0, 4)
    assert parse_term_label("x*y*z") == (1, 1, 1)
    assert parse_term_label("x^4*y^3") == (4, 3, 0)
    # 变量顺序无关
    assert parse_term_label("y*x") == (1, 1, 0)
    assert parse_term_label("z^2*x") == (1, 0, 2)


def test_parse_term_label_rejects_invalid():
    with pytest.raises(ValueError, match="次数.*越界"):
        parse_term_label("x^5")          # 每变量次数上限 4
    with pytest.raises(ValueError, match="无法解析"):
        parse_term_label("w")            # 非法变量
    with pytest.raises(ValueError, match="无法解析"):
        parse_term_label("x^2.5")        # 非整数次数
    with pytest.raises(ValueError, match="重复"):
        parse_term_label("x*x")          # 同变量重复
    with pytest.raises(ValueError, match="重复"):
        parse_term_label("x*x^2")        # 同变量重复（不同次）


# ---------------------------------------------------------------------------
# fit_result_from_coeff_map / eval_fit_3d
# ---------------------------------------------------------------------------

def test_fit_result_from_coeff_map_eval():
    """V(u) = 2 + 3*u_x^2，在 u=0.5 (x=50µm, L=100µm) 处 V = 2.75。"""
    fit = fit_result_from_coeff_map(
        {"1": 2.0, "x^2": 3.0},
        center_um=(0.0, 0.0, 0.0),
        scale_um=100.0,
    )
    V = eval_fit_3d(fit, np.array([[50.0, 0.0, 0.0]]))  # x=50µm → u=0.5
    np.testing.assert_allclose(V, [2.75], atol=1e-12)


def test_fit_result_from_coeff_map_rejects_duplicate():
    """同一单项式以不同标签形式重复指定应报错（x*y 与 y*x 同为 (1,1,0)）。"""
    with pytest.raises(ValueError, match="重复"):
        fit_result_from_coeff_map({"x*y": 1.0, "y*x": 2.0})


# ---------------------------------------------------------------------------
# JSON 往返：导出 -> 读回，梯度一致
# ---------------------------------------------------------------------------

def test_export_load_grad_roundtrip(tmp_path):
    """make_ideal_trap_fit → 导出 JSON（含 center/scale）→ 读回 → 梯度一致。"""
    fit = make_ideal_trap_fit(
        freq_MHz=(1.0, 1.2, 0.8),
        mass_amu=135.0,
        charge_ec=1.0,
        center_um=(0.0, 0.0, 0.0),
        scale_um=50.0,
    )
    out = tmp_path / "coeff.json"
    write_potential_fit_coeff_json(fit, out, csv="x.csv", config="default.json")

    loaded = load_poly_potential_json(out)
    # center/scale 经 JSON 往返保持
    assert loaded.scale_um == pytest.approx(50.0)
    # 梯度在多点上一致
    pts = np.array(
        [[0.0, 0.0, 0.0], [10.0, -5.0, 3.0], [-20.0, 8.0, -1.0], [5.0, 5.0, 5.0]]
    )
    np.testing.assert_allclose(grad_fit_3d(fit, pts), grad_fit_3d(loaded, pts), atol=1e-12)


def test_load_poly_potential_json_missing_coefficients_raises(tmp_path):
    out = tmp_path / "bad.json"
    out.write_text('{"center_um": [0,0,0]}', encoding="utf-8")
    with pytest.raises(ValueError, match="缺少 'coefficients'"):
        load_poly_potential_json(out)


def test_load_poly_potential_json_warns_missing_scale(tmp_path):
    """缺 center_um/scale_um 的文件回退默认值并告警（防止旧版导出文件误用）。"""
    import json

    out = tmp_path / "no_scale.json"
    out.write_text(json.dumps({"coefficients": {"x^2": 1.0}}), encoding="utf-8")
    with pytest.warns(UserWarning, match="scale_um"):
        fit = load_poly_potential_json(out)
    # 回退到默认
    assert fit.scale_um == 100.0
    assert fit.center_um == (0.0, 0.0, 0.0)


def test_load_poly_potential_json_no_warn_when_complete(tmp_path):
    """显式给出 center_um/scale_um 时不告警。"""
    import json

    out = tmp_path / "complete.json"
    out.write_text(
        json.dumps(
            {"coefficients": {"x^2": 1.0}, "center_um": [0, 0, 0], "scale_um": 150.0}
        ),
        encoding="utf-8",
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 任何 UserWarning 都转错误
        fit = load_poly_potential_json(out)
    assert fit.scale_um == 150.0


# ---------------------------------------------------------------------------
# 动力学力：build_poly_potential_force 二次势解析对照
# ---------------------------------------------------------------------------

def _cfg(dl=1e-6, dV=1.0):
    """最小 Config 替身（build_poly_potential_force 仅用 dl/dV）。"""
    return SimpleNamespace(dl=dl, dV=dV)


def test_build_poly_potential_force_quadratic():
    """
    V = u^2+v^2+w^2 (V)，center=0, L=1µm, dl=1µm, dV=1V, charge=1。
    r_norm=0.5 → r_um=0.5µm → dV/dx=2*0.5=1 V/µm → E_norm=-1 → F_x=-1。
    """
    fit = fit_result_from_coeff_map(
        {"x^2": 1.0, "y^2": 1.0, "z^2": 1.0},
        center_um=(0.0, 0.0, 0.0),
        scale_um=1.0,
    )
    force = build_poly_potential_force(fit, _cfg(), charge=np.array([1.0]), gamma=0.0)

    r = np.array([[0.5, 0.0, 0.0]])
    v = np.zeros_like(r)
    F = force(r, v, 0.0)
    np.testing.assert_allclose(F, [[-1.0, 0.0, 0.0]], atol=1e-12)


def test_build_poly_potential_force_multi_ion_and_gamma():
    """多离子 + 阻尼：F = -2*r_um - gamma*v（同上 V，gamma=1）。"""
    fit = fit_result_from_coeff_map(
        {"x^2": 1.0, "y^2": 1.0, "z^2": 1.0},
        center_um=(0.0, 0.0, 0.0),
        scale_um=1.0,
    )
    force = build_poly_potential_force(
        fit, _cfg(), charge=np.array([1.0, 1.0, 1.0]), gamma=1.0
    )
    r = np.array([[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, -0.5]])
    v = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    F = force(r, v, 0.0)
    expected = -2.0 * r - 1.0 * v   # r 此处数值等于 r_um（dl_um=1）
    np.testing.assert_allclose(F, expected, atol=1e-12)


def test_build_poly_potential_force_charge_scaling():
    """
    charge 缩放：charge=2 时 F_x = 2*(-1) = -2（单位电荷下 x^2 势在 r=0.5 处 F_x=-1，
    见 quadratic 测试）。注：build_poly_potential_force 用模块级状态（同 make_force），
    故单进程内不复用多个实例，这里直接对 charge=2 断言。
    """
    fit = fit_result_from_coeff_map(
        {"x^2": 1.0}, center_um=(0.0, 0.0, 0.0), scale_um=1.0
    )
    r = np.array([[0.5, 0.0, 0.0]])
    v = np.zeros_like(r)
    force = build_poly_potential_force(fit, _cfg(), charge=np.array([2.0]), gamma=0.0)
    np.testing.assert_allclose(force(r, v, 0.0), [[-2.0, 0.0, 0.0]], atol=1e-12)
