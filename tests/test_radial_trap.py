"""radial_trap 模块测试。

基线参数：A=5e-3 V/µm², B=D=F=0, E=1.5e-4 V/µm², f_RF=35.28 MHz, Ba135+
（对应 c_x2≈5.14e-4, c_y2≈2.14e-4 V/µm²，q≈0.291，f_x≈4.31 MHz，f_y≈2.78 MHz）。
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.constants import e as ELEMENTARY_CHARGE
from scipy.constants import epsilon_0, pi

from equilibrium.potential_fit_3d import eval_fit_3d, hessian_fit_3d
from FieldConfiguration.ion_species import ION_SPECIES

from radial_trap.cli import create_parser, main
from radial_trap.lattice import (
    find_radial_equilibrium,
    micromotion_amplitude,
    solve_inplane_phonons,
)
from radial_trap.potential import (
    alpha_eff_um2_per_V,
    build_total_potential_fit,
    evaluate_conditions,
    rf_field_V_per_um,
    rf_potential_V,
    total_potential_coefficients,
    total_potential_V,
)
from radial_trap.types import RadialTrapParams

COULOMB_K = 1.0 / (4.0 * pi * epsilon_0)


def make_params(**kwargs) -> RadialTrapParams:
    base = dict(
        A_V_per_um2=5e-3,
        B_V_per_um4=0.0,
        D_dimless=0.0,
        E_dc_V_per_um2=1.5e-4,
        F_dc_V_per_um4=0.0,
    )
    base.update(kwargs)
    return RadialTrapParams(**base)


# ---------------------------------------------------------------------------
# T1: RF 场 = +∇φ_rf
# ---------------------------------------------------------------------------


def test_rf_field_matches_hand_formula():
    x = np.array([-3.0, 0.0, 2.5])
    y = np.array([1.0, -4.0, 0.5])
    A, B = 5e-3, 1e-7
    got = rf_field_V_per_um(x, y, A, B)
    ex = 2 * A * x + 4 * B * x**3 - 12 * B * x * y**2
    ey = -2 * A * y - 12 * B * x**2 * y + 4 * B * y**3
    np.testing.assert_allclose(got[:, 0], ex, rtol=1e-12)
    np.testing.assert_allclose(got[:, 1], ey, rtol=1e-12)


def test_rf_field_matches_numeric_gradient():
    # 网格数值梯度交叉验证（含 B≠0 的四次项）。
    # np.gradient 边界为单侧差分（截断误差大），只比较内部点。
    A, B = 5e-3, 1e-6
    n = 201
    x = np.linspace(-5.0, 5.0, n)
    y = np.linspace(-5.0, 5.0, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    v = rf_potential_V(xx, yy, A, B)
    dv_dx, dv_dy = np.gradient(v, x, y)
    got = rf_field_V_per_um(xx, yy, A, B)
    np.testing.assert_allclose(got[1:-1, 1:-1, 0], dv_dx[1:-1, 1:-1], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got[1:-1, 1:-1, 1], dv_dy[1:-1, 1:-1], rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# T2: 系数表/张量构造 —— FitResult3D 求值 == 独立解析求值
# ---------------------------------------------------------------------------


def test_alpha_eff_units_match_si_energy_form():
    # 独立单位链校验：φ_pp[V] = (Ψ_J/Q) = Q·E_SI²/(4mΩ²)，E_SI = E[V/µm]·1e6
    sp = ION_SPECIES["Ba135+"]
    alpha = alpha_eff_um2_per_V(35.28, sp)
    omega = 2 * np.pi * 35.28e6
    x_um, y_um = 3.0, 5.0
    A = 5e-3
    ex_um = 2 * A * x_um  # V/µm
    ey_um = -2 * A * y_um
    e_sq_si = (ex_um**2 + ey_um**2) * 1e12  # (V/m)²
    pp_volts_si = sp.charge_C * e_sq_si / (4 * sp.mass_kg * omega**2)
    pp_volts_alpha = alpha * (ex_um**2 + ey_um**2)
    assert pp_volts_alpha == pytest.approx(pp_volts_si, rel=1e-12)


def test_fit_eval_matches_direct_potential():
    # 全系数非零（演示参数 B=1e-7, D=0.5, F=-5e-8），随机点交叉验证
    params = make_params(B_V_per_um4=1e-7, D_dimless=0.5, F_dc_V_per_um4=-5e-8)
    fit = build_total_potential_fit(params)
    rng = np.random.default_rng(7)
    r = np.column_stack(
        [rng.uniform(-30, 30, 50), rng.uniform(-80, 80, 50), np.zeros(50)]
    )
    v_fit = eval_fit_3d(fit, r)
    v_direct = total_potential_V(r[:, 0], r[:, 1], params)
    np.testing.assert_allclose(v_fit, v_direct, rtol=1e-10)


def test_flat_potential_raises():
    with pytest.raises(ValueError, match="平坦势"):
        build_total_potential_fit(make_params(A_V_per_um2=0.0, E_dc_V_per_um2=0.0))


def test_scale_um_invariance():
    # 归一化正确性：scale 只改变存储系数 c=a·L^n，不改变物理求值
    params100 = make_params(B_V_per_um4=1e-7, D_dimless=0.5, F_dc_V_per_um4=-5e-8)
    params200 = make_params(
        B_V_per_um4=1e-7, D_dimless=0.5, F_dc_V_per_um4=-5e-8, scale_um=200.0
    )
    fit100 = build_total_potential_fit(params100)
    fit200 = build_total_potential_fit(params200)
    rng = np.random.default_rng(11)
    r = np.column_stack(
        [rng.uniform(-30, 30, 20), rng.uniform(-80, 80, 20), np.zeros(20)]
    )
    np.testing.assert_allclose(
        eval_fit_3d(fit100, r), eval_fit_3d(fit200, r), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# T3: 原点 Hessian = diag(2c_x2, 2c_y2, 0)
# ---------------------------------------------------------------------------


def test_hessian_at_origin():
    params = make_params(B_V_per_um4=1e-7, D_dimless=0.5, F_dc_V_per_um4=-5e-8)
    fit = build_total_potential_fit(params)
    c = total_potential_coefficients(params)
    h = hessian_fit_3d(fit, np.zeros((1, 3)))[0]
    expected = np.diag([2 * c[(2, 0)], 2 * c[(0, 2)], 0.0])
    np.testing.assert_allclose(h, expected, rtol=1e-9, atol=1e-15)


# ---------------------------------------------------------------------------
# T4/T5: 晶格条件
# ---------------------------------------------------------------------------


def test_conditions_quadrupole_only_all_pass():
    conds = evaluate_conditions(make_params())
    assert conds.pass_cross
    assert conds.pass_y_high
    assert conds.pass_y_confine
    assert conds.pass_x_confine
    assert conds.c_y2 < conds.c_x2
    assert conds.f_trap_x_mhz > conds.f_trap_y_mhz > 0.0
    assert conds.q_mathieu_x == pytest.approx(0.2907, rel=2e-3)
    assert conds.q_mathieu_y == pytest.approx(-conds.q_mathieu_x, rel=1e-12)
    assert not conds.xy_degenerate
    # 手算复核默认参数阱频
    assert conds.f_trap_x_mhz == pytest.approx(4.31, rel=2e-3)
    assert conds.f_trap_y_mhz == pytest.approx(2.78, rel=2e-3)


def test_conditions_y_unconfined_when_dc_too_large():
    # E_dc > 4αA² + DA → c_y2 < 0：条件(3)失败，条件(4)仍过，f_y 为 NaN
    params = make_params(E_dc_V_per_um2=5e-4)
    conds = evaluate_conditions(params)
    assert not conds.pass_y_confine
    assert conds.pass_x_confine
    assert conds.pass_cross and conds.pass_y_high  # B=F=0 时高次条件仍满足
    assert np.isnan(conds.f_trap_y_mhz)
    assert conds.eps4_y == np.inf
    # c_y2 精确复核：4αA² − E
    sp = ION_SPECIES["Ba135+"]
    alpha = alpha_eff_um2_per_V(params.freq_rf_mhz, sp)
    assert conds.c_y2 == pytest.approx(4 * alpha * params.A_V_per_um2**2 - 5e-4)


def test_conditions_cross_term_tolerance():
    # DB+F = 0 精确抵消（演示参数）：B≠0 时条件(1)不通过（要求 B→0）
    params = make_params(B_V_per_um4=1e-7, D_dimless=0.5, F_dc_V_per_um4=-5e-8)
    conds = evaluate_conditions(params)
    assert conds.db_plus_f == pytest.approx(0.0, abs=1e-20)
    assert not conds.pass_cross  # B≠0
    assert conds.y4_comb == pytest.approx(conds.s_y4)  # y4_comb = −16αAB+DB+F = s_y4


# ---------------------------------------------------------------------------
# T7–T12: 平衡构型与解析 micromotion（lattice.py）
# ---------------------------------------------------------------------------


def _alpha() -> float:
    return alpha_eff_um2_per_V(35.28, ION_SPECIES["Ba135+"])


def test_single_ion_at_origin():
    # T7: N=1 → r_eq ≈ 原点，a_mm ≈ 0，excess 比全 NaN（r≈0 无定义）
    params = make_params(n_ions=1)
    fit = build_total_potential_fit(params)
    eq = find_radial_equilibrium(fit, params)
    assert eq.converged
    assert not eq.hit_bounds
    np.testing.assert_allclose(np.abs(eq.r_eq_um), 0.0, atol=1e-6)
    mm = micromotion_amplitude(eq.r_eq_um, params)
    np.testing.assert_allclose(mm.mag_um, 0.0, atol=1e-12)
    assert np.all(np.isnan(mm.excess_ratio))


def test_two_ion_chain_spacing_analytic():
    # T8: N=2 链沿 y（弱轴），间距 d = 2·(k_e·Q·1e6/(8·c_y2))^{1/3}
    params = make_params(n_ions=2)
    fit = build_total_potential_fit(params)
    eq = find_radial_equilibrium(fit, params)
    assert eq.converged
    assert not eq.hit_bounds

    sp = ION_SPECIES["Ba135+"]
    c_y2 = 4.0 * _alpha() * params.A_V_per_um2**2 - params.E_dc_V_per_um2
    assert c_y2 == pytest.approx(2.14e-4, rel=5e-3)
    y0 = (COULOMB_K * sp.charge_C * 1e6 / (8.0 * c_y2)) ** (1.0 / 3.0)
    np.testing.assert_allclose(np.sort(np.abs(eq.r_eq_um[:, 1])), [y0, y0], rtol=1e-4)
    np.testing.assert_allclose(eq.r_eq_um[:, 0], 0.0, atol=1e-6)


def test_two_ion_chain_flips_to_x_axis():
    # T9: E 变号（镜像参数）→ c_x2 ↔ c_y2 互换，链翻转到 x 轴。
    # （注意 E < −4αA² 会使 c_x2<0 即 x 无束缚而非翻转，翻转条件是 E<0。）
    params = make_params(n_ions=2, E_dc_V_per_um2=-1.5e-4)
    fit = build_total_potential_fit(params)
    eq = find_radial_equilibrium(fit, params)
    assert eq.converged
    assert not eq.hit_bounds

    sp = ION_SPECIES["Ba135+"]
    c_x2 = 4.0 * _alpha() * params.A_V_per_um2**2 + params.E_dc_V_per_um2
    x0 = (COULOMB_K * sp.charge_C * 1e6 / (8.0 * c_x2)) ** (1.0 / 3.0)
    np.testing.assert_allclose(np.sort(np.abs(eq.r_eq_um[:, 0])), [x0, x0], rtol=1e-4)
    np.testing.assert_allclose(eq.r_eq_um[:, 1], 0.0, atol=1e-6)


def test_two_ion_micromotion_identity():
    # T10: 四极极限恒等式 a_mm,y = −(q/2)·y_i（q 由测试内 SI 常数重算）
    params = make_params(n_ions=2)
    fit = build_total_potential_fit(params)
    eq = find_radial_equilibrium(fit, params)
    mm = micromotion_amplitude(eq.r_eq_um, params)

    sp = ION_SPECIES["Ba135+"]
    omega = 2.0 * np.pi * 35.28e6
    q = 4.0 * ELEMENTARY_CHARGE * params.A_V_per_um2 * 1e12 / (sp.mass_kg * omega**2)
    np.testing.assert_allclose(mm.a_mm_um[:, 1], -0.5 * q * eq.r_eq_um[:, 1], rtol=1e-9)
    # x 轴同恒等式（链上 r_x 收敛残差 ~1e-9 µm ≠ 0，断言恒等式而非绝对零）
    np.testing.assert_allclose(mm.a_mm_um[:, 0], 0.5 * q * eq.r_eq_um[:, 0], rtol=1e-9)
    assert mm.a_mm_um.shape == (2, 3) and np.all(mm.a_mm_um[:, 2] == 0.0)
    # 模长 == 矢量范数
    np.testing.assert_allclose(mm.mag_um, np.linalg.norm(mm.a_mm_um, axis=1), rtol=1e-12)


def test_excess_ratio_equals_field_ratio_when_b_nonzero():
    # T11: B≠0 时 excess 比 ≡ 沿轴 |E_full|/|E_quad|（带符号，逐点代数恒等式）
    params = make_params(n_ions=2, B_V_per_um4=1e-7)
    fit = build_total_potential_fit(params)
    eq = find_radial_equilibrium(fit, params)
    mm = micromotion_amplitude(eq.r_eq_um, params)

    x, y = eq.r_eq_um[:, 0], eq.r_eq_um[:, 1]
    e_full = rf_field_V_per_um(x, y, params.A_V_per_um2, params.B_V_per_um4)
    e_quad = rf_field_V_per_um(x, y, params.A_V_per_um2, 0.0)
    ratio_y = e_full[:, 1] / e_quad[:, 1]
    np.testing.assert_allclose(mm.excess_ratio[:, 1], ratio_y, rtol=1e-9)
    # B 很小 → ρ_y 接近 1 但系统性偏离
    assert np.all(np.abs(mm.excess_ratio[:, 1] - 1.0) > 0.0)


def test_excess_ratio_one_in_quadrupole_limit():
    # T12: B=0 时所有有限 ρ==1；r_a≈0（N=1 原点）处为 NaN
    params = make_params(n_ions=2)
    fit = build_total_potential_fit(params)
    eq = find_radial_equilibrium(fit, params)
    mm = micromotion_amplitude(eq.r_eq_um, params)

    rho = mm.excess_ratio
    finite = np.isfinite(rho)
    assert np.any(finite)  # 链沿 y：y 轴分量必有限
    assert np.allclose(rho[finite], 1.0, rtol=1e-9)
    assert np.all(np.isnan(rho[~finite]))
    # |r_a| >= 1e-9 的分量必须有限（无 0/0 以外的未定义路径）
    big = np.abs(eq.r_eq_um) >= 1e-9
    xy_mask = big.copy()
    xy_mask[:, 2] = False  # z 轴 q=0 恒无定义
    assert np.all(np.isfinite(rho[xy_mask]))


# ---------------------------------------------------------------------------
# T13: 面内声子（N=2 解析谱）
# ---------------------------------------------------------------------------


def test_two_ion_inplane_phonons_analytic():
    # 4 个面内模：COM-x(f_x)、COM-y(f_y)、轴 stretch(√3·f_y)、横向 stretch(√(f_x²−f_y²))
    params = make_params(n_ions=2)
    fit = build_total_potential_fit(params)
    eq = find_radial_equilibrium(fit, params)
    ph = solve_inplane_phonons(fit, eq.r_eq_um, params)

    assert ph.freq_hz_signed.size == 4
    assert np.all(ph.freq_hz_signed > 0)
    assert ph.dof_indices.tolist() == [0, 1, 3, 4]
    c = evaluate_conditions(params)
    fy, fx = c.f_trap_y_mhz, c.f_trap_x_mhz
    expected = np.sort(
        np.array([fx, fy, np.sqrt(3.0) * fy, np.sqrt(fx * fx - fy * fy)])
    )[::-1]
    got = np.sort(ph.freq_hz_signed / 1e6)[::-1]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# T14/T15/T18/T19: CLI
# ---------------------------------------------------------------------------


def test_cli_parser_defaults():
    args = create_parser().parse_args([])
    # 可覆盖参数默认 None（表示"未显式给出"，合并时用 --json 或内置默认）
    assert args.A is None and args.E is None and args.N is None
    assert args.maxiter == 5000 and args.tol == 1e-15
    assert args.cond_tol == 1e-12
    assert not args.phonon


def test_cli_range_parser():
    # 负数范围必须用 = 语法（argparse 会把 "-5,5" 当旗标）
    args = create_parser().parse_args(["--x-range=-5,5", "--y-range=-40, 40"])
    assert args.x_range == (-5.0, 5.0)
    assert args.y_range == (-40.0, 40.0)
    with pytest.raises(SystemExit):
        create_parser().parse_args(["--x-range=5,-5"])  # lo >= hi


def test_cli_bad_species_exits(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--species", "Xe100+"])
    assert ei.value.code == 2
    assert "未知物种" in capsys.readouterr().err


def test_cli_json_unknown_key_exits(tmp_path, capsys):
    jp = tmp_path / "p.json"
    jp.write_text(json.dumps({"A": 1e-3, "bogus": 1}), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        main(["--json", str(jp)])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "bogus" in err and "合法键" in err


def test_cli_json_roundtrip_and_cli_override(tmp_path):
    # T15: --json 读入 + CLI 旗标覆盖 JSON 值 + 默认参数 npz 落盘
    jp = tmp_path / "p.json"
    jp.write_text(
        json.dumps(
            {"A": 4e-3, "B": 5e-7, "E": 1e-4, "N": 3,
             "x_range": [-5, 5], "y_range": [-40, 40]}
        ),
        encoding="utf-8",
    )
    out = tmp_path / "r.npz"
    rep = tmp_path / "r.json"
    rc = main(["--json", str(jp), "--B", "2e-7", "--out", str(out), "--report", str(rep)])
    assert rc == 0
    d = np.load(out)
    assert float(d["params_A_V_per_um2"]) == pytest.approx(4e-3)  # 来自 json
    assert float(d["params_E_dc_V_per_um2"]) == pytest.approx(1e-4)
    assert int(d["params_n_ions"]) == 3
    assert tuple(d["params_x_range_um"]) == (-5.0, 5.0)
    assert float(d["params_B_V_per_um4"]) == pytest.approx(2e-7)  # CLI 覆盖 json
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["params"]["n_ions"] == 3
    assert report["params"]["A_V_per_um2"] == pytest.approx(4e-3)
    assert report["equilibrium"]["converged"] is True
    assert report["conditions"]["pass_cross"] is False  # B≠0 → 条件(1) FAIL
    assert "phonon" not in report


def test_cli_npz_payload_default(tmp_path):
    # T18: 默认参数 npz 键/形状/有限性
    out = tmp_path / "d.npz"
    rc = main(["--N", "2", "--out", str(out)])
    assert rc == 0
    d = np.load(out)
    assert d["r_eq_um"].shape == (2, 3)
    assert d["r_init_um"].shape == (2, 3)
    assert d["a_mm_um"].shape == (2, 3)
    assert d["a_mm_mag_um"].shape == (2,)
    assert d["a_mm_excess_ratio"].shape == (2, 3)
    assert np.all(np.isfinite(d["r_eq_um"]))
    assert np.all(np.isfinite(d["a_mm_mag_um"]))
    assert bool(d["converged"])
    assert not bool(d["hit_bounds"])
    for key in (
        "cond_c_x2", "cond_c_y2", "cond_f_trap_x_mhz", "cond_f_trap_y_mhz",
        "cond_q_mathieu_x", "cond_pass_cross", "cond_eps4_x",
        "params_freq_rf_mhz", "params_species_name",
    ):
        assert key in d, key
    assert "phonon_freqs_hz" not in d  # 无 --phonon
    assert str(d["params_species_name"]) == "Ba135+"


def test_cli_phonon_flag_writes_modes(tmp_path):
    out = tmp_path / "p.npz"
    rep = tmp_path / "p.json"
    rc = main(["--N", "2", "--phonon", "--out", str(out), "--report", str(rep)])
    assert rc == 0
    d = np.load(out)
    assert d["phonon_freqs_hz"].shape == (4,)
    assert np.all(d["phonon_freqs_hz"] > 0)
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert len(report["phonon"]["freq_mhz"]) == 4


def test_cli_unstable_config_completes(tmp_path, caplog):
    # T19: T5 参数（c_y2<0，y 无二次囚禁）：main() 完成不崩，报告标志失败
    out = tmp_path / "u.npz"
    rep = tmp_path / "u.json"
    rc = main(
        ["--E", "5e-4", "--N", "2", "--y-range=-40,40",
         "--out", str(out), "--report", str(rep)]
    )
    assert rc == 0
    d = np.load(out)
    assert not bool(d["cond_pass_y_confine"])
    assert bool(d["hit_bounds"])  # 离子逃到 y 边界
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["conditions"]["pass_y_confine"] is False
    assert report["conditions"]["f_trap_y_mhz"] is None  # NaN → null
    assert report["equilibrium"]["hit_bounds"] is True
    assert any("条件(3)" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# T17: 绘图冒烟
# ---------------------------------------------------------------------------


def test_cli_plot_smoke(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    png = tmp_path / "lat.png"
    rc = main(
        ["--N", "2", "--plot", "--plot-out", str(png), "--out", str(tmp_path / "x.npz")]
    )
    assert rc == 0
    assert png.exists() and png.stat().st_size > 0


# ---------------------------------------------------------------------------
# T16: --export-poly 往返（依赖次数上限 4→6 扩展）
# ---------------------------------------------------------------------------


def test_cli_export_poly_roundtrip(tmp_path):
    # --export-poly → load_poly_potential_json → eval_fit_3d == 原 fit（含 x^6 项）
    from equilibrium.potential_fit_3d import load_poly_potential_json

    pj = tmp_path / "total_poly.json"
    rc = main(
        ["--B", "1e-7", "--N", "2", "--export-poly", str(pj),
         "--out", str(tmp_path / "x.npz")]
    )
    assert rc == 0
    fit_loaded = load_poly_potential_json(pj)
    assert fit_loaded.coeffs.shape == (7, 7, 7)  # B≠0 → 含 x^6/y^6/x^4y^2/x^2y^4
    fit_orig = build_total_potential_fit(make_params(B_V_per_um4=1e-7))
    rng = np.random.default_rng(3)
    r = np.column_stack(
        [rng.uniform(-30, 30, 30), rng.uniform(-80, 80, 30), np.zeros(30)]
    )
    np.testing.assert_allclose(
        eval_fit_3d(fit_loaded, r), eval_fit_3d(fit_orig, r), rtol=1e-12
    )
