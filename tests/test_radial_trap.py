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
from FieldParser.potential_fit import k2_to_trap_freq_MHz

from radial_trap.cli import create_parser, main
from radial_trap.fitting import fit_rf_ab, load_radial_grid_csv
from radial_trap.lattice import (
    find_radial_equilibrium,
    micromotion_amplitude,
    solve_inplane_phonons,
)
from radial_trap.potential import (
    alpha_eff_um2_per_V,
    bias_potential_V,
    build_total_potential_fit,
    dc_potential_V,
    evaluate_conditions,
    invert_trap_freqs_to_params,
    pseudopotential_V,
    rf_field_V_per_um,
    rf_potential_V,
    total_potential_coefficients,
    total_potential_V,
    trap_freq_MHz_to_k2,
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
    # 收敛残差 ~1e-11 µm 量级（随 seed/范围初始化浮动），断言物理结论"≈0"
    np.testing.assert_allclose(mm.mag_um, 0.0, atol=1e-9)
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

    # ρ 闭式解（docs/radial_trap.md §3）：ρ_x = 1+(2B/A)(x²−3y²)、
    # ρ_y = 1+(2B/A)(3x²−y²)。链上（x=0）仅 y 列有限；离轴探针点两列全查
    ratio2b_a = 2.0 * params.B_V_per_um4 / params.A_V_per_um2
    np.testing.assert_allclose(
        mm.excess_ratio[:, 1], 1.0 + ratio2b_a * (3.0 * x * x - y * y), rtol=1e-9
    )
    probes = np.array([[3.0, 1.0, 0.0], [-2.0, 4.0, 0.0]])
    mm_probe = micromotion_amplitude(probes, params)
    px, py = probes[:, 0], probes[:, 1]
    np.testing.assert_allclose(
        mm_probe.excess_ratio[:, 0], 1.0 + ratio2b_a * (px * px - 3.0 * py * py),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        mm_probe.excess_ratio[:, 1], 1.0 + ratio2b_a * (3.0 * px * px - py * py),
        rtol=1e-9,
    )


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


def test_plot_lattice_labels_sorted_by_chain_axis(monkeypatch):
    """plot_lattice 离子标号按链轴 (x) 坐标排序重编号，右图曲线同步排序。

    数组序号来自随机初始化、与空间位置无关——标号应从链一端 (x 最小) 的 0
    单调递增到另一端，右图 |a_mm| 序列按同一顺序重排，空间依赖才可读。
    """
    from types import SimpleNamespace

    import matplotlib
    import matplotlib.pyplot as plt

    from radial_trap.plots import plot_lattice

    matplotlib.use("Agg")
    # 构造 x 乱序构型：数组下标 0/1/2 分别位于 x=+5/−5/0（链沿 x）
    r = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mm = SimpleNamespace(
        a_mm_um=np.array([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        mag_um=np.array([0.3, 0.3, 0.0]),
        excess_ratio=np.full((3, 3), 1.0),
    )
    result = SimpleNamespace(
        equilibrium=SimpleNamespace(r_eq_um=r), micromotion=mm
    )

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *_a, **_k: None)
    try:
        plot_lattice(result)
        fig = plt.gcf()
    finally:
        real_close("all")

    # 左图标号 = 排序后编号（0 在 x=−5，1 在 x=0，2 在 x=+5）
    ax1 = fig.axes[0]
    by_label = {t.get_text(): t.xy[0] for t in ax1.texts}
    assert set(by_label) == {"0", "1", "2"}
    assert by_label["0"] == pytest.approx(-5.0)
    assert by_label["1"] == pytest.approx(0.0)
    assert by_label["2"] == pytest.approx(5.0)
    # 右图 |a_mm| 按同一顺序重排（x=−5,0,+5 → 0.3,0.0,0.3）
    line = fig.axes[1].lines[0]
    assert line.get_xdata() == pytest.approx([0.0, 1.0, 2.0])
    assert line.get_ydata() == pytest.approx([0.3, 0.0, 0.3])


def test_cli_plot_potential_smoke(tmp_path):
    # --plot-potential：默认参数（D=0 → bias 全零面板守卫）与
    # B/D/F 全非零（角落 r⁴ 爆涨 → 百分位裁剪路径）两种情形
    import matplotlib

    matplotlib.use("Agg")
    png1 = tmp_path / "pot_default.png"
    rc = main(
        ["--N", "2", "--plot-potential", "--potential-out", str(png1),
         "--out", str(tmp_path / "x1.npz")]
    )
    assert rc == 0
    assert png1.exists() and png1.stat().st_size > 0
    png2 = tmp_path / "pot_bneq0.png"
    rc = main(
        ["--N", "2", "--B", "1e-7", "--D", "0.5", "--F=-5e-8",
         "--plot-potential", "--potential-out", str(png2),
         "--out", str(tmp_path / "x2.npz")]
    )
    assert rc == 0
    assert png2.exists() and png2.stat().st_size > 0


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
    # E 取 dataclass 默认（与 CLI 单一来源一致）
    fit_orig = build_total_potential_fit(
        RadialTrapParams(A_V_per_um2=5e-3, B_V_per_um4=1e-7)
    )
    rng = np.random.default_rng(3)
    r = np.column_stack(
        [rng.uniform(-30, 30, 30), rng.uniform(-80, 80, 30), np.zeros(30)]
    )
    np.testing.assert_allclose(
        eval_fit_3d(fit_loaded, r), eval_fit_3d(fit_orig, r), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# v1.1: 势组分函数 / 阱频反算 / 平衡热启动
# ---------------------------------------------------------------------------


def test_potential_components_sum_to_total():
    # pp + bias + dc == total（B,D,F 全非零），一维点与二维网格两种形状
    params = make_params(B_V_per_um4=1e-7, D_dimless=0.5, F_dc_V_per_um4=-5e-8)
    rng = np.random.default_rng(5)
    x = rng.uniform(-40, 40, 64)
    y = rng.uniform(-40, 40, 64)
    total = total_potential_V(x, y, params)
    parts = (
        pseudopotential_V(x, y, params)
        + bias_potential_V(x, y, params)
        + dc_potential_V(x, y, params)
    )
    np.testing.assert_allclose(parts, total, rtol=1e-12, atol=1e-15)
    assert np.all(pseudopotential_V(x, y, params) >= 0.0)  # 赝势恒非负

    xx, yy = np.meshgrid(x, y, indexing="ij")
    total2 = total_potential_V(xx, yy, params)
    assert total2.shape == xx.shape  # 网格输入（UI/绘图路径）
    parts2 = (
        pseudopotential_V(xx, yy, params)
        + bias_potential_V(xx, yy, params)
        + dc_potential_V(xx, yy, params)
    )
    np.testing.assert_allclose(parts2, total2, rtol=1e-12, atol=1e-15)


def test_k2_freq_roundtrip():
    # k2 → f → k2 正逆往返（k2_to_trap_freq_MHz 的精确逆）
    sp = ION_SPECIES["Ba135+"]
    for k2 in (1e-5, 5e-4, 3e-3):
        f = k2_to_trap_freq_MHz(k2, sp.mass_kg, charge=sp.charge_C)
        assert trap_freq_MHz_to_k2(f, sp.mass_kg, sp.charge_C) == pytest.approx(
            k2, rel=1e-12
        )
    with pytest.raises(ValueError):
        trap_freq_MHz_to_k2(0.0, sp.mass_kg, sp.charge_C)
    with pytest.raises(ValueError):
        trap_freq_MHz_to_k2(-1.0, sp.mass_kg, sp.charge_C)
    with pytest.raises(ValueError):
        trap_freq_MHz_to_k2(float("nan"), sp.mass_kg, sp.charge_C)


def test_invert_trap_freqs_roundtrip():
    # 目标 (0.7091, 5.086) MHz → A≈5e-3, E≈−3.5e-4 → evaluate_conditions 频率复原
    f_x, f_y = 0.7091, 5.086
    A, E, q_x = invert_trap_freqs_to_params(f_x, f_y, 35.28, "Ba135+")
    assert A == pytest.approx(5e-3, rel=1e-3)
    assert E == pytest.approx(-3.5e-4, rel=1e-3)
    conds = evaluate_conditions(make_params(A_V_per_um2=A, E_dc_V_per_um2=E))
    assert conds.f_trap_x_mhz == pytest.approx(f_x, rel=1e-9)
    assert conds.f_trap_y_mhz == pytest.approx(f_y, rel=1e-9)
    assert conds.q_mathieu_x == pytest.approx(q_x, rel=1e-12)


def test_invert_trap_freqs_with_bias():
    # D≠0：E 平移 −D·A 后 c_x2/c_y2（经 D·A 分配）仍精确达成目标频率
    f_x, f_y, D = 0.7091, 5.086, 0.5
    A, E, _ = invert_trap_freqs_to_params(f_x, f_y, 35.28, "Ba135+", D=D)
    conds = evaluate_conditions(
        make_params(A_V_per_um2=A, D_dimless=D, E_dc_V_per_um2=E)
    )
    assert conds.f_trap_x_mhz == pytest.approx(f_x, rel=1e-9)
    assert conds.f_trap_y_mhz == pytest.approx(f_y, rel=1e-9)


def test_invert_trap_freqs_degenerate():
    # f_x == f_y → E = −D·A（D=0 时 E=0，二次系数简并）
    A, E, _ = invert_trap_freqs_to_params(1.0, 1.0, 35.28, "Ba135+")
    assert E == pytest.approx(0.0, abs=1e-15)
    conds = evaluate_conditions(make_params(A_V_per_um2=A, E_dc_V_per_um2=E))
    assert conds.xy_degenerate


def test_invert_trap_freqs_invalid():
    with pytest.raises(ValueError):
        invert_trap_freqs_to_params(0.0, 1.0, 35.28, "Ba135+")
    with pytest.raises(ValueError):
        invert_trap_freqs_to_params(1.0, float("nan"), 35.28, "Ba135+")


def test_equilibrium_warm_start_matches_cold():
    # 热启动：同一收敛点、迭代数不增（UI 连续微调场景）
    params = make_params(n_ions=10)
    fit = build_total_potential_fit(params)
    cold = find_radial_equilibrium(fit, params)
    warm = find_radial_equilibrium(fit, params, r_init_um=cold.r_eq_um)
    assert warm.converged
    np.testing.assert_allclose(warm.r_eq_um, cold.r_eq_um, atol=1e-6)
    assert warm.n_iter <= cold.n_iter


# ---------------------------------------------------------------------------
# v1.1: 轴向定位默认（x=链/弱轴，对应 3D xoz 面晶格）
# ---------------------------------------------------------------------------


def test_default_params_chain_on_x_axis():
    # dataclass 默认（单一来源）：E=−3.5e-4 → x 弱轴，N=10 线性链落在 x 轴上
    params = RadialTrapParams(A_V_per_um2=5e-3)  # 其余取默认
    assert params.E_dc_V_per_um2 == pytest.approx(-3.5e-4)
    assert params.x_range_um == (-30.0, 30.0)
    assert params.y_range_um == (-10.0, 10.0)
    conds = evaluate_conditions(params)
    assert conds.c_x2 < conds.c_y2  # x = 弱/链轴
    assert conds.f_trap_x_mhz == pytest.approx(0.709, rel=2e-3)
    assert conds.f_trap_y_mhz == pytest.approx(5.09, rel=2e-3)
    assert conds.pass_cross and conds.pass_y_high
    assert conds.pass_x_confine and conds.pass_y_confine
    fit = build_total_potential_fit(params)
    eq = find_radial_equilibrium(fit, params)
    assert eq.converged and not eq.hit_bounds
    np.testing.assert_allclose(eq.r_eq_um[:, 1], 0.0, atol=1e-6)  # 链在 x 轴
    assert np.abs(eq.r_eq_um[:, 0]).max() > 8.0  # 链跨度 ~±10.7 µm


def test_cli_default_npz_new_defaults(tmp_path):
    # CLI 默认（单一来源）→ npz 带 E=−3.5e-4 与新 ranges，N=2 链沿 x
    out = tmp_path / "d.npz"
    rc = main(["--N", "2", "--out", str(out)])
    assert rc == 0
    d = np.load(out)
    assert float(d["params_E_dc_V_per_um2"]) == pytest.approx(-3.5e-4)
    assert tuple(d["params_x_range_um"]) == (-30.0, 30.0)
    assert tuple(d["params_y_range_um"]) == (-10.0, 10.0)
    np.testing.assert_allclose(d["r_eq_um"][:, 1], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# v1.1: CLI --from-freq 阱频反算
# ---------------------------------------------------------------------------


def test_cli_from_freq_roundtrip(tmp_path, capsys):
    # --from-freq 0.7091 5.086 → A≈5e-3, E≈−3.5e-4；stdout 换算行 + npz 一致
    out = tmp_path / "f.npz"
    rc = main(["--from-freq", "0.7091", "5.086", "--N", "2", "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "阱频反算" in captured.out
    assert "q_x" in captured.out
    d = np.load(out)
    assert float(d["params_A_V_per_um2"]) == pytest.approx(5e-3, rel=1e-3)
    assert float(d["params_E_dc_V_per_um2"]) == pytest.approx(-3.5e-4, rel=1e-2)
    assert float(d["cond_f_trap_x_mhz"]) == pytest.approx(0.7091, rel=1e-6)
    assert float(d["cond_f_trap_y_mhz"]) == pytest.approx(5.086, rel=1e-6)


def test_cli_from_freq_with_bias_from_json(tmp_path):
    # D 来自 json 参与反算（E 平移 −D·A）；species/freq_rf 同样可来自 json
    jp = tmp_path / "p.json"
    jp.write_text(json.dumps({"D": 0.5, "species": "Ba135+"}), encoding="utf-8")
    out = tmp_path / "f.npz"
    rc = main(["--from-freq", "0.7091", "5.086", "--json", str(jp),
               "--N", "2", "--out", str(out)])
    assert rc == 0
    d = np.load(out)
    assert float(d["params_D_dimless"]) == pytest.approx(0.5)
    _, e_solved, _ = invert_trap_freqs_to_params(0.7091, 5.086, 35.28, "Ba135+", D=0.5)
    assert float(d["params_E_dc_V_per_um2"]) == pytest.approx(e_solved, rel=1e-12)
    assert float(d["cond_f_trap_x_mhz"]) == pytest.approx(0.7091, rel=1e-6)


def test_cli_from_freq_conflicts(tmp_path, capsys):
    # 四种冲突（--A / --E / json 含 A / json 含 E）→ parser.error (exit 2)
    jp_a = tmp_path / "pa.json"
    jp_a.write_text(json.dumps({"A": 5e-3}), encoding="utf-8")
    jp_e = tmp_path / "pe.json"
    jp_e.write_text(json.dumps({"E": -3.5e-4}), encoding="utf-8")
    base = ["--from-freq", "0.7", "5.0"]
    for extra in (
        ["--A", "5e-3"],
        ["--E=-3.5e-4"],
        ["--json", str(jp_a)],
        ["--json", str(jp_e)],
    ):
        with pytest.raises(SystemExit) as ei:
            main(base + extra + ["--N", "2", "--out", str(tmp_path / "x.npz")])
        assert ei.value.code == 2
        assert "冲突" in capsys.readouterr().err


def test_cli_from_freq_invalid_freq_exits(tmp_path, capsys):
    # f≤0 → 反算 ValueError → parser.error (exit 2)
    with pytest.raises(SystemExit) as ei:
        main(["--from-freq", "0.0", "5.0", "--out", str(tmp_path / "x.npz")])
    assert ei.value.code == 2
    assert "反算失败" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 格点 CSV 拟合 A/B（--fit-ab）
# ---------------------------------------------------------------------------

_DEF_A, _DEF_B, _DEF_V0 = 5.0e-3, -1.0e-8, 0.37


def _write_grid_csv(path, a=_DEF_A, b=_DEF_B, v0=_DEF_V0, noise=0.0, seed=7):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-40.0, 40.0, 15)
    ys = np.linspace(-25.0, 25.0, 11)
    xx, yy = np.meshgrid(xs, ys)
    vv = (v0 + a * (xx ** 2 - yy ** 2)
          + b * (xx ** 4 - 6.0 * xx ** 2 * yy ** 2 + yy ** 4))
    if noise:
        vv = vv + rng.normal(0.0, noise, vv.shape)
    lines = ["x_um,y,v_rf"]
    lines += [f"{xi:.6f},{yi:.6f},{vi:.9e}"
              for xi, yi, vi in zip(xx.ravel(), yy.ravel(), vv.ravel())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_fit_rf_ab_recovers_known_coeffs(tmp_path):
    # 无噪声：A/B/V0 精确回收；有噪声：误差同量级且 R² 接近 1
    xs = np.linspace(-40.0, 40.0, 21)
    ys = np.linspace(-30.0, 30.0, 17)
    xx, yy = np.meshgrid(xs, ys)
    q2 = xx ** 2 - yy ** 2
    q4 = xx ** 4 - 6.0 * xx ** 2 * yy ** 2 + yy ** 4
    clean = fit_rf_ab(xx, yy, _DEF_V0 + _DEF_A * q2 + _DEF_B * q4)  # meshgrid 二维阵直接传入
    assert clean.A_V_per_um2 == pytest.approx(_DEF_A, rel=1e-10)
    assert clean.B_V_per_um4 == pytest.approx(_DEF_B, rel=1e-10)
    assert clean.v_offset_V == pytest.approx(_DEF_V0, abs=1e-12)
    assert clean.rms_residual_V == pytest.approx(0.0, abs=1e-11)  # q4 基达 40⁴，lstsq 余差 ~1e-12
    assert clean.r_squared == pytest.approx(1.0, abs=1e-12)
    assert clean.n_points == 21 * 17
    assert clean.x_range_um == (-40.0, 40.0)

    rng = np.random.default_rng(3)
    noisy = fit_rf_ab(xx.ravel(), yy.ravel(),
                      (_DEF_V0 + _DEF_A * q2 + _DEF_B * q4
                       + rng.normal(0.0, 1e-5, q2.shape)).ravel())
    assert noisy.A_V_per_um2 == pytest.approx(_DEF_A, rel=1e-4)
    assert noisy.B_V_per_um4 == pytest.approx(_DEF_B, rel=2e-2)
    assert noisy.r_squared > 0.999
    assert noisy.rms_residual_V == pytest.approx(1e-5, rel=0.5)


def test_load_radial_grid_csv_variants(tmp_path):
    # 表头别名 + 无表头 3 列 + BOM；错误分支：缺列 / 列数错 / 点数不足
    p = tmp_path / "g.csv"
    _write_grid_csv(p)
    gx, gy, gv = load_radial_grid_csv(str(p))
    assert gx.size == 15 * 11
    assert gv[0] == pytest.approx(
        _DEF_V0 + _DEF_A * (40.0 ** 2 - 25.0 ** 2)
        + _DEF_B * (40.0 ** 4 - 6 * 40.0 ** 2 * 25.0 ** 2 + 25.0 ** 4),
        rel=1e-9,
    )

    p2 = tmp_path / "nohdr.csv"
    p2.write_text("-40,-25,1.0\n0,0,2.0\n40,25,3.0\n0,25,4.0\n", encoding="utf-8")
    gx2, gy2, gv2 = load_radial_grid_csv(str(p2))
    assert list(gx2) == [-40.0, 0.0, 40.0, 0.0]
    assert list(gv2) == [1.0, 2.0, 3.0, 4.0]

    pb = tmp_path / "bom.csv"
    pb.write_bytes(b"\xef\xbb\xbfx,y,v\n0,0,0\n10,0,1\n-10,0,-1\n0,10,-1\n")
    gxb, gyb, gvb = load_radial_grid_csv(str(pb))
    assert list(gxb) == [0.0, 10.0, -10.0, 0.0]

    pbad = tmp_path / "bad.csv"
    pbad.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n", encoding="utf-8")
    with pytest.raises(ValueError, match="找不到 x 坐标"):
        load_radial_grid_csv(str(pbad))

    p4 = tmp_path / "twocol.csv"
    p4.write_text("1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="3 列"):
        load_radial_grid_csv(str(p4))

    with pytest.raises(ValueError, match="不足"):
        p3 = tmp_path / "few.csv"
        p3.write_text("x,y,v\n0,0,0\n10,0,1\n-10,0,-1\n", encoding="utf-8")
        load_radial_grid_csv(str(p3))


def _write_comsol_csv(path, unit="mm", expr="esbe.V (V)", n_extra_expr=0):
    # 仿 Comsol 导出：% 元数据行（含带逗号的引号日期）+ % x,y,<表达式> 列头 + 数值行
    xs = np.linspace(-40.0, 40.0, 15)
    ys = np.linspace(-25.0, 25.0, 11)
    xx, yy = np.meshgrid(xs, ys)
    vv = (_DEF_V0 + _DEF_A * (xx ** 2 - yy ** 2)
          + _DEF_B * (xx ** 4 - 6.0 * xx ** 2 * yy ** 2 + yy ** 4))
    factor = {"mm": 1e-3, "um": 1.0, "µm": 1.0, "in": 2.54e-5}[unit]
    lines = [
        "% Model,demo.mph",
        "% Version,COMSOL 6.4.0.293",
        '% Date,"Sep 18 2026, 08:34"',
        "% Dimension,2",
        "% Nodes,165",
        f"% Length unit,{unit}",
        "% x,y," + ",".join([expr] + [f"e{i} (V)" for i in range(n_extra_expr)]),
    ]
    for xi, yi, vi in zip(xx.ravel(), yy.ravel(), vv.ravel()):
        row = [f"{xi * factor:.9e}", f"{yi * factor:.9e}", f"{vi:.9e}"]
        row += ["1.0e+00"] * n_extra_expr
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_radial_grid_csv_comsol(tmp_path):
    # Comsol 导出：% 元数据行剥离 + Length unit mm→µm 换算 + 任意电势列名 + 拟合回收 A/B
    p = tmp_path / "comsol.csv"
    _write_comsol_csv(p, unit="mm")
    gx, gy, gv = load_radial_grid_csv(str(p))
    assert gx.size == 15 * 11
    assert gx.min() == pytest.approx(-40.0, rel=1e-9)
    assert gy.max() == pytest.approx(25.0, rel=1e-9)
    fit = fit_rf_ab(gx, gy, gv)
    assert fit.A_V_per_um2 == pytest.approx(_DEF_A, rel=1e-6)
    assert fit.B_V_per_um4 == pytest.approx(_DEF_B, rel=1e-6)
    assert fit.v_offset_V == pytest.approx(_DEF_V0, rel=1e-9)
    assert fit.x_range_um == (-40.0, 40.0)
    assert fit.y_range_um == (-25.0, 25.0)

    # um / µm（U+00B5）单位不换算
    for u in ("um", "µm"):
        p2 = tmp_path / f"comsol_{u}.csv"
        _write_comsol_csv(p2, unit=u)
        gx2, _, _ = load_radial_grid_csv(str(p2))
        assert gx2.max() == pytest.approx(40.0, rel=1e-9)

    # 未知单位报错；多表达式导出（4 列）报错并提示只保留一个电势表达式
    p3 = tmp_path / "bad_unit.csv"
    _write_comsol_csv(p3, unit="in")
    with pytest.raises(ValueError, match="无法识别的坐标单位"):
        load_radial_grid_csv(str(p3))
    p4 = tmp_path / "two_expr.csv"
    _write_comsol_csv(p4, n_extra_expr=1)
    with pytest.raises(ValueError, match="Comsol 列头.*表达式"):
        load_radial_grid_csv(str(p4))


def test_load_radial_grid_csv_headerless_v_name_fallback(tmp_path):
    # 非 Comsol 简单表头、恰 3 列且前两列为 x/y：电势列名不限（回退路径）
    p = tmp_path / "hdr.csv"
    p.write_text("x,y,esbe.V (V)\n0,0,0\n10,0,1\n-10,0,-1\n0,10,-2\n",
                 encoding="utf-8")
    gx, gy, gv = load_radial_grid_csv(str(p))
    assert list(gx) == [0.0, 10.0, -10.0, 0.0]
    assert list(gv) == [0.0, 1.0, -1.0, -2.0]

    # 多于 3 列且电势列名不在别名表 → 无法唯一确定，报错
    p2 = tmp_path / "amb.csv"
    p2.write_text("x,y,a,b\n0,0,0,0\n10,0,1,1\n-10,0,-1,1\n0,10,-2,1\n",
                  encoding="utf-8")
    with pytest.raises(ValueError, match="无法确定电势列"):
        load_radial_grid_csv(str(p2))


def test_fit_rf_ab_range_restriction():
    # 高阶（六阶）泄漏：全域拟合 A/B 被污染，限制拟合范围去偏
    # （q6 系数与 2D_demo.csv 实测同比例：C6/A = -6.149e-17/7.833e-7）
    xs = np.linspace(-300.0, 300.0, 61)
    ys = np.linspace(-100.0, 100.0, 41)
    xx, yy = np.meshgrid(xs, ys)
    q2 = xx ** 2 - yy ** 2
    q4 = xx ** 4 - 6.0 * xx ** 2 * yy ** 2 + yy ** 4
    q6 = np.real((xx + 1j * yy) ** 6)
    c6 = _DEF_A * (-6.149e-17 / 7.833e-7)
    vv = _DEF_V0 + _DEF_A * q2 + _DEF_B * q4 + c6 * q6

    full = fit_rf_ab(xx, yy, vv)
    assert full.x_range_um == (-300.0, 300.0)
    assert abs(full.A_V_per_um2 - _DEF_A) / _DEF_A > 0.2  # 全域确被污染

    restr = fit_rf_ab(xx, yy, vv, fit_range_um=(80.0, 80.0))
    n_expect = int(((np.abs(xx) <= 80.0) & (np.abs(yy) <= 80.0)).sum())
    assert restr.n_points == n_expect
    assert restr.x_range_um == (-80.0, 80.0)
    assert restr.y_range_um == (-80.0, 80.0)
    assert restr.A_V_per_um2 == pytest.approx(_DEF_A, rel=2e-2)
    assert restr.B_V_per_um4 == pytest.approx(_DEF_B, rel=0.2)
    assert (abs(restr.A_V_per_um2 - _DEF_A)
            < abs(full.A_V_per_um2 - _DEF_A) / 5)
    assert (abs(restr.B_V_per_um4 - _DEF_B)
            < abs(full.B_V_per_um4 - _DEF_B) / 5)

    # 非正/空范围报错
    with pytest.raises(ValueError, match="格点不足"):
        fit_rf_ab(xx, yy, vv, fit_range_um=(1e-3, 1e-3))
    with pytest.raises(ValueError, match="正有限"):
        fit_rf_ab(xx, yy, vv, fit_range_um=(-80.0, 80.0))


def test_cli_fit_ab_range(tmp_path, capsys):
    # --fit-ab-range 限制拟合域：stdout 报告范围；单独使用（无 --fit-ab）exit 2
    p = tmp_path / "g.csv"
    _write_grid_csv(p)
    out = tmp_path / "f.npz"
    rc = main(["--fit-ab", str(p), "--fit-ab-range", "30", "25",
               "--N", "2", "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "限制 |x|≤30" in captured.out
    d = np.load(out)
    # 限域后格点写入精度（%.9e）主导回收误差，容差略宽于全域往返
    assert float(d["params_A_V_per_um2"]) == pytest.approx(_DEF_A, rel=1e-7)

    with pytest.raises(SystemExit) as ei:
        main(["--fit-ab-range", "30", "25", "--N", "2",
              "--out", str(tmp_path / "x.npz")])
    assert ei.value.code == 2
    assert "同用" in capsys.readouterr().err


def test_cli_fit_ab_roundtrip(tmp_path, capsys):
    # 拟合值作为运行参数：stdout 报告 + npz 中 A/B 精确一致
    p = tmp_path / "g.csv"
    _write_grid_csv(p)
    out = tmp_path / "f.npz"
    rc = main(["--fit-ab", str(p), "--N", "2", "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "格点拟合 A/B" in captured.out
    assert "R²" in captured.out
    d = np.load(out)
    assert float(d["params_A_V_per_um2"]) == pytest.approx(_DEF_A, rel=1e-9)
    assert float(d["params_B_V_per_um4"]) == pytest.approx(_DEF_B, rel=1e-9)


def test_cli_fit_ab_conflicts(tmp_path, capsys):
    # --A / --B / --from-freq / json 含 A 或 B → parser.error (exit 2)
    p = tmp_path / "g.csv"
    _write_grid_csv(p)
    jp = tmp_path / "pa.json"
    jp.write_text(json.dumps({"B": -1e-8}), encoding="utf-8")
    for extra in (
        ["--A", "5e-3"],
        ["--B", "1e-8"],
        ["--from-freq", "0.7", "5.0"],
        ["--json", str(jp)],
    ):
        with pytest.raises(SystemExit) as ei:
            main(["--fit-ab", str(p)] + extra
                 + ["--N", "2", "--out", str(tmp_path / "x.npz")])
        assert ei.value.code == 2
        assert "冲突" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# v1.1: UI 构造冒烟（Agg 无头）+ --ui 后端守卫
# ---------------------------------------------------------------------------


def test_ui_construct_and_refresh_smoke(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from radial_trap.ui import RadialTrapUI

    ui = RadialTrapUI(make_params(n_ions=3), n_pts=(41, 31))
    try:
        # 初始读出已填充（系数/频率/q/条件）
        text = ui.readout_text.get_text()
        assert "c_x2" in text and "q_x" in text and "PASS" in text
        # 廉价路径：滑块 E → −3.5e-4 → x 变弱轴（不重解平衡）
        ui.sliders["E_dc_V_per_um2"].set_val(-3.5e-4)
        assert ui.params.E_dc_V_per_um2 == pytest.approx(-3.5e-4)
        assert ui.conditions.c_x2 < ui.conditions.c_y2
        # 重解路径：热启动 → 链翻到 x 轴
        ui.refresh(compute_eq=True)
        assert ui.eq.converged
        np.testing.assert_allclose(ui.eq.r_eq_um[:, 1], 0.0, atol=1e-5)
        # 存档 → --json 读回（严格键集兼容）
        jp = ui.save_json(tmp_path / "ui_params.json")
        rc = main(["--json", str(jp), "--out", str(tmp_path / "r.npz")])
        assert rc == 0
        d = np.load(tmp_path / "r.npz")
        assert float(d["params_E_dc_V_per_um2"]) == pytest.approx(-3.5e-4)
    finally:
        ui.close()


def test_ui_solve_button_updates_sliders(tmp_path):
    # Solve A,E：文本框目标频率 → 反算 → 滑块/参数更新 + 一次重解
    import matplotlib

    matplotlib.use("Agg")
    from radial_trap.ui import RadialTrapUI

    ui = RadialTrapUI(make_params(n_ions=2), n_pts=(41, 31))
    try:
        ui.tb_fx.set_val("0.7091")
        ui.tb_fy.set_val("5.086")
        ui._on_solve()
        assert ui.params.A_V_per_um2 == pytest.approx(5e-3, rel=1e-3)
        assert ui.params.E_dc_V_per_um2 == pytest.approx(-3.5e-4, rel=1e-2)
        assert "Solved" in ui._status_text.get_text()
        # 非法输入 → 状态报错、参数不变
        before = ui.params
        ui.tb_fx.set_val("-1")
        ui._on_solve()
        assert ui.params == before
        assert "Solve failed" in ui._status_text.get_text()
    finally:
        ui.close()


def test_cli_ui_requires_interactive_backend(tmp_path, capsys, monkeypatch):
    import matplotlib

    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    with pytest.raises(SystemExit) as ei:
        main(["--ui", "--N", "2", "--out", str(tmp_path / "x.npz")])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "--ui" in err and "交互式" in err


# ---------------------------------------------------------------------------
# v1.2: UI 数值/范围/N/区域文本框 + 键盘步进 + DB+F=0 约束
# ---------------------------------------------------------------------------


def _make_ui(**params_kwargs):
    import matplotlib

    matplotlib.use("Agg")
    from radial_trap.ui import RadialTrapUI

    return RadialTrapUI(make_params(**params_kwargs), n_pts=(41, 31))


def test_ui_value_box_exact_set(tmp_path):
    ui = _make_ui(n_ions=2)
    try:
        # 界内精确设值：滑块/参数/文本框三方同步 + 一次重解
        assert ui.set_coeff_value("A_V_per_um2", 8e-3)
        assert ui.params.A_V_per_um2 == pytest.approx(8e-3)
        assert ui.sliders["A_V_per_um2"].val == pytest.approx(8e-3)
        assert ui.value_boxes["A_V_per_um2"].text == "8.000e-03"
        assert ui.eq.converged
        # 界外设值：自动扩展滑块范围而非拒绝
        assert ui.set_coeff_value("A_V_per_um2", 3e-2)
        assert ui.sliders["A_V_per_um2"].valmax >= 3e-2
        assert ui.params.A_V_per_um2 == pytest.approx(3e-2)
        assert ui.range_boxes["A_V_per_um2"].text == "0.0001,0.03"
        # f_x/f_y 框随刷新同步当前阱频（Solve 的初值提示）
        assert ui.tb_fx.text == f"{ui.conditions.f_trap_x_mhz:.4f}"
        # 非法输入：状态报错、参数不变
        before = ui.params.A_V_per_um2
        assert not ui.set_coeff_value("A_V_per_um2", float("nan"))
        ui._on_value_submit("A_V_per_um2", "abc")
        assert ui.params.A_V_per_um2 == before
        assert "not a number" in ui._status_text.get_text()
    finally:
        ui.close()


def test_ui_range_box_recreates_slider(tmp_path):
    ui = _make_ui(n_ions=2)
    try:
        # 改范围：滑块按新界重建，当前值保持
        assert ui.set_coeff_range("B_V_per_um4", -1e-6, 1e-6)
        s = ui.sliders["B_V_per_um4"]
        assert s.valmin == pytest.approx(-1e-6)
        assert s.valmax == pytest.approx(1e-6)
        assert ui.params.B_V_per_um4 == pytest.approx(0.0)
        assert ui.range_boxes["B_V_per_um4"].text == "-1e-06,1e-06"
        # 收窄范围：越界的当前值被裁剪并同步进 params
        ui.set_coeff_value("B_V_per_um4", 5e-7, ressolve=False)
        assert ui.set_coeff_range("B_V_per_um4", -1e-7, 1e-7)
        assert ui.params.B_V_per_um4 == pytest.approx(1e-7)
        # 非法：lo >= hi / 解析失败 → 报错且界不变
        assert not ui.set_coeff_range("B_V_per_um4", 1e-6, -1e-6)
        ui._on_range_submit("B_V_per_um4", "junk")
        assert "bad range" in ui._status_text.get_text()
        assert ui.sliders["B_V_per_um4"].valmin == pytest.approx(-1e-7)
    finally:
        ui.close()


def test_ui_n_ions_and_region(tmp_path):
    ui = _make_ui(n_ions=3)
    try:
        # N：热启动增离子后重解
        assert ui.set_n_ions(5)
        assert ui.params.n_ions == 5
        assert ui.eq.r_eq_um.shape == (5, 3)
        assert ui.tb_n.text == "5"
        assert not ui.set_n_ions(0)
        assert not ui.set_n_ions(201)
        # 区域：网格 + pcolormesh 重建
        assert ui.set_region("x", -40.0, 40.0)
        assert ui.params.x_range_um == (-40.0, 40.0)
        assert ui._grid_x[0] == pytest.approx(-40.0)
        assert ui._grid_x[-1] == pytest.approx(40.0)
        assert ui.tb_xrange.text == "-40,40"
        # 跨度守卫与解析失败
        assert not ui.set_region("y", 0.0, 1e-6)  # span < 1e-3 µm
        ui._on_region_submit("x", "a,b")
        assert "bad range" in ui._status_text.get_text()
    finally:
        ui.close()


def test_ui_constraint_db_plus_f(tmp_path):
    from radial_trap.ui import _F_FIELD

    ui = _make_ui(n_ions=2, B_V_per_um4=1e-7, D_dimless=0.5)
    try:
        # 勾选（set_active(0) 触发 on_clicked → 以复选框状态为准）
        ui.cb_constraint.set_active(0)
        assert ui._f_constrained
        f_expected = -ui.params.D_dimless * ui.params.B_V_per_um4
        assert ui.params.F_dc_V_per_um4 == pytest.approx(f_expected)
        assert ui.conditions.db_plus_f == pytest.approx(0.0, abs=1e-30)
        # F 控件隐藏，F 不再独立可调
        assert not ui._slider_ax_map[_F_FIELD].get_visible()
        assert not ui.set_coeff_value(_F_FIELD, 1e-7)
        assert "constrained" in ui._status_text.get_text()
        # 拖 B/D → F 联动（廉价路径下即生效）
        ui.sliders["D_dimless"].set_val(0.8)
        assert ui.params.F_dc_V_per_um4 == pytest.approx(-0.8e-7)
        # 取消勾选：F 冻结在当前联动值，B 再变 F 不动
        ui.cb_constraint.set_active(0)
        assert not ui._f_constrained
        f_frozen = ui.params.F_dc_V_per_um4
        ui.sliders["B_V_per_um4"].set_val(2e-7)
        assert ui.params.F_dc_V_per_um4 == pytest.approx(f_frozen)
        assert ui._slider_ax_map[_F_FIELD].get_visible()
        # 程序化开关路径：复选框视觉同步（守卫下不重复执行、幂等）
        ui.set_constraint(True)
        assert ui.cb_constraint.get_status() == [True]
        assert ui._f_constrained
        ui.set_constraint(True)
        assert ui.cb_constraint.get_status() == [True]
    finally:
        ui.close()


def test_ui_keyboard_stepping(tmp_path):
    from types import SimpleNamespace

    ui = _make_ui(n_ions=2)
    try:
        # 滑块轴上 → 步进 1% 跨度；重解推迟到 key_release
        s = ui.sliders["E_dc_V_per_um2"]
        sax = ui._slider_ax_map["E_dc_V_per_um2"]
        v0, span = s.val, s.valmax - s.valmin
        ui._on_key(SimpleNamespace(key="right", inaxes=sax))
        assert s.val == pytest.approx(v0 + 0.01 * span)
        assert ui.params.E_dc_V_per_um2 == pytest.approx(s.val)
        assert ui._needs_resolve
        ui._on_key_release(SimpleNamespace(key="right"))
        assert not ui._needs_resolve
        assert ui.eq.converged
        # Home/End 跳端点
        ui._on_key(SimpleNamespace(key="end", inaxes=sax))
        assert s.val == pytest.approx(s.valmax)

        # 聚焦数值框：↑↓ 步进（Shift ×5），鼠标位置无关
        vb = ui.value_boxes["A_V_per_um2"]
        vb.capturekeystrokes = True
        a0 = ui.params.A_V_per_um2
        s_a = ui.sliders["A_V_per_um2"]
        span_a = s_a.valmax - s_a.valmin
        ui._on_key(SimpleNamespace(key="shift+up", inaxes=None))
        assert ui.params.A_V_per_um2 == pytest.approx(a0 + 0.05 * span_a)
        assert ui._needs_resolve
        vb.capturekeystrokes = False
        ui._on_key_release(SimpleNamespace(key="shift+up"))

        # 聚焦 N 框：↑ +1 离子并重解
        ui.tb_n.capturekeystrokes = True
        ui._on_key(SimpleNamespace(key="up", inaxes=None))
        assert ui.params.n_ions == 3
        assert ui.eq.r_eq_um.shape[0] == 3
        ui.tb_n.capturekeystrokes = False

        # 聚焦区域框：↑ 以中心为锚放宽 10%
        ui.tb_xrange.capturekeystrokes = True
        ui._on_key(SimpleNamespace(key="up", inaxes=None))
        assert ui.params.x_range_um == pytest.approx((-33.0, 33.0))
        ui.tb_xrange.capturekeystrokes = False
    finally:
        ui.close()


def test_ui_release_debounce_headless(tmp_path):
    # v1.3：Agg（无定时器后端）下 release 的重解立即冲刷；非滑块轴 release
    # 无副作用；_flush_resolve 清挂起标记
    from types import SimpleNamespace

    ui = _make_ui(n_ions=2)
    try:
        sax = ui._slider_ax_map["E_dc_V_per_um2"]
        ui.sliders["E_dc_V_per_um2"].set_val(-3.5e-4)  # 廉价路径 + 挂起标记
        assert ui._needs_resolve
        ui._on_release(SimpleNamespace(inaxes=sax))
        assert not ui._needs_resolve  # 无头后端立即执行
        assert ui.eq.converged
        # 非滑块轴 release：不触发重解，标记保持挂起
        ui.sliders["A_V_per_um2"].set_val(8e-3)
        assert ui._needs_resolve
        ui._on_release(SimpleNamespace(inaxes=None))
        assert ui._needs_resolve
        ui._flush_resolve()
        assert not ui._needs_resolve
    finally:
        ui.close()


# ---------------------------------------------------------------------------
# v1.4: --export-fz 轴向囚禁（导出势 z² 项，3D 链的前提）
# ---------------------------------------------------------------------------


def test_augment_fit_axial_confinement():
    # z² 项系数 = c_z2·L²；eval 差 = c_z2·z²；Hessian zz = 2c_z2 → f_z 复原
    from radial_trap.potential import augment_fit_with_axial_confinement

    sp = ION_SPECIES["Ba135+"]
    f_z = 0.55
    c_z2 = trap_freq_MHz_to_k2(f_z, sp.mass_kg, sp.charge_C)
    fit = build_total_potential_fit(make_params())
    fit_z = augment_fit_with_axial_confinement(fit, f_z, "Ba135+")
    assert (0, 0, 2) in fit_z.basis_exps
    assert fit_z.coeffs[0, 0, 2] == pytest.approx(c_z2 * fit.scale_um**2)
    # 原 fit 不被修改（z 项仍缺失）
    assert (0, 0, 2) not in fit.basis_exps
    # 幂等：对已增广 fit 重复调用只覆盖系数
    fit_z2 = augment_fit_with_axial_confinement(fit_z, f_z, "Ba135+")
    np.testing.assert_allclose(
        eval_fit_3d(fit_z2, np.zeros((1, 3))),
        eval_fit_3d(fit_z, np.zeros((1, 3))),
    )
    z = np.array([[0.0, 0.0, 7.0]])
    np.testing.assert_allclose(
        eval_fit_3d(fit_z, z) - eval_fit_3d(fit_z, np.zeros((1, 3))),
        c_z2 * 49.0,
        rtol=1e-12,
    )
    h = hessian_fit_3d(fit_z, z)[0]  # 单离子 3×3 块
    f_back = k2_to_trap_freq_MHz(h[2, 2] / 2.0, sp.mass_kg, charge=sp.charge_C)
    assert f_back == pytest.approx(f_z, rel=1e-12)


def test_cli_export_fz(tmp_path):
    # --export-fz + --export-poly → JSON 含 z^2 项且读回一致；
    # 不带 --export-poly / 非正频率 → exit 2
    from equilibrium.potential_fit_3d import load_poly_potential_json

    pj = tmp_path / "poly_fz.json"
    rc = main(
        ["--N", "2", "--export-poly", str(pj), "--export-fz", "0.55",
         "--out", str(tmp_path / "x.npz")]
    )
    assert rc == 0
    fit_loaded = load_poly_potential_json(pj)
    assert (0, 0, 2) in fit_loaded.basis_exps
    sp = ION_SPECIES["Ba135+"]
    c_z2 = trap_freq_MHz_to_k2(0.55, sp.mass_kg, sp.charge_C)
    z = np.array([[0.0, 0.0, 5.0], [3.0, -2.0, -8.0]])
    fit_default = build_total_potential_fit(RadialTrapParams(A_V_per_um2=5e-3))
    np.testing.assert_allclose(
        eval_fit_3d(fit_loaded, z) - eval_fit_3d(fit_default, z),
        c_z2 * z[:, 2] ** 2,
        rtol=1e-12,
    )
    with pytest.raises(SystemExit) as ei:
        main(["--export-fz", "0.55"])
    assert ei.value.code == 2
    with pytest.raises(SystemExit) as ei:
        main(["--export-poly", str(tmp_path / "p2.json"), "--export-fz=-0.5"])
    assert ei.value.code == 2
