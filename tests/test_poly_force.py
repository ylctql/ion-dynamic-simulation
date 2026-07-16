"""poly_force 多项式拟合力函数测试"""
import numpy as np
import pytest

from FieldParser.poly_force import calc_field_from_poly


def _make_quadratic_grid(n_per_axis=5, span_um=100.0, dl=1e-6, dV=1.0):
    """
    生成一个已知二次势场的规则格点数据。

    V_norm(x,y,z) = a * (x² + y² + z²)  其中 a 为归一化系数
    电场 E = -grad(V) = -2a * (x, y, z)

    Returns: grid_coord, grid_voltage, dl, dV, a (系数)
    """
    # 归一化坐标范围：±span_um µm → ±span_um * 1e-6 / dl
    half = span_um * 1e-6 / dl  # 归一化半范围
    lin = np.linspace(-half, half, n_per_axis)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_coord = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # 归一化势 V_norm = a * (x² + y² + z²)，取 a 使得势场深度合理
    a = 0.5 / (half ** 2)  # V_norm 在边界处 ≈ 1.5
    grid_voltage = (a * (grid_coord ** 2).sum(axis=1)).reshape(-1, 1)

    return grid_coord, grid_voltage, dl, dV, a


def test_calc_field_from_poly_returns_correct_count():
    """返回的 callable 数量应等于电极基数"""
    grid_coord, grid_voltage, dl, dV, a = _make_quadratic_grid(n_per_axis=5)
    # 3 个电极基
    grid_voltage_3 = np.tile(grid_voltage, (1, 3))

    field_interps = calc_field_from_poly(grid_coord, grid_voltage_3, dl, dV)
    assert len(field_interps) == 3


def test_calc_field_from_poly_gradient_quadratic():
    """
    对二次势场 V = a*(x²+y²+z²) 拟合后，梯度应为 -2a*(x,y,z)。
    使用 quadratic 模式（4 项：常数 + u²+v²+w²）可精确拟合。
    """
    n_per_axis = 6
    span_um = 80.0
    dl = 1e-6  # 1 µm 特征长度
    dV = 1.0

    half_norm = span_um * 1e-6 / dl
    a = 0.5 / (half_norm ** 2)

    lin = np.linspace(-half_norm, half_norm, n_per_axis)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_coord = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    grid_voltage = (a * (grid_coord ** 2).sum(axis=1)).reshape(-1, 1)

    field_interps = calc_field_from_poly(
        grid_coord, grid_voltage, dl, dV,
        fit_mode="quadratic", n_pts_per_axis=6,
    )

    # 在格点中心附近采样验证
    test_r = np.array([
        [0.0, 0.0, 0.0],
        [half_norm * 0.3, 0.0, 0.0],
        [0.0, half_norm * 0.5, 0.0],
        [0.0, 0.0, -half_norm * 0.4],
        [half_norm * 0.2, -half_norm * 0.3, half_norm * 0.1],
    ])
    E = field_interps[0](test_r)

    # 解析解：E_norm = -2a * r_norm
    E_exact = -2.0 * a * test_r
    np.testing.assert_allclose(E, E_exact, atol=1e-8, rtol=1e-6,
                               err_msg="二次势场梯度与解析解不符")


def test_calc_field_from_poly_output_shape():
    """返回的 callable 应接受 (M,3) 输入并返回 (M,3) 输出"""
    grid_coord, grid_voltage, dl, dV, _ = _make_quadratic_grid()
    field_interps = calc_field_from_poly(grid_coord, grid_voltage, dl, dV)

    r_single = np.array([[0.0, 0.0, 0.0]])
    E = field_interps[0](r_single)
    assert E.shape == (1, 3)

    r_multi = np.zeros((10, 3))
    E = field_interps[0](r_multi)
    assert E.shape == (10, 3)


def test_calc_field_from_poly_quartic_on_quadratic():
    """
    quartic 模式（35 项）拟合纯二次势场。
    由于基函数多于二次项，R² < 1.0 时会有数值偏差，
    验证梯度在合理精度内（~10%）即可。
    """
    grid_coord, grid_voltage, dl, dV, a = _make_quadratic_grid(n_per_axis=6)

    field_interps = calc_field_from_poly(
        grid_coord, grid_voltage, dl, dV,
        fit_mode="quartic",
    )

    half_norm = grid_coord.max()
    test_r = np.array([
        [half_norm * 0.1, -half_norm * 0.2, half_norm * 0.05],
    ])
    E = field_interps[0](test_r)
    E_exact = -2.0 * a * test_r
    # quartic 有多余项，数值精度约 10%
    np.testing.assert_allclose(E, E_exact, atol=1e-3, rtol=0.1)


def test_calc_field_from_poly_rejects_nonregular_grid():
    """非规则网格应抛出 ValueError"""
    rng = np.random.default_rng(42)
    grid_coord = rng.random((20, 3))  # 随机点，非规则网格
    grid_voltage = rng.random((20, 1))

    with pytest.raises(ValueError, match="格点非规则网格"):
        calc_field_from_poly(grid_coord, grid_voltage, 1e-6, 1.0)


# ============== 整数 fit_mode（任意次多项式拟合）==============

def test_fit_mode_basis_exponents_int_total_degree():
    """正整数 N → 总次数 i+j+k≤N 的完整基，项数 C(N+3,3)；N=4 与 'quartic' 同基。"""
    from math import comb
    from equilibrium.potential_fit_3d import fit_mode_basis_exponents

    assert len(fit_mode_basis_exponents(2)) == comb(2 + 3, 3) == 10
    assert len(fit_mode_basis_exponents(4)) == 35
    assert len(fit_mode_basis_exponents(6)) == 84
    # N=4 与字符串 'quartic' 生成同一组基
    assert fit_mode_basis_exponents(4) == fit_mode_basis_exponents("quartic")
    # 所有单项式均满足总次数约束
    for i, j, k in fit_mode_basis_exponents(6):
        assert i + j + k <= 6


def test_calc_field_from_poly_int_mode_2_exact_quadratic():
    """
    fit_mode=2（总次数 i+j+k≤2，10 项）对纯二次势 V=a(x²+y²+z²) 精确拟合。
    解析梯度 E_norm = -2a*(x,y,z)。
    """
    n_per_axis = 6
    span_um = 80.0
    dl = 1e-6
    dV = 1.0
    half_norm = span_um * 1e-6 / dl
    a = 0.5 / (half_norm ** 2)

    lin = np.linspace(-half_norm, half_norm, n_per_axis)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_coord = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    grid_voltage = (a * (grid_coord ** 2).sum(axis=1)).reshape(-1, 1)

    field_interps = calc_field_from_poly(
        grid_coord, grid_voltage, dl, dV,
        fit_mode=2, n_pts_per_axis=n_per_axis,
    )

    test_r = np.array([
        [half_norm * 0.3, 0.0, 0.0],
        [0.0, half_norm * 0.5, 0.0],
        [half_norm * 0.2, -half_norm * 0.3, half_norm * 0.1],
    ])
    E = field_interps[0](test_r)
    E_exact = -2.0 * a * test_r
    np.testing.assert_allclose(E, E_exact, atol=1e-8, rtol=1e-6,
                               err_msg="fit_mode=2 应精确拟合二次势梯度")


def test_calc_field_from_poly_int_mode_4_equals_quartic():
    """整数 fit_mode=4 与字符串 'quartic' 生成同一组 35 项基，梯度应一致。"""
    grid_coord, grid_voltage, dl, dV, _ = _make_quadratic_grid(n_per_axis=6)

    E_int = calc_field_from_poly(grid_coord, grid_voltage, dl, dV, fit_mode=4)[0]
    E_str = calc_field_from_poly(grid_coord, grid_voltage, dl, dV, fit_mode="quartic")[0]

    h = grid_coord.max()
    test_r = np.array([
        [h * 0.1, -h * 0.2, h * 0.05],
        [0.0, h * 0.3, 0.0],
    ])
    np.testing.assert_allclose(E_int(test_r), E_str(test_r), atol=1e-12,
                               err_msg="fit_mode=4 与 'quartic' 应完全等价")


def test_calc_field_from_poly_int_mode_6_recovers_sixth_order():
    """
    fit_mode=6（总次数 i+j+k≤6，84 项）回收六次势梯度，
    验证 coeffs 扩展到 (7,7,7) 后动态梯度路径正确。
    """
    n_per_axis = 8
    span_um = 80.0
    dl = 1e-6
    dV = 1.0
    half_norm = span_um * 1e-6 / dl

    lin = np.linspace(-half_norm, half_norm, n_per_axis)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_coord = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    xn = grid_coord / half_norm
    a = 0.5  # 归一化到 [-1,1] 后六次方，边界 V_norm ≈ 1.5
    grid_voltage = (a * (xn ** 6).sum(axis=1)).reshape(-1, 1)

    field_interps = calc_field_from_poly(
        grid_coord, grid_voltage, dl, dV,
        fit_mode=6, n_pts_per_axis=n_per_axis,
    )

    test_r = np.array([
        [half_norm * 0.3, 0.0, 0.0],
        [0.0, half_norm * 0.4, 0.0],
        [half_norm * 0.2, -half_norm * 0.25, half_norm * 0.1],
    ])
    E = field_interps[0](test_r)
    # V_norm = a*Σ(r/h)^6 → E_norm = -dV_norm/dr_norm = -6a*r^5/h^6（分量-wise）
    E_exact = -6.0 * a * test_r ** 5 / half_norm ** 6
    np.testing.assert_allclose(E, E_exact, rtol=1e-5, atol=1e-8,
                               err_msg="fit_mode=6 应回收六次势梯度")
