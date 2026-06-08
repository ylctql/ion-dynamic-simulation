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
