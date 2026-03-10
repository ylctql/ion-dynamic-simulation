"""参数解析模块测试"""
import numpy as np
import pytest

from Interface.parameters import Parameters, from_argparse


def test_from_argparse_basic():
    class Args:
        N = 10
        alpha = 0.0
        t0 = 0.0
        time = 1.0  # 终止时刻 1 μs
        device = "cpu"
        calc_method = "VV"

    dt = 1e-8  # 示例 dt
    p = from_argparse(Args(), dt)
    assert p.N == 10
    assert p.device == "cpu"
    assert p.calc_method == "VV"
    assert p.duration == 1.0 / (dt * 1e6)
    assert p.t0 == 0.0


def test_from_argparse_time_as_end_time():
    """--time 为终止时刻：t0=100, time=200 → duration=100 μs"""
    class Args:
        N = 5
        alpha = 0.0
        t0 = 100.0
        time = 200.0
        device = "cpu"
        calc_method = "VV"

    dt = 1e-8
    p = from_argparse(Args(), dt)
    expected_duration_dt = 100.0 / (dt * 1e6)  # 200-100=100 μs
    assert p.duration == expected_duration_dt


def test_from_argparse_t0_ge_time_raises():
    """t0 >= time 时应报错"""
    class Args:
        N = 5
        alpha = 0.0
        t0 = 200.0
        time = 100.0
        device = "cpu"
        calc_method = "VV"

    with pytest.raises(ValueError, match="终止时刻不能早于或等于起始时刻"):
        from_argparse(Args(), 1e-8)


def test_from_argparse_infinite_time():
    class Args:
        N = 5
        alpha = 0.0
        t0 = 0.0
        time = None
        device = "cpu"
        calc_method = "VV"

    p = from_argparse(Args(), 1e-8)
    assert np.isinf(p.duration)


def test_parameters_isotope_doping():
    p = Parameters(N=6, alpha=0.1)
    assert p.m is not None
    assert len(p.m) == 6


def test_parameters_single_isotope():
    """单同位素模式：alpha 为指定同位素丰度，其余为 Ba135"""
    p = Parameters(N=10, alpha=0.3, isotope_type="Ba133")
    assert p.m is not None
    assert len(p.m) == 10
    # 前 3 个应为 Ba133 (133/135)，后 7 个为 Ba135 (1.0)
    np.testing.assert_array_almost_equal(p.m[:3], 133 / 135)
    np.testing.assert_array_almost_equal(p.m[3:], 1.0)


def test_from_argparse_isotope():
    class Args:
        N = 10
        alpha = 0.2
        isotope = "Ba136"
        t0 = 0.0
        time = None
        device = "cpu"
        calc_method = "VV"

    p = from_argparse(Args(), 1e-8)
    assert p.isotope_type == "Ba136"
    assert p.alpha == 0.2
    # 2 个 Ba136，8 个 Ba135
    np.testing.assert_array_almost_equal(p.m[:2], 136 / 135)
    np.testing.assert_array_almost_equal(p.m[2:], 1.0)


def test_parameters_get_r0_get_v0():
    p = Parameters(N=3, r0=np.zeros((3, 3)), v0=np.ones((3, 3)))
    r = p.get_r0()
    v = p.get_v0()
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    np.testing.assert_array_equal(v, np.ones((3, 3)))
