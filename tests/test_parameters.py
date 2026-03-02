"""参数解析模块测试"""
import numpy as np

from Interface.parameters import Parameters, from_argparse


def test_from_argparse_basic():
    class Args:
        N = 10
        alpha = 0.0
        t0 = 0.0
        time = 1.0  # 1 μs
        device = "cpu"
        calc_method = "VV"

    dt = 1e-8  # 示例 dt
    p = from_argparse(Args(), dt)
    assert p.N == 10
    assert p.device == "cpu"
    assert p.calc_method == "VV"
    assert p.duration == 1.0 / (dt * 1e6)
    assert p.t0 == 0.0


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


def test_parameters_get_r0_get_v0():
    p = Parameters(N=3, r0=np.zeros((3, 3)), v0=np.ones((3, 3)))
    r = p.get_r0()
    v = p.get_v0()
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    np.testing.assert_array_equal(v, np.ones((3, 3)))
