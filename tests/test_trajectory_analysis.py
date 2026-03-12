"""
TrajectoryAnalysis 模块测试：平衡位置计算
"""
from __future__ import annotations

import numpy as np
import pytest

from TrajectoryAnalysis import equilibrium_from_trajectory


def test_equilibrium_mean_constant():
    """恒定轨迹应返回该常量"""
    r_const = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    r_list = [r_const.copy() for _ in range(100)]
    r_eq = equilibrium_from_trajectory(r_list, method="mean")
    np.testing.assert_allclose(r_eq, r_const)


def test_equilibrium_mean_oscillation():
    """正弦振荡的平均应为振幅中心"""
    # r(t) = r0 + A*sin(omega*t), 平均后应为 r0
    r0 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    A = 0.1
    n_steps = 1000
    t = np.linspace(0, 10 * np.pi, n_steps)  # 5 个完整周期
    r_list = [
        r0 + A * np.sin(t[i]) * np.array([[1, 0, 0], [1, 0, 0]])
        for i in range(n_steps)
    ]
    r_eq = equilibrium_from_trajectory(r_list, t=t, method="mean")
    np.testing.assert_allclose(r_eq, r0, atol=1e-10)


def test_equilibrium_skip_initial():
    """skip_initial_dt 应排除初始段"""
    r0 = np.array([[1.0, 0.0, 0.0]])
    # 前 100 步为 [10,10,10]，之后为 r0
    r_list = [np.array([[10.0, 10.0, 10.0]]) for _ in range(100)]
    r_list += [r0.copy() for _ in range(200)]
    t = np.arange(len(r_list), dtype=float)
    r_eq = equilibrium_from_trajectory(
        r_list, t=t, skip_initial_dt=100, method="mean"
    )
    np.testing.assert_allclose(r_eq, r0)


def test_equilibrium_rf_cycle_mean():
    """rf_cycle_mean 对振荡轨迹应收敛到中心"""
    r0 = np.array([[0.0, 0.0, 0.0]])
    n_per = 5
    n_groups = 50
    n_steps = n_per * n_groups
    t = np.arange(n_steps, dtype=float)
    # 叠加 RF 频率振荡 (每 5 点约 1.6 周期) + 慢振荡
    r_list = []
    for i in range(n_steps):
        rf = 0.2 * np.cos(2 * i)  # 2 rad/dt
        sec = 0.1 * np.sin(0.1 * i)  # 慢振荡
        r_list.append(r0 + (rf + sec) * np.array([[1, 0, 0]]))
    r_eq = equilibrium_from_trajectory(
        r_list, t=t, method="rf_cycle_mean", n_per_rf_cycle=n_per
    )
    # 中心应在 0 附近，残余来自非整数周期
    np.testing.assert_allclose(r_eq, r0, atol=0.05)


def test_equilibrium_invalid_method():
    """非法 method 应报错"""
    r_list = [np.zeros((2, 3)) for _ in range(10)]
    with pytest.raises(ValueError, match="method 须为"):
        equilibrium_from_trajectory(r_list, method="invalid")


def test_equilibrium_too_short():
    """过短轨迹应报错"""
    r_list = [np.zeros((2, 3))]
    with pytest.raises(ValueError, match="至少需要 2 个时间步"):
        equilibrium_from_trajectory(r_list)
