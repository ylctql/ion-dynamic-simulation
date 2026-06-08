"""
field_optimize 模块数据类型定义
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OptimizationConfig:
    """优化配置（不可变）"""

    # 目标阱频 (fx, fy, fz) MHz
    target_freq_MHz: tuple[float, float, float]

    # 评估中心 (µm)
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 各轴拟合范围 (µm)
    fit_range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-50.0, 50.0),
        (-20.0, 20.0),
        (-150.0, 150.0),
    )

    # 1D 多项式拟合参数
    n_fit_pts: int = 200          # 每轴采样点数
    fit_degree: int = 2           # 多项式阶数 (2 或 4)

    # 目标函数权重
    w_freq: float = 1.0           # 频率误差
    w_parity: float = 0.1         # 多项式奇偶性惩罚
    w_offdiag: float = 0.1        # Hessian 离轴比惩罚

    # 优化器设置
    maxiter: int = 200
    tol: float = 1e-8
    method: str = "L-BFGS-B"      # "L-BFGS-B" | "Nelder-Mead"

    # 电压上下界
    v_bias_bounds: tuple[float, float] = (-100.0, 100.0)
    v0_rf_bounds: tuple[float, float] = (50.0, 500.0)

    # 模式
    optimize_rf_v0: bool = False  # 是否同时优化 RF 幅值

    # 对称性评估
    symmetry_fit_mode: str = "quartic"
    symmetry_n_pts: int = 6       # 3D 拟合每轴采样点数


@dataclass
class OptimizationResult:
    """优化结果"""

    success: bool
    message: str
    n_iterations: int
    n_evaluations: int

    # 电压配置（优化前后）
    initial_voltages: list[dict]    # [{name, V_bias, V0, type}, ...]
    optimized_voltages: list[dict]

    # 阱频 (MHz)
    initial_freqs_MHz: dict[str, float]    # {f_x, f_y, f_z}
    optimized_freqs_MHz: dict[str, float]
    target_freqs_MHz: dict[str, float]

    # 目标函数值
    initial_objective: float
    final_objective: float

    # 对称性度量
    initial_symmetry: dict | None = None
    optimized_symmetry: dict | None = None
