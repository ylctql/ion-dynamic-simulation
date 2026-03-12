"""
动力学轨迹分析：平衡位置、阱频、加热率、声子模式等

基于动力学模拟结果 (r_list, v_list, t) 进行后处理分析。
"""
from __future__ import annotations

from .equilibrium import equilibrium_from_trajectory
from .visualize import plot_equilibrium

__all__ = [
    "equilibrium_from_trajectory",
    "plot_equilibrium",
    "run_equilibrium",
]


def __getattr__(name: str):
    """延迟导入 run_equilibrium，避免 python -m 时的 RuntimeWarning"""
    if name == "run_equilibrium":
        from .run_equilibrium import run_equilibrium
        return run_equilibrium
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
