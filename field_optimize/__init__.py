"""
field_optimize：从目标阱频反推电极电压

给定目标阱频 (fx, fy, fz)，优化电极电压使其匹配，
同时以势场对称性作为正则化惩罚项。
"""
from __future__ import annotations

from .cli import main
from .optimizer import optimize_voltages
from .types import OptimizationConfig, OptimizationResult

__all__ = [
    "main",
    "optimize_voltages",
    "OptimizationConfig",
    "OptimizationResult",
]
