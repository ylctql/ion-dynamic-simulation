"""
动力学模拟核心
- backend: CalculationBackend，与 queue_control/queue_data 交互
- ionsim: C++ 扩展，calculate_trajectory 轨迹计算
"""

from ComputeKernel.backend import CalculationBackend

__all__ = ["CalculationBackend"]
