"""
离子晶格平衡构型计算模块

通过势场拟合与能量最小化求离子晶格的平衡构型。

- fit_potential_3d_quartic: 直接拟合 V(x,y,z) 的 3D 四次多项式
- eval_fit_3d, grad_fit_3d: 拟合势场的求值与梯度
- trap/coulomb/total energy: 外势能、库伦势能与总势能
"""

from equilibrium.potential_fit_3d import (
    FitResult3D,
    eval_fit_3d,
    fit_potential_3d_quartic,
    grad_fit_3d,
)
from equilibrium.energy import (
    EnergyBreakdown,
    coulomb_energy_and_grad,
    total_energy_and_grad,
    trap_energy_and_grad,
)

__all__ = [
    "FitResult3D",
    "EnergyBreakdown",
    "eval_fit_3d",
    "fit_potential_3d_quartic",
    "grad_fit_3d",
    "trap_energy_and_grad",
    "coulomb_energy_and_grad",
    "total_energy_and_grad",
]
