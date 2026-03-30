"""
离子晶格平衡构型计算模块

通过势场拟合与能量最小化求离子晶格的平衡构型。

- fit_potential_3d_quartic: 拟合外势；fit_mode: none(125)/even(27)/quartic(35)/quartic_even(10)/quadratic(4)
- eval_fit_3d, grad_fit_3d: 拟合势场的求值与梯度
- trap/coulomb/total energy: 外势能、库伦势能与总势能
"""

from equilibrium.potential_fit_3d import (
    FitResult3D,
    eval_fit_3d,
    fit_potential_3d_quartic,
    grad_fit_3d,
    hessian_fit_3d,
)
from equilibrium.energy import (
    EnergyBreakdown,
    coulomb_energy_and_grad,
    total_energy_and_grad,
    trap_energy_and_grad,
)
from equilibrium.phonon import (
    PhononResult,
    coulomb_hessian,
    solve_phonon_modes,
    total_hessian,
    trap_hessian,
)

__all__ = [
    "FitResult3D",
    "EnergyBreakdown",
    "eval_fit_3d",
    "fit_potential_3d_quartic",
    "grad_fit_3d",
    "hessian_fit_3d",
    "trap_energy_and_grad",
    "coulomb_energy_and_grad",
    "total_energy_and_grad",
    "PhononResult",
    "trap_hessian",
    "coulomb_hessian",
    "total_hessian",
    "solve_phonon_modes",
]
