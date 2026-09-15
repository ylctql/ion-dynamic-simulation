"""radial_trap — 2D 径向多项式势阱模块。

以 Laplace 多项式基解析指定 RF+DC 势（A,B,D,E,F），求解径向平面 (x,y)
平衡构型、各离子解析 micromotion 幅度与晶格成形条件；可选面内声子。

运行: python -m radial_trap --N 10   （详见 docs/radial_trap.md）
"""
from radial_trap.lattice import (
    find_radial_equilibrium,
    initial_positions,
    micromotion_amplitude,
    solve_inplane_phonons,
)
from radial_trap.potential import (
    alpha_eff_um2_per_V,
    build_total_potential_fit,
    evaluate_conditions,
    mathieu_q_per_axis,
    rf_field_V_per_um,
    rf_potential_V,
    total_potential_coefficients,
)
from radial_trap.types import (
    EquilibriumResult,
    LatticeConditions,
    MicromotionResult,
    RadialTrapParams,
    RadialTrapResult,
)

__all__ = [
    "RadialTrapParams",
    "LatticeConditions",
    "MicromotionResult",
    "EquilibriumResult",
    "RadialTrapResult",
    "rf_potential_V",
    "rf_field_V_per_um",
    "alpha_eff_um2_per_V",
    "total_potential_coefficients",
    "build_total_potential_fit",
    "evaluate_conditions",
    "mathieu_q_per_axis",
    "initial_positions",
    "find_radial_equilibrium",
    "micromotion_amplitude",
    "solve_inplane_phonons",
]
