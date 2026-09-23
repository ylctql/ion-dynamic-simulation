"""radial_trap — 2D 径向多项式势阱模块。

以 Laplace 多项式基解析指定 RF+DC 势（A,B,D,E,F），求解径向平面 (x,y)
平衡构型、各离子解析 micromotion 幅度与晶格成形条件；可选面内声子、
势场分布图、阱频反算与滑块交互 UI。

运行: python -m radial_trap --N 10   （详见 docs/radial_trap.md）
"""
from radial_trap.fitting import FitABResult, fit_rf_ab, load_radial_grid_csv
from radial_trap.lattice import (
    find_radial_equilibrium,
    initial_positions,
    micromotion_amplitude,
    solve_inplane_phonons,
)
from radial_trap.plots import plot_lattice, plot_potential_maps
from radial_trap.potential import (
    alpha_eff_um2_per_V,
    augment_fit_with_axial_confinement,
    bias_potential_V,
    build_total_potential_fit,
    dc_potential_V,
    evaluate_conditions,
    invert_trap_freqs_to_params,
    mathieu_q_per_axis,
    pseudopotential_V,
    rf_field_V_per_um,
    rf_potential_V,
    total_potential_coefficients,
    total_potential_V,
    trap_freq_MHz_to_k2,
)
from radial_trap.types import (
    EquilibriumResult,
    LatticeConditions,
    MicromotionResult,
    RadialTrapParams,
    RadialTrapResult,
)
from radial_trap.ui import RadialTrapUI

__all__ = [
    "RadialTrapParams",
    "LatticeConditions",
    "MicromotionResult",
    "EquilibriumResult",
    "RadialTrapResult",
    "RadialTrapUI",
    "FitABResult",
    "fit_rf_ab",
    "load_radial_grid_csv",
    "rf_potential_V",
    "rf_field_V_per_um",
    "alpha_eff_um2_per_V",
    "total_potential_coefficients",
    "build_total_potential_fit",
    "augment_fit_with_axial_confinement",
    "evaluate_conditions",
    "mathieu_q_per_axis",
    "pseudopotential_V",
    "bias_potential_V",
    "dc_potential_V",
    "total_potential_V",
    "trap_freq_MHz_to_k2",
    "invert_trap_freqs_to_params",
    "initial_positions",
    "find_radial_equilibrium",
    "micromotion_amplitude",
    "solve_inplane_phonons",
    "plot_lattice",
    "plot_potential_maps",
]
