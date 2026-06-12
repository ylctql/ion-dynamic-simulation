"""
电场可视化：静电势、RF 赝势、总电势的空间分布
支持 1D（单坐标）与 2D（热力图/三维）绘图，阱频计算与扫描，势场对称性分析，拉普拉斯调和分解
"""
from __future__ import annotations

from .core import (
    AXIS_INDEX,
    CoordAxis,
    apply_offset_min,
    apply_savgol_smooth,
    build_grid_1d,
    build_grid_2d,
    compute_potentials,
    norm_to_um,
    set_ylim_from_data,
    um_to_norm,
)
from .plots import plot_1d, plot_2d, plot_bilayer, plot_freq_scan_1d, plot_freq_scan_2d
from .trap_freq import (
    compute_freq_scan_1d,
    compute_freq_scan_2d,
    compute_trap_freqs_at_point,
)
from .symmetry import (
    SymmetryReport,
    PotentialSymmetry,
    MirrorSymmetryResult,
    RotationalSymmetryResult,
    PolynomialSymmetryResult,
    ScaledCoeffEntry,
    HessianSymmetryResult,
    compute_symmetry_report,
    compute_potential_symmetry,
)
from .laplace_decompose import (
    LaplaceTerm,
    LaplaceDecompositionResult,
    LaplaceConvergenceResult,
    harmonic_poly_2d,
    fit_laplace_2d,
    eval_laplace_fit,
    laplace_convergence,
    print_laplace_report,
    print_laplace_convergence,
)
from .cli import main

__all__ = [
    "main",
    "apply_savgol_smooth",
    "CoordAxis",
    "AXIS_INDEX",
    "um_to_norm",
    "norm_to_um",
    "compute_potentials",
    "apply_offset_min",
    "set_ylim_from_data",
    "build_grid_1d",
    "build_grid_2d",
    "compute_trap_freqs_at_point",
    "compute_freq_scan_1d",
    "compute_freq_scan_2d",
    "plot_1d",
    "plot_2d",
    "plot_bilayer",
    "plot_freq_scan_1d",
    "plot_freq_scan_2d",
    # symmetry
    "SymmetryReport",
    "PotentialSymmetry",
    "MirrorSymmetryResult",
    "RotationalSymmetryResult",
    "PolynomialSymmetryResult",
    "ScaledCoeffEntry",
    "HessianSymmetryResult",
    "compute_symmetry_report",
    "compute_potential_symmetry",
    # laplace decomposition
    "LaplaceTerm",
    "LaplaceDecompositionResult",
    "LaplaceConvergenceResult",
    "harmonic_poly_2d",
    "fit_laplace_2d",
    "eval_laplace_fit",
    "laplace_convergence",
    "print_laplace_report",
    "print_laplace_convergence",
]
