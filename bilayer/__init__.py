"""双层离子晶格：沿 y 方向拼接单层构型并计算库仑 Hessian。"""

from bilayer.lattice import (
    BilayerHessianBlockStats,
    bilayer_hessian_block_stats,
    coulomb_hessian_bilayer,
    load_positions_from_npz,
    stack_bilayer_along_y,
)
from bilayer.visualize import plot_bilayer_hessian, plot_hessian_heatmap, subsample_hessian

__all__ = [
    "BilayerHessianBlockStats",
    "bilayer_hessian_block_stats",
    "load_positions_from_npz",
    "stack_bilayer_along_y",
    "coulomb_hessian_bilayer",
    "plot_hessian_heatmap",
    "plot_bilayer_hessian",
    "subsample_hessian",
]
