"""
motion_analysis：基于 continuous_sampling 轨迹的动力学后分析。

micromotion：RF micromotion 幅度与有效调制深度 q_eff 的数值测量。
"""
from __future__ import annotations

from .micromotion import (
    CrossCheck,
    IonAxisResult,
    MicromotionReport,
    TrajectoryData,
    WarmupInfo,
    WindowMM,
    analyze_run,
    compute_micromotion,
    cross_check_q,
    detect_warmup,
    load_continuous_sampling,
    report_to_dict,
)
from .plots import (
    plot_beta_vs_secular,
    plot_ion_timeseries,
    plot_lattice_micromotion,
    plot_qeff_histogram,
    plot_qeff_vs_displacement,
)

__all__ = [
    # dataclasses
    "TrajectoryData",
    "WindowMM",
    "IonAxisResult",
    "MicromotionReport",
    "CrossCheck",
    "WarmupInfo",
    # core API
    "load_continuous_sampling",
    "compute_micromotion",
    "analyze_run",
    "cross_check_q",
    "detect_warmup",
    "report_to_dict",
    # plots
    "plot_ion_timeseries",
    "plot_qeff_histogram",
    "plot_qeff_vs_displacement",
    "plot_beta_vs_secular",
    "plot_lattice_micromotion",
]
