"""
Dataclasses for single-frame CCD/CMOS-style image simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraParams:
    """
    Parameters
    ----------
    pixel_um : float
        Square pixel size (µm).
    l : int
        Field of view in pixels along the **horizontal** image axis (simulation **z**).
    h : int
        Field of view in pixels along the **vertical** image axis (simulation **x**).
    x0_um : float
        World coordinate of image center (µm) on the **z** (column) axis, default 0.
    y0_um : float
        World coordinate of image center (µm) on the **x** (row) axis, default 0.
    """

    pixel_um: float
    l: int
    h: int
    x0_um: float = 0.0
    y0_um: float = 0.0

    def __post_init__(self) -> None:
        if self.pixel_um <= 0:
            raise ValueError("pixel_um must be positive")
        if self.l < 1 or self.h < 1:
            raise ValueError("l and h must be at least 1")


@dataclass(frozen=True)
class BeamParams:
    """
    Gaussian illumination in the 2D ion crystal plane. Intensity is I at peak
    (dimensionless, user scale). Axes are simulation **z** (``xb_um``) and **x** (``yb_um``);
    *w_um* is the 1/e^2 radius in that plane (circular, symmetric).
    """

    w_um: float
    xb_um: float = 0.0
    yb_um: float = 0.0
    I: float = 1.0

    def __post_init__(self) -> None:
        if self.w_um <= 0:
            raise ValueError("w_um (1/e^2 radius) must be positive")
        if self.I < 0:
            raise ValueError("I must be non-negative")


@dataclass(frozen=True)
class NoiseParams:
    """
    readout_sigma is the standard deviation of Z ~ N(0, readout_sigma^2) in the *same
    signal units* as the post-PSF image, before the readout_factor scaling. The additive
    readout term is: readout_factor * Z.
    """

    shot_factor: float = 1.0
    shot_scale: float = 1.0
    readout_factor: float = 0.0
    readout_sigma: float = 0.0
    bg_offset: float = 0.0
    seed: int | None = None


@dataclass(frozen=True)
class IntegrationParams:
    """
    Time window and integrator substeps for the exposure leg ``[t_start, t_start + t_cum]``.

    Step count for the exposure window is set **either** by ``n_step`` (total substeps)
    **or** ``n_step_per_us`` (substeps per microsecond of ``t_cum_us``), not both.

    When ``t_start_us > 0``, the preliminary leg ``[0, t_start]`` uses the **same**
    ``n_step_per_us`` as the exposure leg: ``max(32, round(n_step_per_us * t_start_us))``
    substeps, unless ``n_step_pre`` is set (fixed total substeps for that leg only).
    If neither ``n_step_per_us`` nor ``n_step_pre`` applies, a dimensionless-time heuristic
    is used.
    """

    t_start_us: float
    t_cum_us: float
    n_step: int | None = None
    n_step_per_us: float | None = None
    n_step_pre: int | None = None

    def __post_init__(self) -> None:
        if self.t_cum_us <= 0:
            raise ValueError("t_cum_us must be positive")
        if self.n_step is not None and self.n_step < 1:
            raise ValueError("n_step must be >= 1 if given")
        if self.n_step_pre is not None and self.n_step_pre < 1:
            raise ValueError("n_step_pre must be >= 1 if given")
        if self.n_step_per_us is not None and self.n_step_per_us <= 0:
            raise ValueError("n_step_per_us must be positive if given")
        if self.n_step is not None and self.n_step_per_us is not None:
            raise ValueError("set only one of n_step (total) or n_step_per_us, not both")


def default_n_step(t_cum_us: float, dt_si: float) -> int:
    """
    Heuristic default: enough substeps to resolve the exposure window relative to one dt.
    """
    t_cum_dim = t_cum_us * 1e-6 / dt_si
    return max(32, int(math.ceil(32.0 * max(t_cum_dim, 1e-9))))
