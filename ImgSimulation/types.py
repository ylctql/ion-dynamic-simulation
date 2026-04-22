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
        Field of view in pixels along x.
    h : int
        Field of view in pixels along y.
    x0_um : float
        World x of image center (µm), default 0.
    y0_um : float
        World y of image center (µm), default 0.
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
    Gaussian illumination in the ion plane. Intensity is I at peak (dimensionless, user scale).

    w_um is the 1/e^2 radius of the beam (circular, symmetric).
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
    """Time window and number of integrator steps along [t_start, t_start + t_cum]."""

    t_start_us: float
    t_cum_us: float
    n_step: int | None = None

    def __post_init__(self) -> None:
        if self.t_cum_us <= 0:
            raise ValueError("t_cum_us must be positive")
        if self.n_step is not None and self.n_step < 1:
            raise ValueError("n_step must be >= 1 if given")


def default_n_step(t_cum_us: float, dt_si: float) -> int:
    """
    Heuristic default: enough substeps to resolve the exposure window relative to one dt.
    """
    t_cum_dim = t_cum_us * 1e-6 / dt_si
    return max(32, int(math.ceil(32.0 * max(t_cum_dim, 1e-9))))
