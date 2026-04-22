"""
Gaussian beam illumination (1/e^2 radius) in the imaging plane: *x_um* = sim **z**,
*y_um* = sim **x** (see :mod:`ImgSimulation.geometry`).
"""
from __future__ import annotations

import numpy as np


def beam_intensity(
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    w_um: float,
    xb_um: float,
    yb_um: float,
    I: float,
) -> np.ndarray:
    """
    Circular Gaussian; *w_um* is the 1/e^2 radius in the (z, x) imaging plane:
    I = I * exp(-2 * ((x-xb)^2 + (y-yb)^2) / w_um^2) with *x* = sim z, *y* = sim x.
    """
    dx = x_um - xb_um
    dy = y_um - yb_um
    r2 = dx * dx + dy * dy
    return I * np.exp(-2.0 * r2 / (w_um * w_um))
