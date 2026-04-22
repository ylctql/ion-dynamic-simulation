"""
Gaussian beam illumination (1/e^2 radius) in the ion plane.
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
    Circular Gaussian; *w_um* is the 1/e^2 radius:
    I(x,y) = I * exp(-2 * ((x-xb)^2 + (y-yb)^2) / w_um^2).
    """
    dx = x_um - xb_um
    dy = y_um - yb_um
    r2 = dx * dx + dy * dy
    return I * np.exp(-2.0 * r2 / (w_um * w_um))
