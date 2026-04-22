"""
Shot (Poisson), readout (Gaussian), and background offset.
"""
from __future__ import annotations

import numpy as np

from .types import NoiseParams


def add_noise(
    signal: np.ndarray,
    p: NoiseParams,
) -> np.ndarray:
    """
    Apply, in order: Poisson-based shot term, readout, background.

    - Shot: ``shot_factor * Poisson( max(0, shot_scale * signal) )`` element-wise
    - Readout: ``readout_factor * N(0, readout_sigma^2)`` i.i.d. per pixel
    - Background: ``+ bg_offset``
    """
    rng = np.random.default_rng(p.seed)
    s = np.asarray(signal, dtype=np.float64)
    lam = np.clip(p.shot_scale * s, 0.0, None)
    shot = p.shot_factor * rng.poisson(lam).astype(np.float64)
    read = p.readout_factor * rng.normal(0.0, p.readout_sigma, size=s.shape)
    return shot + read + p.bg_offset
