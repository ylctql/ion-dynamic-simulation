"""
Optional post-processing: per-image value scaling (after noise, before return / display).

``max`` and ``minmax`` use **percentiles** so single hot pixels (from shot/readout noise)
do not dominate the scale the way global min/max or max do.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

NormalizeMode = Literal["none", "max", "minmax"]


def _validate_quantiles(
    q_low: float, q_high: float, q_scale: float, *, for_minmax: bool
) -> None:
    for name, q in (
        ("q_low", q_low),
        ("q_high", q_high),
        ("q_scale", q_scale),
    ):
        if not (0.0 < q < 100.0):
            raise ValueError(f"{name} must be in (0, 100), got {q}")
    if for_minmax and q_low >= q_high:
        raise ValueError("q_low must be < q_high for minmax mode")


def normalize_image(
    image: np.ndarray,
    mode: NormalizeMode = "none",
    *,
    eps: float = 1e-12,
    q_low: float = 2.0,
    q_high: float = 98.0,
    q_scale: float = 99.0,
) -> np.ndarray:
    """
    Per-frame scaling of a 2D array.

    Parameters
    ----------
    mode
        * ``"none"`` — return a float64 copy (no scaling).
        * ``"max"`` — divide by the ``q_scale``-th **percentile** of ``image`` (default 99).
          Less sensitive to a single outlier than ``max`` of all pixels.
        * ``"minmax"`` — map pixel values to ``[0, 1]`` using the ``[q_low, q_high]``
          percentiles as robust bounds: clip to ``[P(q_low), P(q_high)]`` then
          ``(x - P_lo) / (P_hi - P_lo)``. Common defaults: 2 and 98.
    q_low, q_high
        Percentiles (0–100, exclusive) for **minmax**; typical 2 and 98.
    q_scale
        Percentile used as the scale in **max** mode (e.g. 99).
    eps
        Small value to avoid division by zero.
    """
    a = np.asarray(image, dtype=np.float64, copy=True)
    if a.ndim != 2:
        raise ValueError("normalize_image expects a 2D array")
    if mode == "none":
        return a
    if mode == "max":
        _validate_quantiles(q_low, q_high, q_scale, for_minmax=False)
        scale = float(np.percentile(a, q_scale))
        if scale <= eps:
            return np.zeros_like(a)
        return a / scale
    if mode == "minmax":
        _validate_quantiles(q_low, q_high, q_scale, for_minmax=True)
        lo = float(np.percentile(a, q_low))
        hi = float(np.percentile(a, q_high))
        span = hi - lo
        if span <= eps:
            return np.zeros_like(a)
        a = np.clip(a, lo, hi)
        return (a - lo) / (span + eps)
    raise ValueError(f"unknown normalize mode: {mode!r}")
