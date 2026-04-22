"""
World coordinates (µm) to pixel grid indices for the camera. Output image shape: (h, l).

**Ion image convention (ImgSimulation):** the first world coordinate ``x_um`` (columns)
is simulation **z**; the second ``y_um`` (rows) is simulation **x** (same as Plotter
``zox``). ``x`` increases with column, ``y`` with row (``imshow`` + ``origin=lower``).
"""
from __future__ import annotations

import numpy as np


def world_um_to_fractional_col_row(
    x_um: float | np.ndarray,
    y_um: float | np.ndarray,
    *,
    x0_um: float,
    y0_um: float,
    pixel_um: float,
    l: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map physical coordinates in µm to fractional (column, row) in [0, l-1] x [0, h-1].

    In ImgSimulation, *x_um* is simulation **z** (columns) and *y_um* is simulation **x**
    (rows). Column i center: x0 + (i - (l-1)/2) * pixel_um; row j center:
    y0 + (j - (h-1)/2) * pixel_um.
    """
    xc = (x_um - x0_um) / pixel_um + 0.5 * (l - 1)
    yc = (y_um - y0_um) / pixel_um + 0.5 * (h - 1)
    return np.asarray(xc, dtype=np.float64), np.asarray(yc, dtype=np.float64)


def bilinear_splat2d(
    image: np.ndarray,
    col_f: float | np.ndarray,
    row_f: float | np.ndarray,
    weight: float | np.ndarray,
) -> None:
    """
    Add *weight* to a 2D float image (h, l) at fractional (col, row) using bilinear weights.
    In-place. Pixels outside the grid receive no partial contribution.
    """
    h, l = image.shape
    c = np.atleast_1d(np.asarray(col_f, dtype=np.float64).ravel())
    r = np.atleast_1d(np.asarray(row_f, dtype=np.float64).ravel())
    w = np.atleast_1d(np.asarray(weight, dtype=np.float64).ravel())
    n = c.size
    if r.size != n or w.size != n:
        raise ValueError("col_f, row_f, weight must have the same shape")
    for idx in range(n):
        cc, rr, wt = c[idx], r[idx], w[idx]
        if not (np.isfinite(cc) and np.isfinite(rr) and np.isfinite(wt)) or wt == 0.0:
            continue
        i0 = int(np.floor(cc))
        j0 = int(np.floor(rr))
        if i0 < -1 or j0 < -1 or i0 > l or j0 > h:
            continue
        dc, dr = cc - i0, rr - j0
        t00 = (1.0 - dc) * (1.0 - dr) * wt
        t10 = dc * (1.0 - dr) * wt
        t01 = (1.0 - dc) * dr * wt
        t11 = dc * dr * wt
        if 0 <= i0 < l and 0 <= j0 < h:
            image[j0, i0] += t00
        if 0 <= i0 + 1 < l and 0 <= j0 < h:
            image[j0, i0 + 1] += t10
        if 0 <= i0 < l and 0 <= j0 + 1 < h:
            image[j0 + 1, i0] += t01
        if 0 <= i0 + 1 < l and 0 <= j0 + 1 < h:
            image[j0 + 1, i0 + 1] += t11
