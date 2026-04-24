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


def fractional_col_row_to_world_um(
    col: float | np.ndarray,
    row: float | np.ndarray,
    *,
    x0_um: float,
    y0_um: float,
    pixel_um: float,
    l: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse of :func:`world_um_to_fractional_col_row`.

    Returns ``(x_um, y_um)`` in µm: horizontal axis / column direction = simulation **z**,
    vertical / row = simulation **x** (``zox``).
    """
    px = float(pixel_um)
    x_out = (np.asarray(col, dtype=np.float64) - 0.5 * (l - 1)) * px + x0_um
    y_out = (np.asarray(row, dtype=np.float64) - 0.5 * (h - 1)) * px + y0_um
    return x_out, y_out


def bilinear_splat2d(
    image: np.ndarray,
    col_f: float | np.ndarray,
    row_f: float | np.ndarray,
    weight: float | np.ndarray,
) -> None:
    """
    Add *weight* to a 2D float image (h, l) at fractional (col, row) using bilinear weights.
    In-place. Pixels outside the grid receive no partial contribution.

    Vectorized over many points (same semantics as the former per-point loop; accumulation
    order may differ slightly at the float level when many ions share a pixel).
    """
    h, l = image.shape
    c = np.atleast_1d(np.asarray(col_f, dtype=np.float64).ravel())
    r = np.atleast_1d(np.asarray(row_f, dtype=np.float64).ravel())
    w = np.atleast_1d(np.asarray(weight, dtype=np.float64).ravel())
    if r.size != c.size or w.size != c.size:
        raise ValueError("col_f, row_f, weight must have the same shape")
    ok = np.isfinite(c) & np.isfinite(r) & np.isfinite(w) & (w != 0.0)
    c = c[ok]
    r = r[ok]
    w = w[ok]
    if c.size == 0:
        return
    i0 = np.floor(c).astype(np.intp, copy=False)
    j0 = np.floor(r).astype(np.intp, copy=False)
    bad = (i0 < -1) | (j0 < -1) | (i0 > l) | (j0 > h)
    c = c[~bad]
    r = r[~bad]
    w = w[~bad]
    i0 = i0[~bad]
    j0 = j0[~bad]
    if i0.size == 0:
        return
    dc = c - i0.astype(np.float64, copy=False)
    dr = r - j0.astype(np.float64, copy=False)
    t00 = (1.0 - dc) * (1.0 - dr) * w
    t10 = dc * (1.0 - dr) * w
    t01 = (1.0 - dc) * dr * w
    t11 = dc * dr * w
    i1 = i0 + 1
    j1 = j0 + 1
    m00 = (i0 >= 0) & (i0 < l) & (j0 >= 0) & (j0 < h)
    m10 = (i1 >= 0) & (i1 < l) & (j0 >= 0) & (j0 < h)
    m01 = (i0 >= 0) & (i0 < l) & (j1 >= 0) & (j1 < h)
    m11 = (i1 >= 0) & (i1 < l) & (j1 >= 0) & (j1 < h)
    np.add.at(image, (j0[m00], i0[m00]), t00[m00])
    np.add.at(image, (j0[m10], i1[m10]), t10[m10])
    np.add.at(image, (j1[m01], i0[m01]), t01[m01])
    np.add.at(image, (j1[m11], i1[m11]), t11[m11])
