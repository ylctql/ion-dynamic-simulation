"""
Project ion trajectories onto the pixel grid (exposure before PSF).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .geometry import bilinear_splat2d, world_um_to_fractional_col_row
from .illumination import beam_intensity
from .types import BeamParams, CameraParams


def integrate_exposure_xy_um(
    r_xy_list: Sequence[np.ndarray] | np.ndarray,
    camera: CameraParams,
    beam: BeamParams,
    dt_real_s: float,
) -> np.ndarray:
    """
    Trapezoidal rule in time: for each segment between consecutive trajectory samples,
    splat 0.5 * I(r) * dt at each endpoint (per ion), matching trapezoidal integration.

    Parameters
    ----------
    r_xy_list
        Either a list of (N, 2) arrays (one per time step, length = n_step + 1) or a single
        contiguous ``(T, N, 2)`` float64 array (e.g. from :func:`r_list_to_r_plane_lists`).
    dt_real_s : float
        Duration of one integrator sub-step in seconds (uniform).
    """
    if isinstance(r_xy_list, np.ndarray):
        xy = np.asarray(r_xy_list, dtype=np.float64, order="C")
        if xy.ndim != 3 or xy.shape[2] != 2:
            raise ValueError(f"r_xy array must have shape (T, N, 2), got {xy.shape}")
    else:
        if len(r_xy_list) < 2:
            raise ValueError("r_xy_list must have at least 2 samples for trapezoidal integration")
        xy = np.stack([np.asarray(x, dtype=np.float64) for x in r_xy_list], axis=0)
    acc = np.zeros((camera.h, camera.l), dtype=np.float64)
    for k in range(xy.shape[0] - 1):
        r0 = xy[k]
        r1 = xy[k + 1]
        if r0.shape != r1.shape:
            raise ValueError("inconsistent r_xy shape across trajectory")
        I0 = beam_intensity(
            r0[:, 0], r0[:, 1], w_um=beam.w_um, xb_um=beam.xb_um, yb_um=beam.yb_um, I=beam.I
        )
        I1 = beam_intensity(
            r1[:, 0], r1[:, 1], w_um=beam.w_um, xb_um=beam.xb_um, yb_um=beam.yb_um, I=beam.I
        )
        w0 = 0.5 * I0 * dt_real_s
        w1 = 0.5 * I1 * dt_real_s
        c0, r0f = world_um_to_fractional_col_row(
            r0[:, 0],
            r0[:, 1],
            x0_um=camera.x0_um,
            y0_um=camera.y0_um,
            pixel_um=camera.pixel_um,
            l=camera.l,
            h=camera.h,
        )
        c1, r1f = world_um_to_fractional_col_row(
            r1[:, 0],
            r1[:, 1],
            x0_um=camera.x0_um,
            y0_um=camera.y0_um,
            pixel_um=camera.pixel_um,
            l=camera.l,
            h=camera.h,
        )
        bilinear_splat2d(acc, c0, r0f, w0)
        bilinear_splat2d(acc, c1, r1f, w1)
    return acc
