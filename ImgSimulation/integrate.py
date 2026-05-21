"""
Project ion trajectories onto the pixel grid (exposure before PSF).
"""
from __future__ import annotations

import numpy as np

from .geometry import bilinear_splat2d, world_um_to_fractional_col_row
from .illumination import beam_intensity
from .types import BeamParams, CameraParams


def integrate_exposure_xy_um(
    r_xy_list: list[np.ndarray],
    camera: CameraParams,
    beam: BeamParams,
    dt_real_s: float,
) -> np.ndarray:
    """
    Trapezoidal rule in time: for each segment between consecutive trajectory samples,
    splat 0.5 * I(r) * dt at each endpoint (per ion), matching trapezoidal integration.

    Parameters
    ----------
    r_xy_list : list of ndarray
        Each array (N, 2) ion positions in µm on the imaging plane: column axis =
        simulation **z**, row axis = simulation **x** (``zox``-style). Length = n_step + 1.
    dt_real_s : float
        Duration of one integrator sub-step in seconds (uniform).
    """
    if len(r_xy_list) < 2:
        raise ValueError("r_xy_list must have at least 2 samples for trapezoidal integration")
    acc = np.zeros((camera.h, camera.l), dtype=np.float64)
    n_ions = r_xy_list[0].shape[0]
    for k in range(len(r_xy_list) - 1):
        r0 = r_xy_list[k]
        r1 = r_xy_list[k + 1]
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
        for n in range(n_ions):
            bilinear_splat2d(acc, c0[n], r0f[n], w0[n])
            bilinear_splat2d(acc, c1[n], r1f[n], w1[n])
    return acc
