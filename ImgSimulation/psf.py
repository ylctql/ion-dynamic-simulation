"""
Stationary normalized Gaussian PSF (separable, isotropic in pixel space).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def apply_gaussian_psf(image: np.ndarray, psf_sigma_px: float) -> np.ndarray:
    """
    Convolve *image* with a normalized Gaussian kernel of sigma *psf_sigma_px* (pixels).

    Uses ``scipy.ndimage.gaussian_filter`` with constant padding 0 at edges.
    """
    if psf_sigma_px <= 0:
        return np.asarray(image, dtype=np.float64, copy=True)
    return ndimage.gaussian_filter(
        np.asarray(image, dtype=np.float64),
        sigma=psf_sigma_px,
        order=0,
        mode="constant",
        cval=0.0,
    )
