"""
Simulate single-frame CCD/CMOS-like ion images from trap dynamics.

Use :mod:`ImgSimulation.api` for a **single** user import that bundles types,
:func:`run_ion_image`, low-level :func:`render_single_frame`, and
:func:`show_ion_frame`.
"""
from __future__ import annotations

from . import api
from .api import (
    IonImageJsonBundle,
    load_ion_image_json,
    run_ion_image,
    run_ion_image_from_json_file,
    run_ion_image_from_parsed,
)
from .normalize import NormalizeMode, normalize_image
from .noise_model import add_noise
from .pipeline import (
    render_single_frame,
    render_single_frame_from_parsed,
)
from .psf import apply_gaussian_psf
from .types import (
    BeamParams,
    CameraParams,
    IntegrationParams,
    NoiseParams,
    default_n_step,
)
from .visualize import show_ion_frame

__all__ = [
    "add_noise",
    "api",
    "apply_gaussian_psf",
    "BeamParams",
    "CameraParams",
    "IntegrationParams",
    "IonImageJsonBundle",
    "NoiseParams",
    "NormalizeMode",
    "default_n_step",
    "normalize_image",
    "load_ion_image_json",
    "render_batch",
    "render_single_frame",
    "render_single_frame_from_parsed",
    "run_ion_image",
    "run_ion_image_from_json_file",
    "run_ion_image_from_parsed",
    "show_ion_frame",
]


def render_batch(*_args, **_kwargs) -> None:
    """
    Placeholder for future multi-frame or dataset export.

    Not implemented in this version; use :func:`render_single_frame` in a loop
    for simple batch generation.
    """
    raise NotImplementedError("render_batch: reserved for a future release")
