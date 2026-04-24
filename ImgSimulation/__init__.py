"""
Simulate single-frame CCD/CMOS-like ion images from trap dynamics.

Use :mod:`ImgSimulation.api` for a **single** user import that bundles types,
:func:`run_ion_image`, :func:`render_batch`, low-level :func:`render_single_frame`,
:func:`show_ion_frame`, and plane-trajectory / split-JSON helpers (``load_plane_trajectory_npz``,
``export_plane_trajectory_from_simulation``, etc.).
"""
from __future__ import annotations

from . import api
from .api import (
    IonImageBatchConfig,
    IonImageJsonBundle,
    can_share_dynamics_for_noise_only,
    load_ion_image_json,
    render_batch,
    run_ion_image,
    run_ion_image_from_json_file,
    run_ion_image_from_parsed,
)
from .normalize import NormalizeMode, normalize_image
from .noise_model import add_noise
from .pipeline import (
    compute_exposure_map,
    render_from_exposure,
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
    "can_share_dynamics_for_noise_only",
    "compute_exposure_map",
    "IntegrationParams",
    "IonImageBatchConfig",
    "IonImageJsonBundle",
    "load_ion_image_json",
    "NoiseParams",
    "NormalizeMode",
    "default_n_step",
    "normalize_image",
    "render_batch",
    "render_from_exposure",
    "render_single_frame",
    "render_single_frame_from_parsed",
    "run_ion_image",
    "run_ion_image_from_json_file",
    "run_ion_image_from_parsed",
    "show_ion_frame",
]
