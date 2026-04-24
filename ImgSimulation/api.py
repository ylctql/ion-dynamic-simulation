"""
User-facing API: one place to import simulation, visualization, and parameter types.

Typical use::

    from pathlib import Path
    from FieldConfiguration.constants import init_from_config
    from FieldParser.force import _zero_force
    from ImgSimulation.api import run_ion_image, CameraParams, BeamParams, NoiseParams, IntegrationParams
    import numpy as np

    cfg, _ = init_from_config("FieldConfiguration/configs/default.json")
    # ... r0, v0, q, m ...
    img = run_ion_image(
        cfg, _zero_force, r0, v0, q, m,
        camera=CameraParams(pixel_um=0.5, l=64, h=64),
        beam=BeamParams(w_um=20.0),
        noise=NoiseParams(seed=0),
        integ=IntegrationParams(t_start_us=0.0, t_cum_us=1.0),
        psf_sigma_px=1.5,
        show=True,
    )

Closely matches :mod:`ImgSimulation.pipeline` but adds optional display in one call.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from FieldConfiguration.constants import Config

if TYPE_CHECKING:
    from collections.abc import Callable

    from Interface.cli import ParsedRun

# Re-exports: single import surface for users
from .batch import can_share_dynamics_for_noise_only, render_batch
from .json_config import (
    IonDynamicsJsonBundle,
    IonImageBatchConfig,
    IonImageJsonBundle,
    IonImagingJsonBundle,
    export_dynamics_batch_plane_npz,
    load_dynamics_json,
    load_imaging_json,
    load_ion_image_json,
    load_ion_image_merged,
    run_ion_image_from_json_file,
)
from .normalize import NormalizeMode, normalize_image
from .noise_model import add_noise
from .pipeline import (
    render_single_frame,
    render_single_frame_from_parsed,
)
from .plane_trajectory_io import (
    DEFAULT_CONVENTION,
    SCHEMA_VERSION,
    PlaneTrajectoryRecord,
    build_dynamics_provenance_meta,
    export_plane_trajectory_from_simulation,
    load_plane_trajectory_npz,
    render_from_plane_trajectory_file,
    r_xy_list_to_stack,
    save_plane_trajectory_npz,
    stack_to_r_xy_list,
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
    "DEFAULT_CONVENTION",
    "SCHEMA_VERSION",
    "add_noise",
    "export_dynamics_batch_plane_npz",
    "apply_gaussian_psf",
    "BeamParams",
    "build_dynamics_provenance_meta",
    "CameraParams",
    "can_share_dynamics_for_noise_only",
    "export_plane_trajectory_from_simulation",
    "IntegrationParams",
    "IonDynamicsJsonBundle",
    "IonImageBatchConfig",
    "IonImageJsonBundle",
    "IonImagingJsonBundle",
    "load_dynamics_json",
    "load_imaging_json",
    "load_ion_image_json",
    "load_ion_image_merged",
    "load_plane_trajectory_npz",
    "NoiseParams",
    "NormalizeMode",
    "PlaneTrajectoryRecord",
    "default_n_step",
    "normalize_image",
    "render_batch",
    "render_from_plane_trajectory_file",
    "render_single_frame",
    "render_single_frame_from_parsed",
    "r_xy_list_to_stack",
    "run_ion_image",
    "run_ion_image_from_json_file",
    "run_ion_image_from_parsed",
    "save_plane_trajectory_npz",
    "show_ion_frame",
    "stack_to_r_xy_list",
]


def run_ion_image(
    config: Config,
    force: "Callable[[np.ndarray, np.ndarray, float], np.ndarray]",
    r0: np.ndarray,
    v0: np.ndarray,
    charge: np.ndarray,
    mass: np.ndarray,
    camera: CameraParams,
    beam: BeamParams,
    noise: NoiseParams,
    integ: IntegrationParams,
    *,
    psf_sigma_px: float,
    use_cuda: bool = False,
    calc_method: Literal["RK4", "VV"] = "VV",
    use_zero_force: bool = False,
    apply_sensor_noise: bool = True,
    show: bool = False,
    show_block: bool = True,
    figure_path: str | Path | None = None,
    show_title: str = "Ion image (simulation)",
    normalize_mode: NormalizeMode = "none",
    normalize_eps: float = 1e-12,
    normalize_q_low: float = 2.0,
    normalize_q_high: float = 98.0,
    normalize_q_scale: float = 99.0,
    log_interval_sim_us: float | None = None,
) -> np.ndarray:
    """
    Run the full single-frame pipeline (dynamics → exposure → PSF → optional noise) and
    optional per-frame :func:`normalize_image`, then optionally display or save.

    This is a thin wrapper around :func:`render_single_frame` plus
    :func:`show_ion_frame` when ``show`` is True and/or ``figure_path`` is set.

    When ``show`` is True, the figure includes markers at each ion's time-averaged
    position (ground truth in **fractional pixel** ``[col, row]``, mapped to µm on the
    axes). Pure ``figure_path`` saves without those markers (same array as returned).
    """
    mean_plane_px: np.ndarray | None
    if show:
        img, mean_plane_px = render_single_frame(
            config,
            force,
            r0,
            v0,
            charge,
            mass,
            camera,
            beam,
            noise,
            integ,
            psf_sigma_px=psf_sigma_px,
            use_cuda=use_cuda,
            calc_method=calc_method,
            use_zero_force=use_zero_force,
            apply_sensor_noise=apply_sensor_noise,
            normalize_mode=normalize_mode,
            normalize_eps=normalize_eps,
            normalize_q_low=normalize_q_low,
            normalize_q_high=normalize_q_high,
            normalize_q_scale=normalize_q_scale,
            return_mean_plane_px=True,
            log_interval_sim_us=log_interval_sim_us,
        )
    else:
        img = render_single_frame(
            config,
            force,
            r0,
            v0,
            charge,
            mass,
            camera,
            beam,
            noise,
            integ,
            psf_sigma_px=psf_sigma_px,
            use_cuda=use_cuda,
            calc_method=calc_method,
            use_zero_force=use_zero_force,
            apply_sensor_noise=apply_sensor_noise,
            normalize_mode=normalize_mode,
            normalize_eps=normalize_eps,
            normalize_q_low=normalize_q_low,
            normalize_q_high=normalize_q_high,
            normalize_q_scale=normalize_q_scale,
            return_mean_plane_px=False,
            log_interval_sim_us=log_interval_sim_us,
        )
        mean_plane_px = None
    if show or figure_path is not None:
        show_ion_frame(
            img,
            camera=camera,
            title=show_title,
            save_path=figure_path,
            block=bool(show) and show_block,
            equilibrium_positions_px=mean_plane_px if show else None,
        )
    return img


def run_ion_image_from_parsed(
    project_root: Path,
    parsed: "ParsedRun",
    camera: CameraParams,
    beam: BeamParams,
    noise: NoiseParams,
    integ: IntegrationParams,
    *,
    psf_sigma_px: float,
    use_zero_force: bool = False,
    apply_sensor_noise: bool = True,
    show: bool = False,
    show_block: bool = True,
    figure_path: str | Path | None = None,
    show_title: str = "Ion image (simulation)",
    normalize_mode: NormalizeMode = "none",
    normalize_eps: float = 1e-12,
    normalize_q_low: float = 2.0,
    normalize_q_high: float = 98.0,
    normalize_q_scale: float = 99.0,
    log_interval_sim_us: float | None = None,
) -> np.ndarray:
    """
    Like :func:`run_ion_image` but uses ``Interface.cli.parse_and_build`` output
    (:class:`~Interface.cli.ParsedRun`) to build the trap force and initial state, same
    as :func:`render_single_frame_from_parsed`.
    """
    mean_plane_px: np.ndarray | None
    if show:
        img, mean_plane_px = render_single_frame_from_parsed(
            project_root,
            parsed,
            camera,
            beam,
            noise,
            integ,
            psf_sigma_px=psf_sigma_px,
            use_zero_force=use_zero_force,
            apply_sensor_noise=apply_sensor_noise,
            normalize_mode=normalize_mode,
            normalize_eps=normalize_eps,
            normalize_q_low=normalize_q_low,
            normalize_q_high=normalize_q_high,
            normalize_q_scale=normalize_q_scale,
            return_mean_plane_px=True,
            log_interval_sim_us=log_interval_sim_us,
        )
    else:
        img = render_single_frame_from_parsed(
            project_root,
            parsed,
            camera,
            beam,
            noise,
            integ,
            psf_sigma_px=psf_sigma_px,
            use_zero_force=use_zero_force,
            apply_sensor_noise=apply_sensor_noise,
            normalize_mode=normalize_mode,
            normalize_eps=normalize_eps,
            normalize_q_low=normalize_q_low,
            normalize_q_high=normalize_q_high,
            normalize_q_scale=normalize_q_scale,
            return_mean_plane_px=False,
            log_interval_sim_us=log_interval_sim_us,
        )
        mean_plane_px = None
    if show or figure_path is not None:
        show_ion_frame(
            img,
            camera=camera,
            title=show_title,
            save_path=figure_path,
            block=bool(show) and show_block,
            equilibrium_positions_px=mean_plane_px if show else None,
        )
    return img


