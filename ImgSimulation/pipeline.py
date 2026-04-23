"""
End-to-end single frame: dynamics → exposure map → PSF → noise.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import setup_path

from FieldConfiguration.constants import Config

if TYPE_CHECKING:
    from Interface.cli import ParsedRun

from ComputeKernel.backend import (
    _to_ionsim_format,
    _wrap_force,
    get_actual_device,
    ionsim_calculate_trajectory,
)
from .geometry import world_um_to_fractional_col_row
from .integrate import integrate_exposure_xy_um
from .normalize import NormalizeMode, normalize_image
from .noise_model import add_noise
from .psf import apply_gaussian_psf
from .types import (
    BeamParams,
    CameraParams,
    IntegrationParams,
    NoiseParams,
    default_n_step,
)


def _dimless_r_to_ion_image_plane_um(r: np.ndarray, dl: float) -> np.ndarray:
    """
    r: (N,3) dimensionless -> (N,2) in µm on the 2D ion crystal plane.

    Matches Plotter ``zox``: image column (horizontal) = simulation **z** (``r[:,2]``);
    image row (vertical) = simulation **x** (``r[:,0]``). Simulation **y** is not projected.
    """
    s = float(dl) * 1e6
    return np.column_stack((r[:, 2] * s, r[:, 0] * s))


def _run_trajectory(
    r0: np.ndarray,
    v0: np.ndarray,
    charge: np.ndarray,
    mass: np.ndarray,
    t_start: float,
    t_end: float,
    n_step: int,
    force: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    use_cuda: bool,
    calc_method: Literal["RK4", "VV"],
    use_zero_force: bool,
):
    _root = Path(__file__).resolve().parent.parent
    setup_path.ensure_build_in_path(_root)

    r_in, v_in, q_in, m_in = _to_ionsim_format(r0, v0, charge, mass)
    f = _wrap_force(force)
    return ionsim_calculate_trajectory(
        r_in,
        v_in,
        q_in,
        m_in,
        step=n_step,
        time_start=t_start,
        time_end=t_end,
        force=f,
        use_cuda=use_cuda,
        calc_method=calc_method,
        use_zero_force=use_zero_force,
    )


def render_single_frame(
    config: Config,
    force: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
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
    normalize_mode: NormalizeMode = "none",
    normalize_eps: float = 1e-12,
    normalize_q_low: float = 2.0,
    normalize_q_high: float = 98.0,
    normalize_q_scale: float = 99.0,
    return_mean_plane_px: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Simulate one CCD/CMOS-style image (shape (h, l)).

    State ``r0, v0`` is at simulation time 0. If ``t_start_us > 0``, a preliminary
    integration to ``t_start`` is performed (so cost grows with t_start; consider
    warm-starting from saved state in future work).

    Parameters
    ----------
    config : Config
        Provides ``dt`` (s) and ``dl`` (m) for time/length conversion.
    psf_sigma_px : float
        Gaussian PSF standard deviation in **pixels** (x and y).
    apply_sensor_noise : bool, optional
        If False, return the blurred exposure map (float) without Poisson/readout/background
        (useful for unit tests; physical pipelines keep True).
    normalize_mode : str, optional
        One of ``"none"``, ``"max"``, ``"minmax"``. Per-frame scaling **after** noise
        (or after PSF if ``apply_sensor_noise`` is False). Default ``"none"`` keeps
        physical count scale; use ``"max"`` or ``"minmax"`` for display or ML input ranges.
    normalize_eps : float, optional
        Numerical floor for near-flat images.
    normalize_q_low, normalize_q_high : float, optional
        Percentiles (for ``minmax``) used as robust bounds; see :func:`.normalize.normalize_image`.
    normalize_q_scale : float, optional
        Percentile divisor (for ``max`` mode).
    return_mean_plane_px : bool, optional
        If True, return ``(image, mean_plane_px)`` where ``mean_plane_px`` is (N, 2) with
        columns ``[col, row]`` as **fractional pixel indices** (subpixel), consistent
        with :func:`geometry.world_um_to_fractional_col_row` and image shape ``(h, l)``
        (column along simulation **z**, row along simulation **x**). The mean is over
        all trajectory samples in the **exposure** window. Default False returns only
        ``image``.
    """
    dt_si = config.dt
    t_start_dim = integ.t_start_us * 1e-6 / dt_si
    t_cum_dim = integ.t_cum_us * 1e-6 / dt_si
    t_end_leg2 = t_start_dim + t_cum_dim
    if integ.n_step_per_us is not None:
        n_step = max(1, int(round(integ.n_step_per_us * integ.t_cum_us)))
    else:
        n_step = integ.n_step if integ.n_step is not None else default_n_step(integ.t_cum_us, dt_si)
        n_step = max(1, n_step)

    if t_start_dim < 0:
        raise ValueError("t_start must be non-negative in simulation time")
    if t_cum_dim <= 0:
        raise ValueError("t_cum must lead to a positive duration in dimless time")

    if t_start_dim <= 0.0:
        r_list, v_list = _run_trajectory(
            r0, v0, charge, mass,
            0.0, t_end_leg2, n_step, force, use_cuda, calc_method, use_zero_force
        )
    else:
        if integ.n_step_pre is not None:
            n_step_pre_leg = max(1, int(integ.n_step_pre))
        elif integ.n_step_per_us is not None:
            n_step_pre_leg = max(32, int(round(integ.n_step_per_us * integ.t_start_us)))
        else:
            n_step_pre_leg = max(32, int(np.ceil(32.0 * max(t_start_dim, 1e-9))))
        r_list, v_list = _run_trajectory(
            r0, v0, charge, mass,
            0.0, t_start_dim, n_step_pre_leg, force, use_cuda, calc_method, use_zero_force
        )
        r0_b = np.asarray(r_list[-1], dtype=np.float64, order="F")
        v0_b = np.asarray(v_list[-1], dtype=np.float64, order="F")
        r_list, v_list = _run_trajectory(
            r0_b, v0_b, charge, mass,
            t_start_dim, t_end_leg2, n_step, force, use_cuda, calc_method, use_zero_force
        )

    t_total_dim = t_cum_dim
    dt_dim = t_total_dim / n_step
    dt_real_s = float(dt_dim * dt_si)

    r_plane_list: list[np.ndarray] = []
    if return_mean_plane_px:
        r_stack = np.stack([np.asarray(r, dtype=np.float64) for r in r_list], axis=0)
        r_mean_dimless = np.mean(r_stack, axis=0)
        mean_plane_um = _dimless_r_to_ion_image_plane_um(r_mean_dimless, config.dl)
        col, row = world_um_to_fractional_col_row(
            mean_plane_um[:, 0],
            mean_plane_um[:, 1],
            x0_um=camera.x0_um,
            y0_um=camera.y0_um,
            pixel_um=camera.pixel_um,
            l=camera.l,
            h=camera.h,
        )
        mean_plane_px = np.column_stack((col, row))
        for ti in range(r_stack.shape[0]):
            r_plane_list.append(
                _dimless_r_to_ion_image_plane_um(r_stack[ti], config.dl)
            )
    else:
        for r in r_list:
            r = np.asarray(r, dtype=np.float64)
            r_plane_list.append(_dimless_r_to_ion_image_plane_um(r, config.dl))

    exposure = integrate_exposure_xy_um(r_plane_list, camera, beam, dt_real_s)
    blurred = apply_gaussian_psf(exposure, psf_sigma_px)
    if not apply_sensor_noise:
        out = blurred
    else:
        out = add_noise(blurred, noise)
    img = normalize_image(
        out,
        normalize_mode,
        eps=normalize_eps,
        q_low=normalize_q_low,
        q_high=normalize_q_high,
        q_scale=normalize_q_scale,
    )
    if return_mean_plane_px:
        return img, mean_plane_px
    return img


def _initial_rv_from_parsed(parsed: "ParsedRun", cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    p = parsed.params
    r0 = p.get_r0()
    if p.bilayer:
        from Interface.bilayer_init import apply_bilayer_y_split

        r0 = apply_bilayer_y_split(r0, p.N, p.bilayer_y0_um, cfg.dl)
    v0 = p.get_v0()
    return (
        np.asarray(r0, dtype=np.float64, order="C"),
        np.asarray(v0, dtype=np.float64, order="C"),
    )


def _import_main_module(project_root: Path) -> Any:
    """Load project ``main`` without requiring ``import main`` on sys.path as top-level."""
    main_path = project_root / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(f"expected main.py at {main_path}")
    spec = importlib.util.spec_from_file_location("ism_main_imgsim", main_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_single_frame_from_parsed(
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
    normalize_mode: NormalizeMode = "none",
    normalize_eps: float = 1e-12,
    normalize_q_low: float = 2.0,
    normalize_q_high: float = 98.0,
    normalize_q_scale: float = 99.0,
    return_mean_plane_px: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    High-level entry: build ``force`` like ``main`` and run a single image.

    project_root: repository root (directory containing main.py) for ``_build_force`` CSV resolution.
    """
    main_mod = _import_main_module(project_root)

    cfg = parsed.config
    r0, v0 = _initial_rv_from_parsed(parsed, cfg)
    charge = np.asarray(parsed.params.q, dtype=np.float64)
    mass = np.asarray(parsed.params.m, dtype=np.float64)
    force = main_mod._build_force(
        parsed.field_settings,
        cfg,
        charge,
        project_root,
        smooth_axes=parsed.smooth_axes,
        smooth_sg=parsed.smooth_sg,
    )
    dev = get_actual_device(parsed.params.device)
    use_cuda = dev == "cuda"
    return render_single_frame(
        cfg,
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
        calc_method=cast(
            Literal["RK4", "VV"], parsed.params.calc_method
        ),
        use_zero_force=use_zero_force,
        apply_sensor_noise=apply_sensor_noise,
        normalize_mode=normalize_mode,
        normalize_eps=normalize_eps,
        normalize_q_low=normalize_q_low,
        normalize_q_high=normalize_q_high,
        normalize_q_scale=normalize_q_scale,
        return_mean_plane_px=return_mean_plane_px,
    )
