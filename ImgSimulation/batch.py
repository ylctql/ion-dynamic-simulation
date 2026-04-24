"""
Multi-image rendering on a single GPU: optional noise-only fast path and guarded multiprocessing.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import numpy as np

from FieldConfiguration.constants import Config

from .integrate import integrate_exposure_xy_um
from .normalize import NormalizeMode, normalize_image
from .noise_model import add_noise
from .pipeline import (
    compute_exposure_trajectory,
    render_from_exposure,
    render_single_frame,
    r_list_to_r_plane_lists,
)
from .profile_util import print_profile_summary, profile_section, profiling_enabled
from .psf import apply_gaussian_psf
from .types import BeamParams, CameraParams, IntegrationParams, NoiseParams


def _noise_signature_except_seed(n: NoiseParams) -> tuple:
    return (
        float(n.shot_factor),
        float(n.shot_scale),
        float(n.readout_factor),
        float(n.readout_sigma),
        float(n.bg_offset),
    )


def can_share_dynamics_for_noise_only(noises: list[NoiseParams]) -> bool:
    """True if every :class:`NoiseParams` differs at most by ``seed`` (same noise model)."""
    if len(noises) <= 1:
        return True
    ref = _noise_signature_except_seed(noises[0])
    return all(_noise_signature_except_seed(x) == ref for x in noises[1:])


def _effective_batch_log_interval(
    log_interval_sim_us: float | None,
    *,
    allow_batch_progress_log: bool,
) -> float | None:
    if not allow_batch_progress_log:
        return None
    return log_interval_sim_us


def _mp_render_single_frame(pack: dict) -> np.ndarray:
    """Top-level worker for :class:`ProcessPoolExecutor` (must be picklable)."""
    return render_single_frame(**pack)


def _psf_sigma_list(psf_sigma_px: float | Sequence[float], n_images: int) -> list[float]:
    if isinstance(psf_sigma_px, (int, float)):
        if float(psf_sigma_px) < 0:
            raise ValueError("psf_sigma_px must be non-negative")
        return [float(psf_sigma_px)] * n_images
    seq = [float(x) for x in psf_sigma_px]
    if len(seq) != n_images:
        raise ValueError(
            f"psf_sigma_px sequence length {len(seq)} must match number of images {n_images}"
        )
    if any(s < 0 for s in seq):
        raise ValueError("psf_sigma_px values must be non-negative")
    return seq


def render_batch(
    config: Config,
    force: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    r0: np.ndarray,
    v0: np.ndarray,
    charge: np.ndarray,
    mass: np.ndarray,
    camera: CameraParams,
    beam: BeamParams,
    integ: IntegrationParams,
    noises: list[NoiseParams],
    *,
    psf_sigma_px: float | Sequence[float],
    use_cuda: bool = False,
    calc_method: Literal["RK4", "VV"] = "VV",
    use_zero_force: bool = False,
    apply_sensor_noise: bool = True,
    normalize_mode: NormalizeMode = "none",
    normalize_eps: float = 1e-12,
    normalize_q_low: float = 2.0,
    normalize_q_high: float = 98.0,
    normalize_q_scale: float = 99.0,
    max_workers: int = 1,
    share_dynamics: bool | None = None,
    log_interval_sim_us: float | None = None,
    allow_batch_progress_log: bool = False,
) -> list[np.ndarray]:
    """
    Render several images that share the same trap configuration and initial state.

    Parameters
    ----------
    noises
        One :class:`NoiseParams` per output image (e.g. different ``seed``).
    max_workers
        ``1`` (default) runs in-process — recommended on a **single** GPU with CUDA.
        ``> 1`` uses a process pool **only** for the slow path (noise model differs);
        requires a **picklable** ``force`` (trap ``force`` callables often are not).
    psf_sigma_px
        Scalar or length-``len(noises)`` sequence. Fast path computes **one** exposure map,
        then applies Gaussian PSF per item (possibly different ``sigma`` each), then noise
        and normalize.
    share_dynamics
        If ``False``, run a full :func:`render_single_frame` per item (slow path). If
        ``None`` (default) or ``True``, run **one** trajectory + exposure, then per item
        PSF (scalar or per-item sigmas), :func:`add_noise`, and :func:`normalize_image`.
        Per-item :class:`NoiseParams` may differ arbitrarily; PSF width may differ per item
        when ``psf_sigma_px`` is a sequence. If ``apply_sensor_noise`` is False, noise
        parameters are ignored but the list length must still match.
    allow_batch_progress_log
        If ``False`` (default), trajectory integration for batches **ignores**
        ``log_interval_sim_us`` (no segmented ionsim legs), for throughput. If ``True``,
        passes ``log_interval_sim_us`` through like :func:`render_single_frame`.
    """
    if not noises:
        return []
    sigmas = _psf_sigma_list(psf_sigma_px, len(noises))
    if share_dynamics is False:
        use_share = False
    else:
        # None or True: one exposure; per-item PSF, noise (optional), normalize.
        use_share = True
    eff_log = _effective_batch_log_interval(
        log_interval_sim_us, allow_batch_progress_log=allow_batch_progress_log
    )
    times: dict[str, float] = {} if profiling_enabled() else {}

    if use_share:
        with profile_section("trajectory", times=times):
            r_list, _v, dt_real_s = compute_exposure_trajectory(
                config,
                force,
                r0,
                v0,
                charge,
                mass,
                integ,
                use_cuda=use_cuda,
                calc_method=calc_method,
                use_zero_force=use_zero_force,
                log_interval_sim_us=None,
            )
        with profile_section("r_plane_projection", times=times):
            _, _, xy_stack = r_list_to_r_plane_lists(
                r_list, config, camera, return_mean_plane_px=False
            )
        with profile_section("exposure", times=times):
            exposure = integrate_exposure_xy_um(xy_stack, camera, beam, dt_real_s)
        out_list: list[np.ndarray] = []
        with profile_section("psf_noise_normalize_loop", times=times):
            if len(set(sigmas)) == 1:
                sig0 = sigmas[0]
                if apply_sensor_noise:
                    blurred = apply_gaussian_psf(exposure, sig0)
                    for noise in noises:
                        out_list.append(
                            normalize_image(
                                add_noise(blurred, noise),
                                normalize_mode,
                                eps=normalize_eps,
                                q_low=normalize_q_low,
                                q_high=normalize_q_high,
                                q_scale=normalize_q_scale,
                            )
                        )
                else:
                    img0 = render_from_exposure(
                        exposure,
                        sig0,
                        noises[0],
                        apply_sensor_noise=False,
                        normalize_mode=normalize_mode,
                        normalize_eps=normalize_eps,
                        normalize_q_low=normalize_q_low,
                        normalize_q_high=normalize_q_high,
                        normalize_q_scale=normalize_q_scale,
                    )
                    out_list = [np.asarray(img0, copy=True) for _ in noises]
            else:
                for sig, noise in zip(sigmas, noises, strict=True):
                    out_list.append(
                        render_from_exposure(
                            exposure,
                            sig,
                            noise,
                            apply_sensor_noise=apply_sensor_noise,
                            normalize_mode=normalize_mode,
                            normalize_eps=normalize_eps,
                            normalize_q_low=normalize_q_low,
                            normalize_q_high=normalize_q_high,
                            normalize_q_scale=normalize_q_scale,
                        )
                    )
        print_profile_summary(times, prefix="[ImgSimulation batch profile]")
        return out_list

    base_kw = dict(
        config=config,
        force=force,
        r0=r0,
        v0=v0,
        charge=charge,
        mass=mass,
        camera=camera,
        beam=beam,
        integ=integ,
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
        log_interval_sim_us=eff_log,
    )

    if max_workers <= 1:
        return [
            render_single_frame(**base_kw, noise=n, psf_sigma_px=sig)
            for n, sig in zip(noises, sigmas, strict=True)
        ]

    try:
        pickle.dumps(force)
    except Exception as e:
        raise ValueError(
            "render_batch: max_workers>1 requires a picklable force callable; "
            "use max_workers=1 for trap/build_force callables, or run noise-only "
            f"batch with share_dynamics-compatible noises. ({e})"
        ) from e

    ctx = mp.get_context("spawn")
    packs = [
        {**base_kw, "noise": n, "psf_sigma_px": sig}
        for n, sig in zip(noises, sigmas, strict=True)
    ]
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        return list(ex.map(_mp_render_single_frame, packs))
