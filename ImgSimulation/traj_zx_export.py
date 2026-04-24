"""
Batch-export ion images as ``.npy`` from plane-trajectory ``.npz`` files (e.g. under ``traj_zx/``),
using an imaging-only JSON (camera, beam, noise, imaging, optional ``batch``).

For each trajectory file, computes the exposure map once, then applies per-frame PSF, noise,
and normalization — same optimization idea as :func:`ImgSimulation.batch.render_batch`.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from .json_config import (
    load_imaging_json,
    noise_params_list_from_imaging_bundle,
    psf_sigma_px_list_from_imaging_bundle,
)
from .normalize import NormalizeMode
from .pipeline import integrate_exposure_xy_um, render_from_exposure
from .plane_trajectory_io import load_plane_trajectory_npz
from .types import NoiseParams


def default_traj_zx_paths(
    package_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """
    Return ``(traj_dir, imgs_root, meta_root)`` under the ``ImgSimulation`` package directory.
    """
    pkg = Path(package_dir).resolve() if package_dir is not None else Path(__file__).resolve().parent
    return pkg / "traj_zx", pkg / "Imgs", pkg / "meta"


def _noise_params_to_jsonable(n: NoiseParams) -> dict[str, Any]:
    d = asdict(n)
    return {k: (v if v is not None else None) for k, v in d.items()}


def export_ion_npy_from_traj_dir(
    imaging_json: str | Path,
    *,
    traj_dir: str | Path,
    imgs_root: str | Path,
    meta_root: str | Path,
    pattern: str = "*.npz",
    project_root: str | Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """
    For each ``.npz`` under ``traj_dir`` matching ``pattern``, write under ``imgs_root/<stem>/``
    files ``<stem>_0001.npy`` … and under ``meta_root/<stem>/`` matching ``.json`` meta files.

    Frame count matches :func:`noise_params_list_from_imaging_bundle` (one frame if no ``batch``,
    else ``len(batch.seeds)``).

    Parameters
    ----------
    imaging_json
        Imaging-only JSON path (see ``configs/example_imaging.json``).
    traj_dir, imgs_root, meta_root
        Directories; trajectory inputs and output trees.
    pattern
        Glob pattern relative to ``traj_dir`` (default all ``.npz``).
    project_root
        Passed to :func:`load_imaging_json`.
    dry_run
        If ``True``, do not write files or create directories; still returns the sorted list of
        trajectory paths that would be processed.

    Returns
    -------
    list[Path]
        Sorted trajectory ``.npz`` paths processed (or that would be processed when ``dry_run``).
    """
    img = load_imaging_json(imaging_json, project_root=project_root)
    noises = noise_params_list_from_imaging_bundle(img)
    psf_sigmas = psf_sigma_px_list_from_imaging_bundle(img)
    n_frames = len(noises)
    if n_frames != len(psf_sigmas):
        raise ValueError("internal: noise list and PSF list length mismatch")

    traj_base = Path(traj_dir).resolve()
    out_imgs = Path(imgs_root).resolve()
    out_meta = Path(meta_root).resolve()
    imaging_path = Path(imaging_json).resolve()

    traj_paths = sorted(traj_base.glob(pattern))
    traj_paths = [p for p in traj_paths if p.is_file() and p.suffix.lower() == ".npz"]

    norm_mode = cast(NormalizeMode, img.normalize_mode)

    for traj_path in traj_paths:
        stem = traj_path.stem
        img_sub = out_imgs / stem
        meta_sub = out_meta / stem
        if not dry_run:
            img_sub.mkdir(parents=True, exist_ok=True)
            meta_sub.mkdir(parents=True, exist_ok=True)

        if dry_run:
            continue

        rec = load_plane_trajectory_npz(traj_path)
        exposure = integrate_exposure_xy_um(
            rec.xy_stack,
            img.camera,
            img.beam,
            rec.dt_real_s,
        )

        for k in range(n_frames):
            tag = f"{stem}_{k + 1:04d}"
            arr = render_from_exposure(
                exposure,
                psf_sigmas[k],
                noises[k],
                apply_sensor_noise=img.apply_sensor_noise,
                normalize_mode=norm_mode,
                normalize_eps=img.normalize_eps,
                normalize_q_low=img.normalize_q_low,
                normalize_q_high=img.normalize_q_high,
                normalize_q_scale=img.normalize_q_scale,
            )
            npy_path = img_sub / f"{tag}.npy"
            np.save(npy_path, arr)

            meta: dict[str, Any] = {
                "version": 1,
                "noise": _noise_params_to_jsonable(noises[k]),
                "apply_sensor_noise": img.apply_sensor_noise,
                "psf_sigma_px": psf_sigmas[k],
                "frame_index": k + 1,
                "imaging_json": str(imaging_path),
                "traj_npz": str(traj_path.resolve()),
                "image_shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
            meta_path = meta_sub / f"{tag}.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return traj_paths
