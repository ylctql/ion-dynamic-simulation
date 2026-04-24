"""
Save/load ion trajectories projected onto the imaging plane (µm) for dynamics–imaging decoupling.

Convention: ``xy_stack[t, i, :]`` is ``(z_um, x_um)`` (column = simulation **z**, row = simulation **x**, zox),
matching :func:`integrate_exposure_xy_um`.

When saving an NPZ via :func:`save_plane_trajectory_npz`, by default also writes
``ImgSimulation/pos_zx/<stem>.npy`` with shape ``(N, 2)``: time-averaged plane positions (µm).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from FieldConfiguration.constants import Config

from .json_config import _repo_root, _resolve_existing_path
from .normalize import NormalizeMode
from .pipeline import (
    compute_exposure_trajectory,
    render_from_r_plane_lists,
    r_list_to_r_plane_lists,
)
from .types import BeamParams, CameraParams, IntegrationParams, NoiseParams

SCHEMA_VERSION = 1
DEFAULT_CONVENTION = "zox_col_z_row_x_um"


def _img_sim_package_dir() -> Path:
    """Directory of the ``ImgSimulation`` package (parent of this module)."""
    return Path(__file__).resolve().parent


def build_dynamics_provenance_meta(
    dynamics_json_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable dict for NPZ ``meta_json`` with full **dynamics JSON** snapshot,
    resolved **field_config** file content (not just the path), and **field_csv** naming / paths.

    Use when saving plane trajectories so runs stay reproducible even if configs on disk change later.
    """
    path = Path(dynamics_json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dynamics JSON not found: {path}")
    project_root_p = Path(project_root).resolve() if project_root is not None else _repo_root()
    json_dir = path.parent
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise TypeError("dynamics JSON root must be an object")

    out: dict[str, Any] = {
        "dynamics_json_path": str(path),
        "dynamics_json_content": root,
    }

    paths = root.get("paths")
    paths_dict = paths if isinstance(paths, dict) else {}
    field_setup: dict[str, Any] = {}

    field_cfg = paths_dict.get("field_config")
    if isinstance(field_cfg, str) and field_cfg.strip():
        resolved_fc = _resolve_existing_path(field_cfg, json_dir, project_root_p)
        field_setup["field_config_path_as_in_json"] = field_cfg
        field_setup["field_config_resolved_path"] = str(resolved_fc)
        if resolved_fc.is_file():
            raw_fc = resolved_fc.read_text(encoding="utf-8")
            try:
                field_setup["field_config_content"] = json.loads(raw_fc)
            except json.JSONDecodeError:
                field_setup["field_config_content"] = raw_fc
                field_setup["field_config_parse_note"] = "not valid JSON; stored as UTF-8 string"
        else:
            field_setup["field_config_content"] = None
            field_setup["field_config_read_error"] = f"file not found at resolved path {resolved_fc}"

    csv_raw = paths_dict.get("field_csv")
    if isinstance(csv_raw, str) and csv_raw.strip():
        resolved_csv = _resolve_existing_path(csv_raw, json_dir, project_root_p)
        field_setup["field_csv"] = {
            "path_as_in_json": csv_raw,
            "resolved_path": str(resolved_csv),
            "basename": Path(csv_raw).name,
        }
    elif csv_raw is not None:
        field_setup["field_csv"] = {"path_as_in_json": csv_raw}

    if field_setup:
        out["field_setup"] = field_setup
    return out


@dataclass(frozen=True)
class PlaneTrajectoryRecord:
    """In-memory plane trajectory as loaded from :func:`load_plane_trajectory_npz`."""

    xy_stack: np.ndarray
    dt_real_s: float
    meta: dict[str, Any]


def stack_to_r_xy_list(xy_stack: np.ndarray) -> list[np.ndarray]:
    """
    Convert ``(T+1, N, 2)`` stack to the list format expected by
    :func:`integrate_exposure_xy_um`.
    """
    a = np.asarray(xy_stack, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 2:
        raise ValueError(f"xy_stack must have shape (T+1, N, 2), got {a.shape}")
    return [a[t].copy() for t in range(a.shape[0])]


def r_xy_list_to_stack(r_xy_list: list[np.ndarray]) -> np.ndarray:
    """Stack list of ``(N, 2)`` arrays to ``(T+1, N, 2)``."""
    if len(r_xy_list) < 2:
        raise ValueError("r_xy_list must have at least 2 time samples")
    return np.stack(
        [np.asarray(x, dtype=np.float64) for x in r_xy_list],
        axis=0,
    )


def _validate_xy_stack(xy_stack: np.ndarray) -> np.ndarray:
    a = np.asarray(xy_stack, dtype=np.float64, order="C")
    if a.ndim != 3 or a.shape[2] != 2:
        raise ValueError(f"xy_stack must have shape (T+1, N, 2), got {a.shape}")
    if a.shape[0] < 2:
        raise ValueError("xy_stack must have at least 2 time steps for trapezoidal exposure")
    if not np.isfinite(a).all():
        raise ValueError("xy_stack must be finite")
    return a


def _meta_for_saved_plane(meta: dict[str, Any] | None, xy_stack_validated: np.ndarray) -> dict[str, Any]:
    """Convention / schema / shape keys written to NPZ and returned from :func:`export_plane_trajectory_from_simulation`."""
    m = dict(meta) if meta else {}
    m.setdefault("convention", DEFAULT_CONVENTION)
    m.setdefault("schema_version", SCHEMA_VERSION)
    m["n_time"] = int(xy_stack_validated.shape[0])
    m["n_ions"] = int(xy_stack_validated.shape[1])
    return m


def save_plane_trajectory_npz(
    path: str | Path,
    xy_stack: np.ndarray,
    dt_real_s: float,
    *,
    meta: dict[str, Any] | None = None,
    write_mean_pos_zx: bool = True,
) -> None:
    """
    Write a compressed NPZ with ``xy_stack``, ``dt_real_s``, schema, convention, and JSON meta.

    When ``write_mean_pos_zx`` is True (default), also writes ``pos_zx/<stem>.npy`` under the
    ``ImgSimulation`` package directory: same ``stem`` as this NPZ path, one ``(N, 2)`` float64
    row per ion — time average of ``xy_stack`` over the exposure window (z_um, x_um), matching
    :data:`DEFAULT_CONVENTION`. Two different NPZ paths with the same ``stem`` will overwrite the
    same ``pos_zx`` file.
    """
    a = _validate_xy_stack(xy_stack)
    dt = float(dt_real_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_real_s must be finite and positive")

    m = _meta_for_saved_plane(meta, a)
    meta_bytes = json.dumps(m, ensure_ascii=False).encode("utf-8")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        xy_stack=a,
        dt_real_s=np.float64(dt),
        schema_version=np.int32(SCHEMA_VERSION),
        convention=np.bytes_(DEFAULT_CONVENTION.encode("utf-8")),
        meta_json=meta_bytes,
    )
    if write_mean_pos_zx:
        pos_dir = _img_sim_package_dir() / "pos_zx"
        pos_dir.mkdir(parents=True, exist_ok=True)
        pos_zx = np.mean(a, axis=0, dtype=np.float64)
        np.save(pos_dir / f"{out.stem}.npy", pos_zx, allow_pickle=False)


def load_plane_trajectory_npz(path: str | Path) -> PlaneTrajectoryRecord:
    """Load :func:`save_plane_trajectory_npz` output."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"plane trajectory NPZ not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        if "xy_stack" not in z or "dt_real_s" not in z:
            raise ValueError(f"invalid plane trajectory file: missing keys in {p}")
        xy_stack = np.asarray(z["xy_stack"], dtype=np.float64)
        dt_arr = z["dt_real_s"]
        dt_real_s = float(np.asarray(dt_arr).reshape(()))

        meta: dict[str, Any]
        if "meta_json" in z:
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
        else:
            meta = {}

        ver = int(z["schema_version"].item()) if "schema_version" in z else meta.get(
            "schema_version", 1
        )
        if int(ver) != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {ver!r}, expected {SCHEMA_VERSION}"
            )

        conv = meta.get("convention")
        if conv is None and "convention" in z:
            conv = bytes(z["convention"]).decode("utf-8")
        if conv is None:
            conv = DEFAULT_CONVENTION
        if conv != DEFAULT_CONVENTION:
            raise ValueError(
                f"unsupported convention {conv!r}, expected {DEFAULT_CONVENTION!r}"
            )

    xy_stack = _validate_xy_stack(xy_stack)
    if not np.isfinite(dt_real_s) or dt_real_s <= 0.0:
        raise ValueError("loaded dt_real_s must be finite and positive")

    meta = {**meta, "convention": conv, "schema_version": SCHEMA_VERSION}
    return PlaneTrajectoryRecord(xy_stack=xy_stack, dt_real_s=dt_real_s, meta=meta)


def export_plane_trajectory_from_simulation(
    path: str | Path,
    config: Config,
    force: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    r0: np.ndarray,
    v0: np.ndarray,
    charge: np.ndarray,
    mass: np.ndarray,
    integ: IntegrationParams,
    *,
    use_cuda: bool = False,
    calc_method: Literal["RK4", "VV"] = "VV",
    use_zero_force: bool = False,
    log_interval_sim_us: float | None = None,
    meta: dict[str, Any] | None = None,
    dynamics_json_path: str | Path | None = None,
    project_root: str | Path | None = None,
    write_mean_pos_zx: bool = True,
) -> PlaneTrajectoryRecord:
    """
    Run exposure-window dynamics, project to the imaging plane (µm), save NPZ, return record.

    ``camera`` is not required for projection when only plane coordinates are stored
    (same as :func:`r_list_to_r_plane_lists` with ``return_mean_plane_px=False``).
    A minimal placeholder :class:`CameraParams` is used to satisfy the API.

    If ``dynamics_json_path`` is set, ``meta`` is merged with :func:`build_dynamics_provenance_meta`
    under the key ``dynamics_provenance`` (caller-supplied ``meta["dynamics_provenance"]`` entries
    are merged on top and override the same keys).
    """
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
        log_interval_sim_us=log_interval_sim_us,
    )
    placeholder_cam = CameraParams(pixel_um=1.0, l=2, h=2)
    _, _, xy_stack = r_list_to_r_plane_lists(
        r_list,
        config,
        placeholder_cam,
        return_mean_plane_px=False,
    )
    extra = dict(meta) if meta else {}
    if dynamics_json_path is not None:
        built = build_dynamics_provenance_meta(
            dynamics_json_path, project_root=project_root
        )
        user_prov = extra.pop("dynamics_provenance", None)
        merged_prov = {**built, **(user_prov if isinstance(user_prov, dict) else {})}
        extra["dynamics_provenance"] = merged_prov
    a = _validate_xy_stack(xy_stack)
    dt = float(dt_real_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_real_s must be finite and positive")
    save_plane_trajectory_npz(path, a, dt, meta=extra, write_mean_pos_zx=write_mean_pos_zx)
    return PlaneTrajectoryRecord(
        xy_stack=a,
        dt_real_s=dt,
        meta=_meta_for_saved_plane(extra, a),
    )


def render_from_plane_trajectory_file(
    path: str | Path,
    camera: CameraParams,
    beam: BeamParams,
    psf_sigma_px: float,
    noise: NoiseParams,
    *,
    apply_sensor_noise: bool = True,
    normalize_mode: NormalizeMode = "none",
    normalize_eps: float = 1e-12,
    normalize_q_low: float = 2.0,
    normalize_q_high: float = 98.0,
    normalize_q_scale: float = 99.0,
) -> np.ndarray:
    """
    Load a plane-trajectory NPZ and run exposure → PSF → noise → normalize.
    """
    rec = load_plane_trajectory_npz(path)
    r_xy_list = stack_to_r_xy_list(rec.xy_stack)
    return render_from_r_plane_lists(
        r_xy_list,
        camera,
        beam,
        rec.dt_real_s,
        psf_sigma_px,
        noise,
        apply_sensor_noise=apply_sensor_noise,
        normalize_mode=normalize_mode,
        normalize_eps=normalize_eps,
        normalize_q_low=normalize_q_low,
        normalize_q_high=normalize_q_high,
        normalize_q_scale=normalize_q_scale,
    )
