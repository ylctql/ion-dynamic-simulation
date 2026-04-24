"""
Load single-frame simulation parameters from a JSON file (see ``configs/example_ion_image.json``).

* ``paths.field_config`` — voltage / constants JSON for :func:`FieldConfiguration.constants.init_from_config` (``dt``, ``dl``, …). **Required.**
* ``paths.field_csv`` — electric field grid CSV (same as main ``--csv``). **Required** when
  ``dynamics.force`` is ``"trap"``; optional for ``"zero"`` (Python trap force is then zero, Coulomb only in ``ionsim``).
* ``trap`` — optional: ``smooth_axes``, ``smooth_sg`` for ``main._build_force`` (same as main simulation).
* ``simulation.log_interval_sim_us`` — optional: print progress about every that many
  **simulation** µs while integrating the trajectory (``null`` or omit to turn off).

``dynamics`` can match :class:`Interface.parameters.Parameters` / ``--init_file``:
``N``, ``init_file`` / ``r0_um``+``v0_m_s`` / legacy dimensionless ``r0``+``v0`` / random init; ``charge``, ``mass``, ``alpha``, ``isotope`` / ``isotope_type``, ``bilayer``.

``integration`` supports ``t_start_us``, ``t_cum_us``, and step control: either ``n_step_per_us``
(substeps per µs of wall time, recommended) or legacy ``n_step`` (total substeps for the exposure
window). Optional ``n_step_pre`` (total substeps for ``[0, t_start]``) overrides the pre-leg count;
otherwise the pre-leg also uses ``n_step_per_us`` (see :class:`IntegrationParams`).

Paths may be **relative to the JSON file** or to the project root (second try).

Optional **batch** object: ``seeds`` (non-empty list of integers), optional ``figure_paths`` (same length),
optional ``noise_overrides`` (same length as ``seeds``: per-frame overrides merged onto the root ``noise`` block),
optional ``dynamics_overrides`` (same length: shallow-merge each object onto the root ``dynamics`` block, then re-resolve ions;
    optional keys ``t_start_us``, ``t_cum_us``, ``n_step``, ``n_step_per_us``, ``n_step_pre`` are merged onto root ``integration`` instead),
optional ``plane_npz_paths`` (same length: per-frame imaging-plane trajectory ``.npz``; relative paths resolve against the dynamics JSON directory),
optional ``psf_sigma_px`` (same length as ``seeds``: per-frame Gaussian PSF sigma in pixels; omit to use root ``imaging.psf_sigma_px`` for all),
``max_workers``, ``share_dynamics``, ``allow_batch_progress_log``, ``profile``. When present,
use :meth:`IonImageJsonBundle.call_run_batch` instead of :meth:`IonImageJsonBundle.call_run_ion_image`.

**Split configs (dynamics / imaging decoupled)** — use :func:`load_dynamics_json` and :func:`load_imaging_json`,
or :func:`load_ion_image_merged` / ``python -m ImgSimulation dyn.json img.json``:

* **Dynamics JSON** top-level keys only: ``version``, ``description``, ``paths``, ``trap``, ``dynamics``,
  ``integration``, ``simulation`` (``use_cuda``, ``calc_method``, ``use_zero_force``, ``log_interval_sim_us`` only),
  optional ``batch`` with only ``dynamics_overrides`` (per-frame ``dynamics`` / ``integration`` patches; length must match imaging ``batch.seeds`` when merged).
* **Imaging JSON** top-level keys only: ``version``, ``description``, ``camera``, ``beam``, ``noise``,
  ``imaging``, ``display``, optional ``batch``, optional ``simulation`` with only ``apply_sensor_noise``.
  Relative ``display.figure_path`` / batch paths resolve against the **imaging** JSON directory.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import numpy as np

from FieldConfiguration.constants import Config, init_from_config

from .json_dynamics import resolve_dynamics_arrays, resolve_force_callable
from .types import BeamParams, CameraParams, IntegrationParams, NoiseParams

JSON_VERSION = 1


@dataclass(frozen=True)
class IonImageBatchConfig:
    """Multi-image settings from JSON ``batch`` (see :func:`load_ion_image_json`)."""

    seeds: tuple[int, ...]
    figure_paths: tuple[str | None, ...] | None = None
    noise_overrides: tuple[dict[str, Any], ...] | None = None
    dynamics_overrides: tuple[dict[str, Any], ...] | None = None
    plane_npz_paths: tuple[str | None, ...] | None = None
    psf_sigma_px: tuple[float, ...] | None = None
    max_workers: int = 1
    share_dynamics: bool | None = None
    allow_batch_progress_log: bool = False
    profile: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_existing_path(s: str, json_dir: Path, project_root: Path) -> Path:
    """Resolve a path to an *existing* file: try JSON directory first, then project root."""
    p = Path(s)
    if p.is_absolute():
        return p
    a = (json_dir / p).resolve()
    if a.is_file():
        return a
    b = (project_root / p).resolve()
    return b


def _resolve_output_path(s: str, json_dir: Path) -> Path:
    """
    Resolve a *write* path (file may not exist). Relative paths are taken relative
    to the JSON file's directory.
    """
    p = Path(s)
    if p.is_absolute():
        return p
    return (json_dir / p).resolve()


@dataclass(frozen=True)
class IonImageJsonBundle:
    """
    Data produced by :func:`load_ion_image_json`, ready to pass to
    :func:`ImgSimulation.api.run_ion_image` via :meth:`call_run_ion_image`.
    """

    config: Config
    force: Any
    r0: np.ndarray
    v0: np.ndarray
    charge: np.ndarray
    mass: np.ndarray
    camera: CameraParams
    beam: BeamParams
    noise: NoiseParams
    integ: IntegrationParams
    psf_sigma_px: float
    use_cuda: bool
    calc_method: Literal["RK4", "VV"]
    use_zero_force: bool
    apply_sensor_noise: bool
    show: bool
    show_block: bool
    figure_path: Path | None
    show_title: str
    project_root: Path
    source_json: Path
    normalize_mode: str
    normalize_eps: float
    normalize_q_low: float
    normalize_q_high: float
    normalize_q_scale: float
    log_interval_sim_us: float | None = None
    batch: IonImageBatchConfig | None = None
    # Raw JSON slices for :meth:`call_run_batch` when ``batch.dynamics_overrides`` is set
    dynamics: dict[str, Any] | None = None
    paths: dict[str, Any] | None = None
    trap: dict[str, Any] | None = None
    # Directory of the dynamics JSON (``init_file`` / paths are resolved here); monolith: same as ``source_json.parent``
    dynamics_json_dir: Path | None = None
    # Path to the dynamics JSON file (monolith: same as ``source_json``); for NPZ provenance
    dynamics_source_json: Path | None = None

    def call_run_ion_image(self) -> np.ndarray:
        from .api import run_ion_image

        fp = str(self.figure_path) if self.figure_path is not None else None
        return run_ion_image(
            self.config,
            self.force,
            self.r0,
            self.v0,
            self.charge,
            self.mass,
            self.camera,
            self.beam,
            self.noise,
            self.integ,
            psf_sigma_px=self.psf_sigma_px,
            use_cuda=self.use_cuda,
            calc_method=self.calc_method,
            use_zero_force=self.use_zero_force,
            apply_sensor_noise=self.apply_sensor_noise,
            show=self.show,
            show_block=self.show_block,
            figure_path=fp,
            show_title=self.show_title,
            normalize_mode=self.normalize_mode,  # type: ignore[arg-type]
            normalize_eps=self.normalize_eps,
            normalize_q_low=self.normalize_q_low,
            normalize_q_high=self.normalize_q_high,
            normalize_q_scale=self.normalize_q_scale,
            log_interval_sim_us=self.log_interval_sim_us,
        )

    def call_run_batch(self) -> list[np.ndarray]:
        """Run :func:`ImgSimulation.batch.render_batch` using ``batch.seeds`` and save PNGs if configured."""
        import os

        from .batch import render_batch
        from .visualize import show_ion_frame

        if self.batch is None:
            raise ValueError("IonImageJsonBundle.batch is None; add a JSON \"batch\" block")

        cfg_batch = self.batch
        noises = _noise_list_for_batch(self.noise, cfg_batch.seeds, cfg_batch.noise_overrides)

        @contextmanager
        def _profile_env() -> Iterator[None]:
            if not cfg_batch.profile:
                yield
                return
            old = os.environ.get("IMG_SIM_PROFILE")
            os.environ["IMG_SIM_PROFILE"] = "1"
            try:
                yield
            finally:
                if old is None:
                    os.environ.pop("IMG_SIM_PROFILE", None)
                else:
                    os.environ["IMG_SIM_PROFILE"] = old

        with _profile_env():
            psf_arg: float | list[float]
            if cfg_batch.psf_sigma_px is not None:
                psf_arg = list(cfg_batch.psf_sigma_px)
            else:
                psf_arg = self.psf_sigma_px
            if cfg_batch.dynamics_overrides is not None:
                images = _ion_bundle_run_batch_dynamics_varying(
                    self,
                    cfg_batch,
                    noises,
                    psf_arg,
                )
            else:
                images = render_batch(
                    self.config,
                    self.force,
                    self.r0,
                    self.v0,
                    self.charge,
                    self.mass,
                    self.camera,
                    self.beam,
                    self.integ,
                    noises,
                    psf_sigma_px=psf_arg,
                    use_cuda=self.use_cuda,
                    calc_method=self.calc_method,
                    use_zero_force=self.use_zero_force,
                    apply_sensor_noise=self.apply_sensor_noise,
                    normalize_mode=self.normalize_mode,  # type: ignore[arg-type]
                    normalize_eps=self.normalize_eps,
                    normalize_q_low=self.normalize_q_low,
                    normalize_q_high=self.normalize_q_high,
                    normalize_q_scale=self.normalize_q_scale,
                    max_workers=cfg_batch.max_workers,
                    share_dynamics=cfg_batch.share_dynamics,
                    log_interval_sim_us=self.log_interval_sim_us,
                    allow_batch_progress_log=cfg_batch.allow_batch_progress_log,
                )
                if _batch_plane_npz_any(cfg_batch.plane_npz_paths):
                    _save_shared_batch_plane_npz(self, cfg_batch)

        json_dir = self.source_json.parent
        paths = cfg_batch.figure_paths
        if paths is not None:
            for img, raw in zip(images, paths, strict=True):
                if raw is None or raw == "":
                    continue
                outp = _resolve_output_path(str(raw), json_dir)
                show_ion_frame(
                    img,
                    camera=self.camera,
                    title=self.show_title,
                    save_path=outp,
                    block=False,
                )
        return images


def _batch_plane_npz_any(paths: tuple[str | None, ...] | None) -> bool:
    if paths is None:
        return False
    return any(x for x in paths if x not in (None, ""))


def _plane_npz_resolve_base(bundle: IonImageJsonBundle) -> Path:
    """Relative ``batch.plane_npz_paths`` entries resolve against the dynamics JSON directory."""
    return bundle.dynamics_json_dir or bundle.source_json.parent


def _save_shared_batch_plane_npz(bundle: IonImageJsonBundle, cfg_batch: IonImageBatchConfig) -> None:
    """
    One integration (in addition to the one inside :func:`ImgSimulation.batch.render_batch`) to
    write shared-dynamics plane NPZ files. Skipped when ``dynamics_overrides`` is used (NPZ is
    saved inside :func:`ImgSimulation.pipeline.render_single_frame` per item).
    """
    from .batch import _effective_batch_log_interval
    from .pipeline import compute_exposure_trajectory, r_list_to_r_plane_lists
    from .plane_trajectory_io import build_dynamics_provenance_meta, save_plane_trajectory_npz
    from .types import CameraParams

    paths = cfg_batch.plane_npz_paths
    if paths is None:
        return
    base = _plane_npz_resolve_base(bundle)
    djson = bundle.dynamics_source_json or bundle.source_json
    eff_log = _effective_batch_log_interval(
        bundle.log_interval_sim_us,
        allow_batch_progress_log=cfg_batch.allow_batch_progress_log,
    )
    r_list, _v, dt_real_s = compute_exposure_trajectory(
        bundle.config,
        bundle.force,
        bundle.r0,
        bundle.v0,
        bundle.charge,
        bundle.mass,
        bundle.integ,
        use_cuda=bundle.use_cuda,
        calc_method=bundle.calc_method,
        use_zero_force=bundle.use_zero_force,
        log_interval_sim_us=eff_log,
    )
    placeholder_cam = CameraParams(pixel_um=1.0, l=2, h=2)
    _, _, xy_stack = r_list_to_r_plane_lists(
        r_list,
        bundle.config,
        placeholder_cam,
        return_mean_plane_px=False,
    )
    for i, raw in enumerate(paths):
        if raw is None or raw == "":
            continue
        outp = _resolve_output_path(str(raw), base)
        meta: dict[str, Any] = {
            "batch_index": i,
            "batch_shared_dynamics": True,
            "dynamics_provenance": build_dynamics_provenance_meta(djson, project_root=bundle.project_root),
        }
        save_plane_trajectory_npz(outp, xy_stack, dt_real_s, meta=meta)


def _ion_bundle_run_batch_dynamics_varying(
    bundle: IonImageJsonBundle,
    cfg_batch: IonImageBatchConfig,
    noises: list[NoiseParams],
    psf_arg: float | list[float],
) -> list[np.ndarray]:
    """
    One full :func:`ImgSimulation.pipeline.render_single_frame` per seed when ``dynamics`` differs per item.
    ``batch.share_dynamics`` is ignored in this path (each item has its own trajectory).
    """
    import multiprocessing as mp
    import pickle
    from concurrent.futures import ProcessPoolExecutor

    from .batch import (
        _effective_batch_log_interval,
        _mp_render_single_frame,
        _psf_sigma_list,
    )
    from .pipeline import render_single_frame

    base_dyn = dict(bundle.dynamics or {})
    ddir = bundle.dynamics_json_dir or bundle.source_json.parent
    ov = cfg_batch.dynamics_overrides
    if ov is None:
        raise ValueError("internal: dynamics_overrides required")
    n = len(cfg_batch.seeds)
    if len(ov) != n:
        raise ValueError("batch.dynamics_overrides must have the same length as batch.seeds")
    if len(noises) != n:
        raise ValueError("internal: noises length must match batch.seeds")
    pnp = cfg_batch.plane_npz_paths
    if pnp is not None and len(pnp) != n:
        raise ValueError("batch.plane_npz_paths must have the same length as batch.seeds")

    eff_log = _effective_batch_log_interval(
        bundle.log_interval_sim_us,
        allow_batch_progress_log=cfg_batch.allow_batch_progress_log,
    )
    sigmas = _psf_sigma_list(psf_arg, n)

    packs: list[dict[str, Any]] = []
    for i in range(n):
        dyn_patch, integ_patch = _split_dynamics_integration_patch(ov[i])
        merged = {**base_dyn, **dyn_patch}
        integ_i = _merge_integration_params(bundle.integ, integ_patch)
        r0, v0, ch, m = resolve_dynamics_arrays(
            merged,
            bundle.config,
            bundle.project_root,
            ddir,
        )
        frc = resolve_force_callable(
            str(merged.get("force", "zero")),
            project_root=bundle.project_root,
            json_dir=ddir,
            paths=cast(dict[str, Any], bundle.paths or {}),
            cfg=bundle.config,
            charge=ch,
            trap=cast(dict[str, Any], bundle.trap or {}),
        )
        pack_i: dict[str, Any] = {
            "config": bundle.config,
            "force": frc,
            "r0": r0,
            "v0": v0,
            "charge": ch,
            "mass": m,
            "camera": bundle.camera,
            "beam": bundle.beam,
            "noise": noises[i],
            "integ": integ_i,
            "psf_sigma_px": sigmas[i],
            "use_cuda": bundle.use_cuda,
            "calc_method": bundle.calc_method,
            "use_zero_force": bundle.use_zero_force,
            "apply_sensor_noise": bundle.apply_sensor_noise,
            "normalize_mode": bundle.normalize_mode,
            "normalize_eps": bundle.normalize_eps,
            "normalize_q_low": bundle.normalize_q_low,
            "normalize_q_high": bundle.normalize_q_high,
            "normalize_q_scale": bundle.normalize_q_scale,
            "return_mean_plane_px": False,
            "log_interval_sim_us": eff_log,
        }
        if pnp is not None and pnp[i] not in (None, ""):
            pack_i["save_plane_npz"] = str(_resolve_output_path(str(pnp[i]), ddir))
            pack_i["plane_npz_dynamics_json_path"] = str(
                bundle.dynamics_source_json or bundle.source_json
            )
            pack_i["plane_npz_project_root"] = bundle.project_root
            pack_i["plane_npz_meta_extra"] = {"batch_index": i, "batch_dynamics_varying": True}
        packs.append(pack_i)

    if cfg_batch.max_workers <= 1:
        return [render_single_frame(**p) for p in packs]

    try:
        pickle.dumps(packs[0]["force"])
    except Exception as e:
        raise ValueError(
            "call_run_batch (dynamics_overrides): max_workers>1 needs a picklable force; "
            f"use max_workers=1 for trap/build_force, or set dynamics.force to zero. ({e})"
        ) from e

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=cfg_batch.max_workers, mp_context=ctx) as ex:
        return list(ex.map(_mp_render_single_frame, packs))


# --- Split JSON: dynamics-only vs imaging-only (see load_dynamics_json / load_imaging_json) ---

_DYNAMICS_JSON_KEYS = frozenset(
    {"version", "description", "paths", "trap", "dynamics", "integration", "simulation", "batch"}
)
_IMAGING_JSON_KEYS = frozenset(
    {
        "version",
        "description",
        "camera",
        "beam",
        "noise",
        "imaging",
        "display",
        "batch",
        "simulation",
    }
)


@dataclass(frozen=True)
class IonDynamicsJsonBundle:
    """Parameters from a **dynamics-only** JSON (trap, ions, integration, device)."""

    config: Config
    force: Any
    r0: np.ndarray
    v0: np.ndarray
    charge: np.ndarray
    mass: np.ndarray
    integ: IntegrationParams
    use_cuda: bool
    calc_method: Literal["RK4", "VV"]
    use_zero_force: bool
    log_interval_sim_us: float | None
    project_root: Path
    source_json: Path
    dynamics: dict[str, Any]
    paths: dict[str, Any]
    trap: dict[str, Any]
    # From dynamics JSON ``batch`` only; merge with imaging ``batch`` in :func:`load_ion_image_merged`
    dynamics_batch_overrides: tuple[dict[str, Any], ...] | None
    dynamics_batch_plane_npz_paths: tuple[str | None, ...] | None


@dataclass(frozen=True)
class IonImagingJsonBundle:
    """Parameters from an **imaging-only** JSON (camera, beam, PSF, noise, display, batch)."""

    camera: CameraParams
    beam: BeamParams
    noise: NoiseParams
    psf_sigma_px: float
    apply_sensor_noise: bool
    show: bool
    show_block: bool
    figure_path: Path | None
    show_title: str
    normalize_mode: str
    normalize_eps: float
    normalize_q_low: float
    normalize_q_high: float
    normalize_q_scale: float
    batch: IonImageBatchConfig | None
    project_root: Path
    source_json: Path


def _assert_allowed_top_level_keys(root: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    extra = set(root.keys()) - allowed
    if extra:
        raise ValueError(
            f"{label}: unknown top-level keys {sorted(extra)!r}; allowed: {sorted(allowed)!r}"
        )


def _require_json_version(root: dict[str, Any]) -> None:
    version = root.get("version", None)
    if version != JSON_VERSION:
        raise ValueError(f"Unsupported JSON version {version!r}, expected {JSON_VERSION}")


def _load_field_config(
    root: dict[str, Any],
    json_dir: Path,
    project_root: Path,
) -> Config:
    paths = root.get("paths") or {}
    field_cfg = paths.get("field_config")
    if not field_cfg or not isinstance(field_cfg, str):
        raise ValueError("paths.field_config (string) is required")
    field_path = _resolve_existing_path(field_cfg, json_dir, project_root)
    if not field_path.is_file():
        raise FileNotFoundError(
            f"Field config not found: tried {field_path!s} (from paths.field_config={field_cfg!r})"
        )
    cfg, _ = init_from_config(str(field_path))
    return cfg


def _parse_force_and_particles(
    root: dict[str, Any],
    cfg: Config,
    project_root: Path,
    json_dir: Path,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dyn = root.get("dynamics") or {}
    paths_dict = root.get("paths") or {}
    trap = root.get("trap") or {}
    r0, v0, charge, mass = resolve_dynamics_arrays(
        cast(dict[str, Any], dyn), cfg, project_root, json_dir
    )
    force = resolve_force_callable(
        str(dyn.get("force", "zero")),
        project_root=project_root,
        json_dir=json_dir,
        paths=cast(dict[str, Any], paths_dict),
        cfg=cfg,
        charge=charge,
        trap=cast(dict[str, Any], trap),
    )
    return force, r0, v0, charge, mass


def _parse_integration_block(root: dict[str, Any]) -> IntegrationParams:
    it = root.get("integration") or {}
    n_step = None if "n_step" not in it or it["n_step"] is None else int(it["n_step"])
    if "n_step_per_us" not in it or it["n_step_per_us"] is None:
        n_step_per_us = None
    else:
        n_step_per_us = _as_float(it["n_step_per_us"], "integration.n_step_per_us")
    n_step_pre = None if "n_step_pre" not in it or it["n_step_pre"] is None else int(it["n_step_pre"])
    return IntegrationParams(
        t_start_us=_as_float(it["t_start_us"], "integration.t_start_us"),
        t_cum_us=_as_float(it["t_cum_us"], "integration.t_cum_us"),
        n_step=n_step,
        n_step_per_us=n_step_per_us,
        n_step_pre=n_step_pre,
    )


def _parse_camera_beam_noise(root: dict[str, Any], json_dir: Path) -> tuple[CameraParams, BeamParams, NoiseParams]:
    cam = root.get("camera") or {}
    camera = CameraParams(
        pixel_um=_as_float(cam["pixel_um"], "camera.pixel_um"),
        l=_as_int(cam["l"], "camera.l"),
        h=_as_int(cam["h"], "camera.h"),
        x0_um=float(_optional(cam, "x0_um", 0.0)),
        y0_um=float(_optional(cam, "y0_um", 0.0)),
    )
    b = root.get("beam") or {}
    beam = BeamParams(
        w_um=_as_float(b["w_um"], "beam.w_um"),
        xb_um=float(b.get("xb_um", 0.0)),
        yb_um=float(b.get("yb_um", 0.0)),
        I=float(b.get("I", 1.0)),
    )
    nz = root.get("noise") or {}
    noise = NoiseParams(
        shot_factor=float(nz.get("shot_factor", 1.0)),
        shot_scale=float(nz.get("shot_scale", 1.0)),
        readout_factor=float(nz.get("readout_factor", 0.0)),
        readout_sigma=float(nz.get("readout_sigma", 0.0)),
        bg_offset=float(nz.get("bg_offset", 0.0)),
        seed=None if "seed" not in nz or nz["seed"] is None else int(nz["seed"]),
    )
    return camera, beam, noise


def _parse_imaging_normalize_block(root: dict[str, Any]) -> tuple[float, str, float, float, float, float]:
    im = root.get("imaging") or {}
    if "psf_sigma_px" not in im:
        raise ValueError("imaging.psf_sigma_px is required")
    psf_sigma_px = _as_float(im["psf_sigma_px"], "imaging.psf_sigma_px")
    normalize_mode = str(im.get("normalize_mode", "none")).lower()
    if normalize_mode not in ("none", "max", "minmax"):
        raise ValueError(
            f'imaging.normalize_mode must be "none", "max", or "minmax", got {normalize_mode!r}'
        )
    normalize_eps = _as_float(im.get("normalize_eps", 1e-12), "imaging.normalize_eps")
    normalize_q_low = _as_float(im.get("normalize_q_low", 2.0), "imaging.normalize_q_low")
    normalize_q_high = _as_float(im.get("normalize_q_high", 98.0), "imaging.normalize_q_high")
    normalize_q_scale = _as_float(im.get("normalize_q_scale", 99.0), "imaging.normalize_q_scale")
    return psf_sigma_px, normalize_mode, normalize_eps, normalize_q_low, normalize_q_high, normalize_q_scale


def _parse_display_block(root: dict[str, Any], json_dir: Path) -> tuple[bool, bool, Path | None, str]:
    disp = root.get("display") or {}
    show = bool(disp.get("show", False))
    show_block = bool(disp.get("show_block", True))
    show_title = str(disp.get("show_title", "Ion image (simulation)"))
    fpath_raw = disp.get("figure_path", None)
    if fpath_raw is None or fpath_raw == "":
        figure_path: Path | None = None
    else:
        if not isinstance(fpath_raw, str):
            raise TypeError("display.figure_path must be a string or null")
        figure_path = _resolve_output_path(fpath_raw, json_dir)
    return show, show_block, figure_path, show_title


def _parse_simulation_monolith(sim: dict[str, Any]) -> tuple[bool, Literal["RK4", "VV"], bool, bool, float | None]:
    use_cuda = bool(sim.get("use_cuda", False))
    calc_method = sim.get("calc_method", "VV")
    if calc_method not in ("RK4", "VV"):
        raise ValueError('simulation.calc_method must be "RK4" or "VV"')
    use_zero_force = bool(sim.get("use_zero_force", False))
    apply_sensor_noise = bool(sim.get("apply_sensor_noise", True))
    if "log_interval_sim_us" in sim and sim["log_interval_sim_us"] is not None:
        log_interval_sim_us = _as_float(sim["log_interval_sim_us"], "simulation.log_interval_sim_us")
        if log_interval_sim_us < 0.0:
            raise ValueError("simulation.log_interval_sim_us must be non-negative (or null)")
    else:
        log_interval_sim_us = None
    return use_cuda, calc_method, use_zero_force, apply_sensor_noise, log_interval_sim_us


def _parse_simulation_dynamics_only(sim: dict[str, Any]) -> tuple[bool, Literal["RK4", "VV"], bool, float | None]:
    bad = set(sim.keys()) - {"use_cuda", "calc_method", "use_zero_force", "log_interval_sim_us"}
    if bad:
        raise ValueError(
            f"dynamics JSON simulation block: unknown keys {sorted(bad)!r}; "
            "put apply_sensor_noise in the imaging JSON"
        )
    use_cuda = bool(sim.get("use_cuda", False))
    calc_method = sim.get("calc_method", "VV")
    if calc_method not in ("RK4", "VV"):
        raise ValueError('simulation.calc_method must be "RK4" or "VV"')
    use_zero_force = bool(sim.get("use_zero_force", False))
    if "log_interval_sim_us" in sim and sim["log_interval_sim_us"] is not None:
        log_interval_sim_us = _as_float(sim["log_interval_sim_us"], "simulation.log_interval_sim_us")
        if log_interval_sim_us < 0.0:
            raise ValueError("simulation.log_interval_sim_us must be non-negative (or null)")
    else:
        log_interval_sim_us = None
    return use_cuda, calc_method, use_zero_force, log_interval_sim_us


def _parse_simulation_imaging_only(sim: dict[str, Any]) -> bool:
    bad = set(sim.keys()) - {"apply_sensor_noise"}
    if bad:
        raise ValueError(
            f"imaging JSON simulation block: unknown keys {sorted(bad)!r}; "
            "put use_cuda, calc_method, use_zero_force, log_interval_sim_us in the dynamics JSON"
        )
    return bool(sim.get("apply_sensor_noise", True))


def _merged_ion_bundle(
    dyn: IonDynamicsJsonBundle,
    img: IonImagingJsonBundle,
) -> IonImageJsonBundle:
    batch_out = img.batch
    img_ov = batch_out.dynamics_overrides if batch_out else None
    dyn_ov = dyn.dynamics_batch_overrides
    if img_ov is not None and dyn_ov is not None:
        raise ValueError(
            "batch.dynamics_overrides: define in either the dynamics JSON or the imaging JSON, not both"
        )
    merged_ov = img_ov if img_ov is not None else dyn_ov
    if merged_ov is not None:
        if batch_out is None:
            raise ValueError("batch.dynamics_overrides requires batch.seeds in the imaging JSON")
        if len(merged_ov) != len(batch_out.seeds):
            raise ValueError("batch.dynamics_overrides length must match batch.seeds")
    if batch_out is not None and dyn_ov is not None and img_ov is None:
        batch_out = replace(batch_out, dynamics_overrides=merged_ov)

    img_pp = batch_out.plane_npz_paths if batch_out else None
    dyn_pp = dyn.dynamics_batch_plane_npz_paths
    if img_pp is not None and dyn_pp is not None:
        raise ValueError(
            "batch.plane_npz_paths: define in either the dynamics JSON or the imaging JSON, not both"
        )
    merged_pp = img_pp if img_pp is not None else dyn_pp
    if merged_pp is not None:
        if batch_out is None:
            raise ValueError("batch.plane_npz_paths requires batch.seeds in the imaging JSON")
        if len(merged_pp) != len(batch_out.seeds):
            raise ValueError("batch.plane_npz_paths length must match batch.seeds")
    if batch_out is not None and dyn_pp is not None and img_pp is None:
        batch_out = replace(batch_out, plane_npz_paths=merged_pp)

    return IonImageJsonBundle(
        config=dyn.config,
        force=dyn.force,
        r0=dyn.r0,
        v0=dyn.v0,
        charge=dyn.charge,
        mass=dyn.mass,
        camera=img.camera,
        beam=img.beam,
        noise=img.noise,
        integ=dyn.integ,
        psf_sigma_px=img.psf_sigma_px,
        use_cuda=dyn.use_cuda,
        calc_method=dyn.calc_method,
        use_zero_force=dyn.use_zero_force,
        apply_sensor_noise=img.apply_sensor_noise,
        show=img.show,
        show_block=img.show_block,
        figure_path=img.figure_path,
        show_title=img.show_title,
        project_root=dyn.project_root,
        source_json=img.source_json,
        normalize_mode=img.normalize_mode,
        normalize_eps=img.normalize_eps,
        normalize_q_low=img.normalize_q_low,
        normalize_q_high=img.normalize_q_high,
        normalize_q_scale=img.normalize_q_scale,
        log_interval_sim_us=dyn.log_interval_sim_us,
        batch=batch_out,
        dynamics=dyn.dynamics,
        paths=dyn.paths,
        trap=dyn.trap,
        dynamics_json_dir=dyn.source_json.parent,
        dynamics_source_json=dyn.source_json,
    )


_NOISE_OVERRIDE_KEYS = frozenset(
    {"shot_factor", "shot_scale", "readout_factor", "readout_sigma", "bg_offset", "seed"}
)

# Keys allowed in ``batch.dynamics_overrides`` entries (shallow-merged into root ``dynamics``).
_DYNAMICS_OVERRIDE_KEYS = frozenset(
    {
        "init_file",
        "init_npz",
        "r0_um",
        "v0_m_s",
        "r0",
        "v0",
        "N",
        "init_random",
        "init_seed",
        "init_center_um",
        "init_range_um",
        "init_range",
        "force",
        "mass",
        "charge",
        "alpha",
        "isotope",
        "isotope_type",
        "bilayer",
        "bilayer_y0_um",
    }
)

# Optional per-batch-frame overrides merged onto root ``integration`` (same semantics as top-level ``integration``).
_INTEGRATION_OVERRIDE_KEYS = frozenset(
    {
        "t_start_us",
        "t_cum_us",
        "n_step",
        "n_step_per_us",
        "n_step_pre",
    }
)

_BATCH_DYNAMICS_OR_INTEGRATION_KEYS = _DYNAMICS_OVERRIDE_KEYS | _INTEGRATION_OVERRIDE_KEYS


def _assert_dynamics_override_object(obj: dict[str, Any], index_label: str) -> None:
    for key in obj:
        if key not in _BATCH_DYNAMICS_OR_INTEGRATION_KEYS:
            raise ValueError(f"{index_label} unknown key {key!r}")


def _split_dynamics_integration_patch(
    patch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a batch override dict into dynamics vs integration fragments."""
    dyn: dict[str, Any] = {}
    it: dict[str, Any] = {}
    for k, v in patch.items():
        if k in _DYNAMICS_OVERRIDE_KEYS:
            dyn[k] = v
        elif k in _INTEGRATION_OVERRIDE_KEYS:
            it[k] = v
    return dyn, it


def _merge_integration_params(base: IntegrationParams, patch: dict[str, Any]) -> IntegrationParams:
    """Apply ``patch`` (subset of ``integration`` JSON keys) onto ``base``."""
    if not patch:
        return base
    d: dict[str, Any] = {
        "t_start_us": base.t_start_us,
        "t_cum_us": base.t_cum_us,
        "n_step": base.n_step,
        "n_step_per_us": base.n_step_per_us,
        "n_step_pre": base.n_step_pre,
    }
    for k, v in patch.items():
        if k in ("n_step", "n_step_pre"):
            d[k] = None if v is None else int(v)
        elif k == "n_step_per_us":
            d[k] = None if v is None else _as_float(v, f"batch integration override {k}")
        elif k in ("t_start_us", "t_cum_us"):
            d[k] = _as_float(v, f"batch integration override {k}")
    return IntegrationParams(
        t_start_us=float(d["t_start_us"]),
        t_cum_us=float(d["t_cum_us"]),
        n_step=d["n_step"],
        n_step_per_us=d["n_step_per_us"],
        n_step_pre=d["n_step_pre"],
    )


def _noise_list_for_batch(
    base: NoiseParams,
    seeds: tuple[int, ...],
    noise_overrides: tuple[dict[str, Any], ...] | None,
) -> list[NoiseParams]:
    """Build per-frame :class:`NoiseParams` from ``batch.seeds`` and optional ``noise_overrides``."""
    if noise_overrides is None:
        return [replace(base, seed=s) for s in seeds]
    if len(noise_overrides) != len(seeds):
        raise ValueError("batch.noise_overrides must have the same length as batch.seeds")
    out: list[NoiseParams] = []
    for i, s in enumerate(seeds):
        o = noise_overrides[i]
        n = replace(base, seed=s)
        if not o:
            out.append(n)
            continue
        kw: dict[str, Any] = {}
        for key, val in o.items():
            if key not in _NOISE_OVERRIDE_KEYS:
                raise ValueError(f"batch.noise_overrides[{i}] unknown key {key!r}")
            if key == "seed":
                kw["seed"] = None if val is None else int(val)
            else:
                kw[key] = float(val)
        out.append(replace(n, **kw))
    return out


def noise_params_list_from_imaging_bundle(img: IonImagingJsonBundle) -> list[NoiseParams]:
    """
    Build one :class:`NoiseParams` per output frame from an imaging-only JSON bundle.

    If ``batch`` is absent, returns a single-element list using the root ``noise`` block.
    If ``batch`` is present, uses ``batch.seeds`` and optional ``batch.noise_overrides``
    (same rules as :func:`load_imaging_json` / :meth:`IonImageJsonBundle.call_run_batch`).
    """
    if img.batch is None:
        return [img.noise]
    return _noise_list_for_batch(
        img.noise,
        img.batch.seeds,
        img.batch.noise_overrides,
    )


def psf_sigma_px_list_from_imaging_bundle(img: IonImagingJsonBundle) -> list[float]:
    """
    Per-frame Gaussian PSF ``sigma`` in pixels; length matches :func:`noise_params_list_from_imaging_bundle`.
    """
    noises = noise_params_list_from_imaging_bundle(img)
    n = len(noises)
    if img.batch is None or img.batch.psf_sigma_px is None:
        s = float(img.psf_sigma_px)
        if s < 0:
            raise ValueError("imaging.psf_sigma_px must be non-negative")
        return [s] * n
    seq = [float(x) for x in img.batch.psf_sigma_px]
    if len(seq) != n:
        raise ValueError(
            f"batch.psf_sigma_px length {len(seq)} must match number of batch frames {n}"
        )
    if any(s < 0 for s in seq):
        raise ValueError("batch.psf_sigma_px values must be non-negative")
    return seq


def _parse_batch(raw: Any, json_dir: Path) -> IonImageBatchConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("batch must be an object")
    seeds_raw = raw.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("batch.seeds must be a non-empty array of integers")
    seeds = tuple(int(s) for s in seeds_raw)

    figure_paths: tuple[str | None, ...] | None = None
    fp_raw = raw.get("figure_paths")
    if fp_raw is not None:
        if not isinstance(fp_raw, list) or len(fp_raw) != len(seeds):
            raise ValueError("batch.figure_paths must be an array of the same length as batch.seeds")
        figure_paths = tuple(
            None if x is None or x == "" else str(x)
            for x in fp_raw
        )

    max_workers = _as_int(raw.get("max_workers", 1), "batch.max_workers")
    if max_workers < 1:
        raise ValueError("batch.max_workers must be >= 1")

    sd = raw.get("share_dynamics", None)
    share_dynamics: bool | None
    if sd is None:
        share_dynamics = None
    elif isinstance(sd, bool):
        share_dynamics = sd
    else:
        raise TypeError("batch.share_dynamics must be a boolean or null")

    allow_log = bool(raw.get("allow_batch_progress_log", False))
    profile = bool(raw.get("profile", False))

    noise_overrides: tuple[dict[str, Any], ...] | None = None
    nor_raw = raw.get("noise_overrides")
    if nor_raw is not None:
        if not isinstance(nor_raw, list) or len(nor_raw) != len(seeds):
            raise ValueError(
                "batch.noise_overrides must be an array of the same length as batch.seeds"
            )
        parsed: list[dict[str, Any]] = []
        for j, item in enumerate(nor_raw):
            if item is None:
                parsed.append({})
            elif not isinstance(item, dict):
                raise TypeError(f"batch.noise_overrides[{j}] must be an object or null")
            else:
                parsed.append(dict(item))
        noise_overrides = tuple(parsed)

    psf_list: tuple[float, ...] | None = None
    psf_raw = raw.get("psf_sigma_px")
    if psf_raw is not None:
        if not isinstance(psf_raw, list) or len(psf_raw) != len(seeds):
            raise ValueError(
                "batch.psf_sigma_px must be an array of the same length as batch.seeds"
            )
        psf_list = tuple(float(x) for x in psf_raw)
        if any(s < 0 for s in psf_list):
            raise ValueError("batch.psf_sigma_px values must be non-negative")

    dynamics_overrides: tuple[dict[str, Any], ...] | None = None
    dor = raw.get("dynamics_overrides")
    if dor is not None:
        if not isinstance(dor, list) or len(dor) != len(seeds):
            raise ValueError(
                "batch.dynamics_overrides must be an array of the same length as batch.seeds"
            )
        dparsed: list[dict[str, Any]] = []
        for j, item in enumerate(dor):
            if item is None:
                dct: dict[str, Any] = {}
            elif not isinstance(item, dict):
                raise TypeError(f"batch.dynamics_overrides[{j}] must be an object or null")
            else:
                dct = dict(item)
            _assert_dynamics_override_object(dct, f"batch.dynamics_overrides[{j}]")
            dparsed.append(dct)
        dynamics_overrides = tuple(dparsed)

    plane_npz_paths: tuple[str | None, ...] | None = None
    pnp_raw = raw.get("plane_npz_paths")
    if pnp_raw is not None:
        if not isinstance(pnp_raw, list) or len(pnp_raw) != len(seeds):
            raise ValueError(
                "batch.plane_npz_paths must be an array of the same length as batch.seeds"
            )
        plane_npz_paths = tuple(
            None if x is None or x == "" else str(x) for x in pnp_raw
        )

    return IonImageBatchConfig(
        seeds=seeds,
        figure_paths=figure_paths,
        noise_overrides=noise_overrides,
        dynamics_overrides=dynamics_overrides,
        plane_npz_paths=plane_npz_paths,
        psf_sigma_px=psf_list,
        max_workers=max_workers,
        share_dynamics=share_dynamics,
        allow_batch_progress_log=allow_log,
        profile=profile,
    )


def _as_float(x: Any, name: str) -> float:
    if not isinstance(x, (int, float)) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        raise TypeError(f"{name} must be a finite number")
    return float(x)


def _as_int(x: Any, name: str) -> int:
    if not isinstance(x, int) or isinstance(x, bool):
        if isinstance(x, float) and x == int(x):
            return int(x)
        raise TypeError(f"{name} must be an int")
    return int(x)


def _optional(obj: dict[str, Any], key: str, default: Any) -> Any:
    if key in obj and obj[key] is not None:
        return obj[key]
    return default


def _parse_dynamics_json_batch_block(
    raw: Any,
) -> tuple[tuple[dict[str, Any], ...] | None, tuple[str | None, ...] | None]:
    """
    Dynamics JSON ``batch`` may contain only ``dynamics_overrides`` and optional ``plane_npz_paths``.
    """
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        raise TypeError("batch must be an object")
    extra = set(raw.keys()) - {"dynamics_overrides", "plane_npz_paths"}
    if extra:
        raise ValueError(
            f"dynamics JSON batch: unknown keys {sorted(extra)!r}; use the imaging JSON for "
            "seeds, noise_overrides, figure_paths, psf_sigma_px, max_workers, …"
        )
    o = raw.get("dynamics_overrides")
    pnp_raw = raw.get("plane_npz_paths")
    if o is None:
        if pnp_raw is not None:
            raise ValueError("dynamics JSON batch.plane_npz_paths requires batch.dynamics_overrides")
        return None, None
    if not isinstance(o, list) or not o:
        raise ValueError(
            "dynamics JSON batch.dynamics_overrides must be a non-empty array when present"
        )
    out: list[dict[str, Any]] = []
    for j, item in enumerate(o):
        if item is None:
            d: dict[str, Any] = {}
        elif not isinstance(item, dict):
            raise TypeError(f"batch.dynamics_overrides[{j}] must be an object or null")
        else:
            d = dict(item)
        _assert_dynamics_override_object(d, f"batch.dynamics_overrides[{j}]")
        out.append(d)
    overrides = tuple(out)
    n = len(overrides)
    plane_npz_paths: tuple[str | None, ...] | None = None
    if pnp_raw is not None:
        if not isinstance(pnp_raw, list) or len(pnp_raw) != n:
            raise ValueError(
                "batch.plane_npz_paths must be an array of the same length as batch.dynamics_overrides"
            )
        plane_npz_paths = tuple(
            None if x is None or x == "" else str(x) for x in pnp_raw
        )
    return overrides, plane_npz_paths


def load_dynamics_json(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> IonDynamicsJsonBundle:
    """
    Load a **dynamics-only** JSON (``paths``, ``trap``, ``dynamics``, ``integration``, ``simulation``).

    Must not contain imaging keys such as ``camera``, ``beam``, ``imaging``, or ``display``.
    Optional top-level ``batch`` may list only ``dynamics_overrides``; pair with an imaging
    JSON that defines ``batch.seeds`` (and other batch fields) for :func:`load_ion_image_merged`.
    """
    json_path = Path(path).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, dict):
        raise TypeError("JSON root must be an object")
    _require_json_version(root)
    _assert_allowed_top_level_keys(root, _DYNAMICS_JSON_KEYS, "dynamics JSON")

    project_root_p = Path(project_root).resolve() if project_root is not None else _repo_root()
    json_dir = json_path.parent
    cfg = _load_field_config(root, json_dir, project_root_p)
    force, r0, v0, charge, mass = _parse_force_and_particles(root, cfg, project_root_p, json_dir)
    integ = _parse_integration_block(root)
    use_cuda, calc_method, use_zero_force, log_interval_sim_us = _parse_simulation_dynamics_only(
        cast(dict[str, Any], root.get("simulation") or {})
    )
    dyn_block = cast(dict[str, Any], root.get("dynamics") or {})
    paths_block = cast(dict[str, Any], root.get("paths") or {})
    trap_block = cast(dict[str, Any], root.get("trap") or {})
    dyn_b_o, dyn_b_npz = _parse_dynamics_json_batch_block(root.get("batch"))
    return IonDynamicsJsonBundle(
        config=cfg,
        force=force,
        r0=r0,
        v0=v0,
        charge=charge,
        mass=mass,
        integ=integ,
        use_cuda=use_cuda,
        calc_method=calc_method,
        use_zero_force=use_zero_force,
        log_interval_sim_us=log_interval_sim_us,
        project_root=project_root_p,
        source_json=json_path,
        dynamics=dyn_block,
        paths=paths_block,
        trap=trap_block,
        dynamics_batch_overrides=dyn_b_o,
        dynamics_batch_plane_npz_paths=dyn_b_npz,
    )


def export_dynamics_batch_plane_npz(dyn: IonDynamicsJsonBundle) -> list:
    """
    For a dynamics JSON with ``batch.dynamics_overrides`` and ``batch.plane_npz_paths``:
    run exposure-window integration per override and write one plane-trajectory NPZ per path.

    Does not require an imaging JSON or camera parameters. Relative ``plane_npz_paths`` resolve
    against the dynamics JSON directory (same as merged batch behavior).
    """
    from .plane_trajectory_io import PlaneTrajectoryRecord, export_plane_trajectory_from_simulation

    ov = dyn.dynamics_batch_overrides
    if ov is None:
        raise ValueError("export_dynamics_batch_plane_npz: dynamics JSON has no batch.dynamics_overrides")
    pnp = dyn.dynamics_batch_plane_npz_paths
    if pnp is None:
        raise ValueError(
            "export_dynamics_batch_plane_npz: add batch.plane_npz_paths "
            "(one output path per batch.dynamics_overrides entry)"
        )
    if len(pnp) != len(ov):
        raise ValueError(
            "export_dynamics_batch_plane_npz: batch.plane_npz_paths length must match batch.dynamics_overrides"
        )

    base_dyn = dict(dyn.dynamics)
    ddir = dyn.source_json.parent
    records: list[PlaneTrajectoryRecord] = []
    n_batch = len(ov)

    for i, patch in enumerate(ov):
        outp_raw = pnp[i]
        if outp_raw is None or outp_raw == "":
            raise ValueError(
                f"export_dynamics_batch_plane_npz: batch.plane_npz_paths[{i}] must be a non-empty string"
            )
        outp = _resolve_output_path(str(outp_raw), ddir)
        print(
            f"dynamics batch sample {i + 1}/{n_batch} -> {outp}",
            flush=True,
        )
        dyn_patch, integ_patch = _split_dynamics_integration_patch(patch)
        merged = {**base_dyn, **dyn_patch}
        integ_i = _merge_integration_params(dyn.integ, integ_patch)
        r0, v0, ch, m = resolve_dynamics_arrays(
            merged,
            dyn.config,
            dyn.project_root,
            ddir,
        )
        frc = resolve_force_callable(
            str(merged.get("force", "zero")),
            project_root=dyn.project_root,
            json_dir=ddir,
            paths=cast(dict[str, Any], dyn.paths),
            cfg=dyn.config,
            charge=ch,
            trap=cast(dict[str, Any], dyn.trap),
        )
        meta: dict[str, Any] = {
            "batch_index": i,
            "batch_dynamics_plane_export": True,
            "dynamics_override": dict(patch),
        }
        rec = export_plane_trajectory_from_simulation(
            outp,
            dyn.config,
            frc,
            r0,
            v0,
            ch,
            m,
            integ_i,
            use_cuda=dyn.use_cuda,
            calc_method=dyn.calc_method,
            use_zero_force=dyn.use_zero_force,
            log_interval_sim_us=dyn.log_interval_sim_us,
            meta=meta,
            dynamics_json_path=dyn.source_json,
            project_root=dyn.project_root,
        )
        records.append(rec)
    return records


def load_imaging_json(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> IonImagingJsonBundle:
    """
    Load an **imaging-only** JSON (``camera``, ``beam``, ``noise``, ``imaging``, ``display``, optional ``batch``).

    Optional ``simulation`` block may contain only ``apply_sensor_noise``; all other simulation keys
    belong in the dynamics JSON.
    """
    json_path = Path(path).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, dict):
        raise TypeError("JSON root must be an object")
    _require_json_version(root)
    _assert_allowed_top_level_keys(root, _IMAGING_JSON_KEYS, "imaging JSON")

    project_root_p = Path(project_root).resolve() if project_root is not None else _repo_root()
    json_dir = json_path.parent
    camera, beam, noise = _parse_camera_beam_noise(root, json_dir)
    psf_sigma_px, normalize_mode, normalize_eps, normalize_q_low, normalize_q_high, normalize_q_scale = (
        _parse_imaging_normalize_block(root)
    )
    show, show_block, figure_path, show_title = _parse_display_block(root, json_dir)
    batch_cfg = _parse_batch(root.get("batch"), json_dir)
    apply_sensor_noise = _parse_simulation_imaging_only(cast(dict[str, Any], root.get("simulation") or {}))
    return IonImagingJsonBundle(
        camera=camera,
        beam=beam,
        noise=noise,
        psf_sigma_px=psf_sigma_px,
        apply_sensor_noise=apply_sensor_noise,
        show=show,
        show_block=show_block,
        figure_path=figure_path,
        show_title=show_title,
        normalize_mode=normalize_mode,
        normalize_eps=normalize_eps,
        normalize_q_low=normalize_q_low,
        normalize_q_high=normalize_q_high,
        normalize_q_scale=normalize_q_scale,
        batch=batch_cfg,
        project_root=project_root_p,
        source_json=json_path,
    )


def load_ion_image_merged(
    dynamics_path: str | Path,
    imaging_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> IonImageJsonBundle:
    """Merge :func:`load_dynamics_json` + :func:`load_imaging_json` into one :class:`IonImageJsonBundle`."""
    project_root_p = Path(project_root).resolve() if project_root is not None else _repo_root()
    dyn = load_dynamics_json(dynamics_path, project_root=project_root_p)
    img = load_imaging_json(imaging_path, project_root=project_root_p)
    return _merged_ion_bundle(dyn, img)


def load_ion_image_json(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> IonImageJsonBundle:
    """
    Read and validate a **single** monolithic JSON file; return an :class:`IonImageJsonBundle`.

    For split configs, use :func:`load_ion_image_merged` (or pass two paths to the CLI).

    Parameters
    ----------
    path
        JSON file path.
    project_root
        Repository root (directory that contains ``main.py`` and ``FieldConfiguration/``).
        If omitted, the parent of the ``ImgSimulation`` package (project root) is used.
    """
    json_path = Path(path).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        root = json.load(f)

    _require_json_version(root)
    project_root = Path(project_root).resolve() if project_root is not None else _repo_root()
    json_dir = json_path.parent

    cfg = _load_field_config(root, json_dir, project_root)
    force, r0, v0, charge, mass = _parse_force_and_particles(root, cfg, project_root, json_dir)
    integ = _parse_integration_block(root)
    camera, beam, noise = _parse_camera_beam_noise(root, json_dir)
    psf_sigma_px, normalize_mode, normalize_eps, normalize_q_low, normalize_q_high, normalize_q_scale = (
        _parse_imaging_normalize_block(root)
    )
    use_cuda, calc_method, use_zero_force, apply_sensor_noise, log_interval_sim_us = _parse_simulation_monolith(
        cast(dict[str, Any], root.get("simulation") or {})
    )
    show, show_block, figure_path, show_title = _parse_display_block(root, json_dir)
    batch_cfg = _parse_batch(root.get("batch"), json_dir)

    dyn_block = cast(dict[str, Any], root.get("dynamics") or {})
    paths_block = cast(dict[str, Any], root.get("paths") or {})
    trap_block = cast(dict[str, Any], root.get("trap") or {})

    return IonImageJsonBundle(
        config=cfg,
        force=force,
        r0=r0,
        v0=v0,
        charge=charge,
        mass=mass,
        camera=camera,
        beam=beam,
        noise=noise,
        integ=integ,
        psf_sigma_px=psf_sigma_px,
        use_cuda=use_cuda,
        calc_method=calc_method,
        use_zero_force=use_zero_force,
        apply_sensor_noise=apply_sensor_noise,
        show=show,
        show_block=show_block,
        figure_path=figure_path,
        show_title=show_title,
        project_root=project_root,
        source_json=json_path,
        normalize_mode=normalize_mode,
        normalize_eps=normalize_eps,
        normalize_q_low=normalize_q_low,
        normalize_q_high=normalize_q_high,
        normalize_q_scale=normalize_q_scale,
        log_interval_sim_us=log_interval_sim_us,
        batch=batch_cfg,
        dynamics=dyn_block,
        paths=paths_block,
        trap=trap_block,
        dynamics_json_dir=json_dir,
        dynamics_source_json=json_path,
    )


def run_ion_image_from_json_file(
    *paths: str | Path,
    project_root: str | Path | None = None,
) -> np.ndarray | list[np.ndarray]:
    """
    One **monolith** JSON → :func:`load_ion_image_json`; or **two** paths ``(dynamics, imaging)``
    → :func:`load_ion_image_merged`. Then :meth:`IonImageJsonBundle.call_run_ion_image`, or
    :meth:`IonImageJsonBundle.call_run_batch` when the (imaging) JSON defines a ``batch`` block.
    """
    if len(paths) == 1:
        bundle = load_ion_image_json(paths[0], project_root=project_root)
    elif len(paths) == 2:
        bundle = load_ion_image_merged(paths[0], paths[1], project_root=project_root)
    else:
        raise ValueError(
            "run_ion_image_from_json_file: pass one monolith path or two paths (dynamics, imaging)"
        )
    if bundle.batch is not None:
        return bundle.call_run_batch()
    return bundle.call_run_ion_image()
