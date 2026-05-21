"""
Load single-frame simulation parameters from a JSON file (see ``configs/example_ion_image.json``).

* ``paths.field_config`` — voltage / constants JSON for :func:`FieldConfiguration.constants.init_from_config` (``dt``, ``dl``, …). **Required.**
* ``paths.field_csv`` — electric field grid CSV (same as main ``--csv``). **Required** when
  ``dynamics.force`` is ``"trap"``; optional for ``"zero"`` (Python trap force is then zero, Coulomb only in ``ionsim``).
* ``trap`` — optional: ``smooth_axes``, ``smooth_sg`` for ``main._build_force`` (same as main simulation).

``dynamics`` can match :class:`Interface.parameters.Parameters` / ``--init_file``:
``N``, ``init_file`` / ``r0_um``+``v0_m_s`` / legacy dimensionless ``r0``+``v0`` / random init; ``charge``, ``mass``, ``alpha``, ``isotope`` / ``isotope_type``, ``bilayer``.

Paths may be **relative to the JSON file** or to the project root (second try).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from FieldConfiguration.constants import Config, init_from_config

from .json_dynamics import resolve_dynamics_arrays, resolve_force_callable
from .types import BeamParams, CameraParams, IntegrationParams, NoiseParams

JSON_VERSION = 1


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
    n_step_pre: int | None
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
            n_step_pre=self.n_step_pre,
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


def load_ion_image_json(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> IonImageJsonBundle:
    """
    Read and validate a JSON file; return an :class:`IonImageJsonBundle`.

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

    version = root.get("version", None)
    if version != JSON_VERSION:
        raise ValueError(f"Unsupported JSON version {version!r}, expected {JSON_VERSION}")

    project_root = Path(project_root).resolve() if project_root is not None else _repo_root()
    json_dir = json_path.parent

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

    it = root.get("integration") or {}
    integ = IntegrationParams(
        t_start_us=_as_float(it["t_start_us"], "integration.t_start_us"),
        t_cum_us=_as_float(it["t_cum_us"], "integration.t_cum_us"),
        n_step=None if "n_step" not in it or it["n_step"] is None else int(it["n_step"]),
    )

    im = root.get("imaging") or {}
    if "psf_sigma_px" not in im:
        raise ValueError("imaging.psf_sigma_px is required")
    psf_sigma_px = _as_float(im["psf_sigma_px"], "imaging.psf_sigma_px")
    normalize_mode = str(im.get("normalize_mode", "none")).lower()
    if normalize_mode not in ("none", "max", "minmax"):
        raise ValueError(
            f'imaging.normalize_mode must be "none", "max", or "minmax", got {normalize_mode!r}'
        )
    normalize_eps = _as_float(
        im.get("normalize_eps", 1e-12),
        "imaging.normalize_eps",
    )
    normalize_q_low = _as_float(
        im.get("normalize_q_low", 2.0),
        "imaging.normalize_q_low",
    )
    normalize_q_high = _as_float(
        im.get("normalize_q_high", 98.0),
        "imaging.normalize_q_high",
    )
    normalize_q_scale = _as_float(
        im.get("normalize_q_scale", 99.0),
        "imaging.normalize_q_scale",
    )

    sim = root.get("simulation") or {}
    use_cuda = bool(sim.get("use_cuda", False))
    calc_method = sim.get("calc_method", "VV")
    if calc_method not in ("RK4", "VV"):
        raise ValueError('simulation.calc_method must be "RK4" or "VV"')
    use_zero_force = bool(sim.get("use_zero_force", False))
    apply_sensor_noise = bool(sim.get("apply_sensor_noise", True))
    n_step_pre = sim.get("n_step_pre", None)
    n_step_pre = None if n_step_pre is None else int(n_step_pre)

    disp = root.get("display") or {}
    show = bool(disp.get("show", False))
    show_block = bool(disp.get("show_block", True))
    show_title = str(disp.get("show_title", "Ion image (simulation)"))
    fpath_raw = disp.get("figure_path", None)
    figure_path: Path | None
    if fpath_raw is None or fpath_raw == "":
        figure_path = None
    else:
        if not isinstance(fpath_raw, str):
            raise TypeError("display.figure_path must be a string or null")
        figure_path = _resolve_output_path(fpath_raw, json_dir)

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
        calc_method=calc_method,  # type: ignore[arg-type]
        use_zero_force=use_zero_force,
        apply_sensor_noise=apply_sensor_noise,
        n_step_pre=n_step_pre,
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
    )


def run_ion_image_from_json_file(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> np.ndarray:
    """
    :func:`load_ion_image_json` followed by :meth:`IonImageJsonBundle.call_run_ion_image`.
    """
    return load_ion_image_json(path, project_root=project_root).call_run_ion_image()
