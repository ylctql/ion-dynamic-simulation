"""
Parse ``dynamics`` and trap-field sections from the ion-image JSON.

Coordinates follow :class:`Interface.parameters.Parameters` (dimensionless) once loaded;
``init_file`` and ``r0_um`` / ``v0_m_s`` use the same unit conventions as :func:`Interface.cli.parse_and_build`
(positions in μm, velocities in m/s, then convert with ``Config.dl`` / ``Config.dt``).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from FieldConfiguration.constants import Config
from FieldConfiguration.field_settings import FieldSettings
from FieldConfiguration.loader import build_voltage_list, field_settings_from_config, load_field_config
from FieldParser.force import _zero_force
from Interface.parameters import Parameters

def _import_main_module(project_root: Path) -> Any:
    main_path = project_root / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(f"expected main.py at {main_path}")
    spec = importlib.util.spec_from_file_location("ism_main_ion_json", main_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_existing_path(s: str, json_dir: Path, project_root: Path) -> Path:
    p = Path(s)
    if p.is_absolute():
        return p
    a = (json_dir / p).resolve()
    if a.is_file():
        return a
    return (project_root / p).resolve()


def _parse_smooth_axes(val: Any) -> tuple[str, ...] | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "none"):
        return None
    return tuple(x.strip() for x in s.split(",") if x.strip())


def build_field_settings_for_trap(
    field_csv: str,
    field_config: str,
    cfg: Config,
) -> FieldSettings:
    """Build :class:`FieldSettings` the same way as :func:`Interface.cli.parse_and_build` when CSV exists."""
    import pandas as pd

    csv_path = str(Path(field_csv).resolve())
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"field CSV not found: {p}")
    dat = pd.read_csv(p, comment="%", header=None)
    n_voltage = dat.shape[1] - 3
    if n_voltage > 0:
        try:
            return field_settings_from_config(csv_path, field_config, n_voltage, cfg)
        except FileNotFoundError as e:
            import logging

            logging.getLogger(__name__).warning(
                "field_settings_from_config failed: %s; using zero-voltage list", e
            )
            config = load_field_config(field_config) if Path(field_config).exists() else {"g": 0.1, "voltage_list": []}
            if not isinstance(config, dict):
                config = {"g": 0.1, "voltage_list": []}
            if "g" not in config:
                config = {**config, "g": 0.1}
            voltage_list = build_voltage_list(config, n_voltage, cfg)
            return FieldSettings(
                csv_filename=csv_path,
                voltage_list=voltage_list,
                g=float(config.get("g", 0.1)),
            )
    return FieldSettings(csv_filename=csv_path, voltage_list=[], g=0.1)


def build_trap_force(
    project_root: Path,
    field_settings: FieldSettings,
    cfg: Config,
    charge: np.ndarray,
    *,
    smooth_axes: tuple[str, ...] | None = ("z",),
    smooth_sg: tuple[int, int] = (11, 3),
) -> Callable[..., np.ndarray]:
    main = _import_main_module(project_root)
    return main._build_force(
        field_settings,
        cfg,
        np.asarray(charge, dtype=np.float64),
        project_root,
        smooth_axes=smooth_axes,
        smooth_sg=smooth_sg,
    )


def load_r0_v0_from_npz(init_path: Path, cfg: Config, t0_for_us: bool = False) -> tuple[np.ndarray, np.ndarray, int, float | None]:
    """
    Load ``r`` (μm) and ``v`` (m/s) from an ``.npz``; return dimensionless ``r0``, ``v0``, N, and optional ``t0_us`` for message only.
    """
    data = dict(np.load(init_path, allow_pickle=True))
    if "r" not in data or "v" not in data:
        raise ValueError(f"init_file .npz must contain 'r' and 'v' keys, got: {list(data.keys())}")
    r_um = np.asarray(data["r"], dtype=float)
    v_si = np.asarray(data["v"], dtype=float)
    if r_um.ndim != 2 or r_um.shape[1] != 3:
        raise ValueError(f"r must be (N, 3), got {r_um.shape}")
    if v_si.ndim != 2 or v_si.shape[1] != 3:
        raise ValueError(f"v must be (N, 3), got {v_si.shape}")
    n_file = r_um.shape[0]
    r0 = (r_um * 1e-6 / cfg.dl).astype(float, order="C")
    v0 = (v_si * cfg.dt / cfg.dl).astype(float, order="C")
    t0_us: float | None = None
    if "t_us" in data:
        t0_us = float(np.asarray(data["t_us"]).item())
    elif "t" in data:
        t0_us = float(np.asarray(data["t"]).item())
    if t0_us is None:
        m = re.match(r"t(\d+(?:\.\d+)?)us\.npz$", init_path.name, re.IGNORECASE)
        if m is not None:
            t0_us = float(m.group(1))
    return r0, v0, n_file, t0_us


def _charge_mass_from_parameters(n_ions: int, dyn: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Same *q* and *m* rules as :class:`Parameters` (default unit charge, relative mass, optional isotope doping)."""
    p = Parameters(
        N=n_ions,
        m=np.asarray(dyn["mass"], dtype=float) if dyn.get("mass") is not None else None,
        q=np.asarray(dyn["charge"], dtype=float) if dyn.get("charge") is not None else None,
        alpha=float(dyn.get("alpha", 0.0)),
        isotope_type=dyn.get("isotope", dyn.get("isotope_type")),
    )
    return np.asarray(p.q, dtype=np.float64), np.asarray(p.m, dtype=np.float64)


def resolve_dynamics_arrays(
    dyn: dict[str, Any],
    cfg: Config,
    project_root: Path,
    json_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve ``r0``, ``v0`` (dimensionless), ``charge``, ``mass`` (1d, length N)
    from ``dynamics`` using the same semantics as the main ``Parameters`` / ``init_file`` flow.
    """
    init_path_raw = dyn.get("init_file") or dyn.get("init_npz")
    if init_path_raw:
        p = _resolve_existing_path(str(init_path_raw), json_dir, project_root)
        if not p.is_file():
            raise FileNotFoundError(f"dynamics.init_file not found: {p}")
        r0, v0, n_ions, _t0 = load_r0_v0_from_npz(p, cfg)
    elif dyn.get("r0_um") is not None and dyn.get("v0_m_s") is not None:
        r_um = _as_arr2("dynamics.r0_um", dyn["r0_um"], 3)
        v_m_s = _as_arr2("dynamics.v0_m_s", dyn["v0_m_s"], 3)
        n_ions = r_um.shape[0]
        if v_m_s.shape[0] != n_ions:
            raise ValueError("dynamics.r0_um and v0_m_s must have the same N")
        r0 = (r_um * 1e-6 / cfg.dl).astype(float, order="C")
        v0 = (v_m_s * cfg.dt / cfg.dl).astype(float, order="C")
    elif dyn.get("r0") is not None and dyn.get("v0") is not None:
        r0 = _as_arr2("dynamics.r0 (dimensionless, legacy)", dyn["r0"], 3)
        v0 = _as_arr2("dynamics.v0 (dimensionless, legacy)", dyn["v0"], 3)
        n_ions = r0.shape[0]
        if v0.shape[0] != n_ions:
            raise ValueError("dynamics.r0 and v0 must have the same N")
    elif _truthy(dyn.get("init_random", True), default=True) and int(dyn.get("N", 0) or 0) >= 1:
        n_ions = int(dyn.get("N", 0) or 0)
        if "init_seed" in dyn and dyn["init_seed"] is not None:
            np.random.seed(int(dyn["init_seed"]))

        ic = dyn.get("init_center_um", [0.0, 0.0, 0.0])
        if not isinstance(ic, (list, tuple)) or len(ic) != 3:
            raise ValueError("dynamics.init_center_um must be a length-3 list (μm)")
        cx_um, cy_um, cz_um = (float(x) for x in ic)
        if "init_range_um" in dyn and dyn["init_range_um"] is not None:
            ir = dyn["init_range_um"]
            if isinstance(ir, (int, float)):
                span_um = (float(ir),) * 3
            else:
                irl = [float(x) for x in ir]
                if len(irl) == 1:
                    span_um = (irl[0],) * 3
                else:
                    span_um = (irl[0], irl[1], irl[2] if len(irl) > 2 else irl[0])
            init_range: float | tuple[float, float, float] = (
                float(span_um[0] * 1e-6 / cfg.dl),
                float(span_um[1] * 1e-6 / cfg.dl),
                float(span_um[2] * 1e-6 / cfg.dl),
            )
        else:
            init_range = float(dyn.get("init_range", 150.0))
        init_center = (
            cx_um * 1e-6 / cfg.dl,
            cy_um * 1e-6 / cfg.dl,
            cz_um * 1e-6 / cfg.dl,
        )
        p = Parameters(
            N=n_ions,
            r0=None,
            v0=None,
            init_center=init_center,
            init_range=init_range,
            m=np.asarray(dyn["mass"], dtype=float) if dyn.get("mass") is not None else None,
            q=np.asarray(dyn["charge"], dtype=float) if dyn.get("charge") is not None else None,
            alpha=float(dyn.get("alpha", 0.0)),
            isotope_type=dyn.get("isotope", dyn.get("isotope_type")),
            bilayer=bool(dyn.get("bilayer", False)),
            bilayer_y0_um=float(dyn.get("bilayer_y0_um", 100.0)),
        )
        r0 = p.get_r0()
        v0 = p.get_v0()
        if p.bilayer:
            from Interface.bilayer_init import apply_bilayer_y_split

            r0 = apply_bilayer_y_split(r0, p.N, p.bilayer_y0_um, cfg.dl)
        return (
            r0,
            v0,
            np.asarray(p.q, dtype=np.float64),
            np.asarray(p.m, dtype=np.float64),
        )
    else:
        raise ValueError(
            "dynamics: set init_file, or (r0_um + v0_m_s), or (r0 + v0) dimensionless, "
            "or init_random with positive N (and init_center_um / init_range or init_range_um)"
        )

    ch, m = _charge_mass_from_parameters(n_ions, dyn)
    return r0, v0, ch, m


def _truthy(x: Any, *, default: bool) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("0", "false", "no", ""):
        return False
    if s in ("1", "true", "yes"):
        return True
    return default


def _as_arr1(name: str, data: list[Any], n: int) -> np.ndarray:
    a = np.asarray(data, dtype=np.float64).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} must have length {n}")
    return a


def _as_arr2(name: str, data: list[Any], shape_cols: int) -> np.ndarray:
    a = np.asarray(data, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != shape_cols:
        raise ValueError(f"{name} must be a list of {shape_cols}-vectors (N by {shape_cols})")
    return a


def resolve_force_callable(
    force_name: str,
    *,
    project_root: Path,
    json_dir: Path,
    paths: dict[str, Any],
    cfg: Config,
    charge: np.ndarray,
    trap: dict[str, Any],
) -> Callable[..., np.ndarray]:
    name = str(force_name).lower()
    if name in ("0", "zero", "none"):
        return _zero_force
    if name in ("trap", "field", "full", "lattice"):
        field_cfg = paths.get("field_config")
        field_csv = paths.get("field_csv")
        if not field_csv or not isinstance(field_csv, str):
            raise ValueError(
                'dynamics.force is "trap" (or "field") but paths.field_csv is missing; '
                "it must point to the same electric-field CSV as the main simulation."
            )
        csv_resolved = _resolve_existing_path(str(field_csv), json_dir, project_root)
        if not csv_resolved.is_file():
            raise FileNotFoundError(
                f"paths.field_csv not found: tried {csv_resolved} (from {field_csv!r})"
            )
        if not field_cfg or not isinstance(field_cfg, str):
            raise ValueError("paths.field_config (voltage JSON) is required when dynamics.force is trap")
        fconf = _resolve_existing_path(field_cfg, json_dir, project_root)
        if not fconf.is_file():
            raise FileNotFoundError(f"paths.field_config not found: {fconf}")
        field_settings = build_field_settings_for_trap(str(csv_resolved), str(fconf), cfg)
        s_axes = _parse_smooth_axes(trap.get("smooth_axes", "z"))
        sg = trap.get("smooth_sg", [11, 3])
        if not isinstance(sg, (list, tuple)) or len(sg) != 2:
            raise ValueError("trap.smooth_sg must be [window, polyorder], e.g. [11, 3]")
        smooth_sg = (int(sg[0]), int(sg[1]))
        return build_trap_force(
            project_root,
            field_settings,
            cfg,
            charge,
            smooth_axes=s_axes,
            smooth_sg=smooth_sg,
        )
    raise ValueError(
        f"Unsupported dynamics.force {force_name!r}; use 'zero' or 'trap' (requires paths.field_csv)."
    )
