"""Plane trajectory NPZ I/O and decoupled rendering."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ImgSimulation.plane_trajectory_io import (
    DEFAULT_CONVENTION,
    load_plane_trajectory_npz,
    r_xy_list_to_stack,
    save_plane_trajectory_npz,
    stack_to_r_xy_list,
)
from ImgSimulation.types import BeamParams, CameraParams, IntegrationParams, NoiseParams

_ROOT = Path(__file__).resolve().parent.parent


def test_roundtrip_npz():
    rng = np.random.default_rng(0)
    t, n = 5, 3
    xy_stack = rng.standard_normal((t, n, 2)).astype(np.float64)
    dt = 1e-7
    meta = {"source": "test", "note": "α"}

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "traj.npz"
        save_plane_trajectory_npz(path, xy_stack, dt, meta=meta)
        rec = load_plane_trajectory_npz(path)

    assert rec.xy_stack.shape == xy_stack.shape
    assert np.allclose(rec.xy_stack, xy_stack)
    assert rec.dt_real_s == dt
    assert rec.meta.get("source") == "test"
    assert rec.meta.get("note") == "α"
    assert rec.meta.get("convention") == DEFAULT_CONVENTION
    assert rec.meta.get("n_time") == t
    assert rec.meta.get("n_ions") == n

    rlist = stack_to_r_xy_list(rec.xy_stack)
    assert len(rlist) == t
    assert np.allclose(r_xy_list_to_stack(rlist), rec.xy_stack)


def test_stack_to_list_roundtrip_errors():
    with pytest.raises(ValueError, match="shape"):
        stack_to_r_xy_list(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="at least 2"):
        r_xy_list_to_stack([np.zeros((1, 2))])


@pytest.mark.parametrize("use_export", [False, True])
def test_render_matches_r_plane_lists(use_export: bool):
    pytest.importorskip("ionsim")
    from FieldConfiguration.constants import init_from_config
    from FieldParser.force import _zero_force

    from ImgSimulation.pipeline import (
        compute_exposure_trajectory,
        render_from_r_plane_lists,
        r_list_to_r_plane_lists,
    )
    from ImgSimulation.plane_trajectory_io import (
        export_plane_trajectory_from_simulation,
        render_from_plane_trajectory_file,
    )

    cfg, _ = init_from_config(str(_ROOT / "FieldConfiguration/configs/default.json"))
    r0 = np.zeros((1, 3), dtype=np.float64)
    v0 = np.zeros((1, 3), dtype=np.float64)
    q = np.ones(1, dtype=np.float64)
    m = np.ones(1, dtype=np.float64)
    camera = CameraParams(pixel_um=0.5, l=32, h=32, x0_um=0.0, y0_um=0.0)
    beam = BeamParams(w_um=50.0, I=1.0)
    integ = IntegrationParams(t_start_us=0.0, t_cum_us=2.0, n_step=40)
    noise = NoiseParams(seed=7, shot_scale=1e6, readout_sigma=0.05)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.npz"
        if use_export:
            export_plane_trajectory_from_simulation(
                path,
                cfg,
                _zero_force,
                r0,
                v0,
                q,
                m,
                integ,
                use_cuda=False,
                use_zero_force=True,
                meta={"test": "export"},
            )
        else:
            r_list, _v, dt_real_s = compute_exposure_trajectory(
                cfg,
                _zero_force,
                r0,
                v0,
                q,
                m,
                integ,
                use_cuda=False,
                use_zero_force=True,
            )
            _, _, xy_stack = r_list_to_r_plane_lists(
                r_list,
                cfg,
                camera,
                return_mean_plane_px=False,
            )
            save_plane_trajectory_npz(path, xy_stack, dt_real_s, meta={"test": "manual"})

        r_list, _v, dt_real_s = compute_exposure_trajectory(
            cfg,
            _zero_force,
            r0,
            v0,
            q,
            m,
            integ,
            use_cuda=False,
            use_zero_force=True,
        )
        _, _, xy_stack = r_list_to_r_plane_lists(
            r_list,
            cfg,
            camera,
            return_mean_plane_px=False,
        )
        expected = render_from_r_plane_lists(
            xy_stack,
            camera,
            beam,
            dt_real_s,
            1.5,
            noise,
            normalize_mode="none",
        )
        got = render_from_plane_trajectory_file(
            path,
            camera,
            beam,
            1.5,
            noise,
            normalize_mode="none",
        )

    assert expected.shape == got.shape
    assert np.allclose(expected, got, rtol=0, atol=1e-12)


def test_build_dynamics_provenance_meta_snapshot():
    from ImgSimulation.plane_trajectory_io import build_dynamics_provenance_meta

    dyn = _ROOT / "ImgSimulation/configs/example_dynamics.json"
    prov = build_dynamics_provenance_meta(dyn, project_root=_ROOT)
    assert prov["dynamics_json_path"] == str(dyn.resolve())
    content = prov["dynamics_json_content"]
    assert content["dynamics"]["force"] == "trap"
    fs = prov["field_setup"]
    assert isinstance(fs["field_config_content"], dict)
    assert fs["field_csv"]["basename"] == "monolithic20241118.csv"
    assert "resolved_path" in fs["field_csv"]


def test_npz_roundtrip_preserves_dynamics_provenance():
    from ImgSimulation.plane_trajectory_io import build_dynamics_provenance_meta

    prov = build_dynamics_provenance_meta(
        _ROOT / "ImgSimulation/configs/example_dynamics.json",
        project_root=_ROOT,
    )
    rng = np.random.default_rng(1)
    xy = rng.standard_normal((4, 2, 2)).astype(np.float64)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.npz"
        save_plane_trajectory_npz(path, xy, 1e-8, meta={"dynamics_provenance": prov})
        rec = load_plane_trajectory_npz(path)

    assert rec.meta["dynamics_provenance"]["dynamics_json_content"]["version"] == 1
    assert isinstance(
        rec.meta["dynamics_provenance"]["field_setup"]["field_config_content"],
        dict,
    )


def test_export_merges_user_dynamics_provenance():
    pytest.importorskip("ionsim")
    from FieldConfiguration.constants import init_from_config
    from FieldParser.force import _zero_force

    from ImgSimulation.plane_trajectory_io import export_plane_trajectory_from_simulation

    cfg, _ = init_from_config(str(_ROOT / "FieldConfiguration/configs/default.json"))
    r0 = np.zeros((1, 3), dtype=np.float64)
    v0 = np.zeros((1, 3), dtype=np.float64)
    q = np.ones(1, dtype=np.float64)
    m = np.ones(1, dtype=np.float64)
    integ = IntegrationParams(t_start_us=0.0, t_cum_us=0.5, n_step=8)
    dyn_json = _ROOT / "ImgSimulation/configs/example_dynamics.json"

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "p.npz"
        export_plane_trajectory_from_simulation(
            out,
            cfg,
            _zero_force,
            r0,
            v0,
            q,
            m,
            integ,
            use_cuda=False,
            use_zero_force=True,
            meta={"dynamics_provenance": {"note": "user"}},
            dynamics_json_path=dyn_json,
            project_root=_ROOT,
        )
        rec = load_plane_trajectory_npz(out)

    dp = rec.meta["dynamics_provenance"]
    assert dp["note"] == "user"
    assert dp["dynamics_json_content"]["dynamics"]["force"] == "trap"
    assert "field_setup" in dp
