"""Split dynamics / imaging JSON loading."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_MONO = _ROOT / "ImgSimulation/configs/example_ion_image.json"
_DYN = _ROOT / "ImgSimulation/configs/example_dynamics.json"
_IMG = _ROOT / "ImgSimulation/configs/example_imaging.json"


def test_merged_bundle_matches_monolith():
    from ImgSimulation.json_config import load_ion_image_json, load_ion_image_merged

    mono = load_ion_image_json(_MONO)
    merged = load_ion_image_merged(_DYN, _IMG)
    assert mono.camera == merged.camera
    assert mono.beam == merged.beam
    assert mono.noise == merged.noise
    assert mono.integ == merged.integ
    assert mono.psf_sigma_px == merged.psf_sigma_px
    assert mono.use_cuda == merged.use_cuda
    assert mono.calc_method == merged.calc_method
    assert mono.use_zero_force == merged.use_zero_force
    assert mono.apply_sensor_noise == merged.apply_sensor_noise
    assert mono.log_interval_sim_us == merged.log_interval_sim_us
    assert mono.normalize_mode == merged.normalize_mode
    np.testing.assert_array_equal(mono.r0, merged.r0)
    np.testing.assert_array_equal(mono.v0, merged.v0)
    np.testing.assert_array_equal(mono.charge, merged.charge)
    np.testing.assert_array_equal(mono.mass, merged.mass)
    assert merged.dynamics == mono.dynamics
    assert merged.paths == mono.paths
    assert merged.trap == mono.trap
    assert merged.dynamics_json_dir == _DYN.parent
    assert mono.dynamics_json_dir == _MONO.parent
    assert merged.dynamics_json_dir == mono.dynamics_json_dir


def test_dynamics_json_rejects_imaging_keys():
    from ImgSimulation.json_config import load_dynamics_json

    bad = _DYN.read_text(encoding="utf-8")
    root = json.loads(bad)
    root["camera"] = {"pixel_um": 1.0, "l": 1, "h": 1}
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(root, f)
        f.flush()
        p = Path(f.name)
    try:
        with pytest.raises(ValueError, match="unknown top-level"):
            load_dynamics_json(p)
    finally:
        p.unlink(missing_ok=True)


def test_imaging_json_rejects_dynamics_keys():
    from ImgSimulation.json_config import load_imaging_json

    bad = _IMG.read_text(encoding="utf-8")
    root = json.loads(bad)
    root["dynamics"] = {"force": "zero", "N": 1}
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(root, f)
        f.flush()
        p = Path(f.name)
    try:
        with pytest.raises(ValueError, match="unknown top-level"):
            load_imaging_json(p)
    finally:
        p.unlink(missing_ok=True)


def test_run_ion_image_from_json_file_path_count():
    from ImgSimulation.json_config import run_ion_image_from_json_file

    with pytest.raises(ValueError, match="one monolith"):
        run_ion_image_from_json_file()
    with pytest.raises(ValueError, match="one monolith"):
        run_ion_image_from_json_file(_MONO, _DYN, _IMG)


def test_dynamics_overrides_parsed_in_batch():
    from ImgSimulation.json_config import _parse_batch

    raw = {
        "seeds": [1, 2],
        "dynamics_overrides": [{"init_seed": 10}, None],
    }
    c = _parse_batch(raw, _ROOT)
    assert c is not None and c.dynamics_overrides is not None
    assert c.dynamics_overrides[0] == {"init_seed": 10}
    assert c.dynamics_overrides[1] == {}


def test_dynamics_overrides_may_include_integration_keys():
    from ImgSimulation.json_config import _parse_batch

    raw = {
        "seeds": [1],
        "dynamics_overrides": [{"init_seed": 10, "t_start_us": 400.0, "t_cum_us": 50.0}],
    }
    c = _parse_batch(raw, _ROOT)
    assert c is not None and c.dynamics_overrides is not None
    assert c.dynamics_overrides[0]["t_start_us"] == 400.0
    assert c.dynamics_overrides[0]["t_cum_us"] == 50.0


def test_merge_integration_params_from_batch_patch():
    from ImgSimulation.json_config import _merge_integration_params
    from ImgSimulation.types import IntegrationParams

    base = IntegrationParams(
        t_start_us=300.0,
        t_cum_us=100.0,
        n_step=None,
        n_step_per_us=100.0,
        n_step_pre=None,
    )
    m = _merge_integration_params(base, {"t_start_us": 400.0})
    assert m.t_start_us == 400.0
    assert m.t_cum_us == 100.0
    assert m.n_step_per_us == 100.0


def test_example_dynamics_json_accepts_integration_in_overrides():
    from ImgSimulation.json_config import load_dynamics_json

    dyn = load_dynamics_json(_DYN, project_root=_ROOT)
    assert dyn.dynamics_batch_overrides is not None
    assert "t_start_us" in dyn.dynamics_batch_overrides[0]


def test_merged_rejects_dynamics_overrides_in_both_json():
    from ImgSimulation.json_config import _merged_ion_bundle, load_dynamics_json, load_imaging_json
    import copy
    import tempfile

    djson = copy.deepcopy(json.loads(_DYN.read_text(encoding="utf-8")))
    djson["batch"] = {"dynamics_overrides": [{"init_seed": 0}, {"init_seed": 1}]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(djson, f)
        f.flush()
        dyn_p = Path(f.name)
    ij = copy.deepcopy(json.loads(_IMG.read_text(encoding="utf-8")))
    ij["batch"] = {
        "seeds": [10, 20],
        "dynamics_overrides": [{"init_seed": 2}, {"init_seed": 3}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ij, f)
        f.flush()
        img_p = Path(f.name)
    try:
        dyn = load_dynamics_json(dyn_p)
        img = load_imaging_json(img_p)
        with pytest.raises(ValueError, match="not both"):
            _merged_ion_bundle(dyn, img)
    finally:
        dyn_p.unlink(missing_ok=True)
        img_p.unlink(missing_ok=True)


def test_merged_includes_plane_npz_paths_from_dynamics():
    from ImgSimulation.json_config import load_ion_image_merged

    merged = load_ion_image_merged(_DYN, _IMG)
    b = merged.batch
    assert b is not None
    assert b.plane_npz_paths is not None
    assert len(b.plane_npz_paths) == len(b.seeds)
    assert b.plane_npz_paths[0].endswith(".npz")


def test_export_dynamics_batch_plane_npz_requires_paths(tmp_path):
    import copy

    from ImgSimulation.json_config import export_dynamics_batch_plane_npz, load_dynamics_json

    djson = copy.deepcopy(json.loads(_DYN.read_text(encoding="utf-8")))
    djson["paths"]["field_config"] = str(_ROOT / "FieldConfiguration/configs/default.json")
    djson["paths"]["field_csv"] = str(_ROOT / "data/monolithic20241118.csv")
    del djson["batch"]["plane_npz_paths"]
    p = tmp_path / "dyn.json"
    p.write_text(json.dumps(djson), encoding="utf-8")
    dyn = load_dynamics_json(p, project_root=_ROOT)
    with pytest.raises(ValueError, match="plane_npz_paths"):
        export_dynamics_batch_plane_npz(dyn)


def test_merged_rejects_plane_npz_in_both_json():
    from ImgSimulation.json_config import _merged_ion_bundle, load_dynamics_json, load_imaging_json
    import copy
    import tempfile

    djson = copy.deepcopy(json.loads(_DYN.read_text(encoding="utf-8")))
    djson["batch"] = {
        "dynamics_overrides": [{"init_seed": 0}, {"init_seed": 1}],
        "plane_npz_paths": ["a.npz", "b.npz"],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(djson, f)
        f.flush()
        dyn_p = Path(f.name)
    ij = copy.deepcopy(json.loads(_IMG.read_text(encoding="utf-8")))
    ij["batch"] = {"seeds": [1, 2], "plane_npz_paths": ["x.npz", "y.npz"]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ij, f)
        f.flush()
        img_p = Path(f.name)
    try:
        dyn = load_dynamics_json(dyn_p)
        img = load_imaging_json(img_p)
        with pytest.raises(ValueError, match="not both"):
            _merged_ion_bundle(dyn, img)
    finally:
        dyn_p.unlink(missing_ok=True)
        img_p.unlink(missing_ok=True)
