"""
Batch ion-image rendering: fast path vs sequential, JSON batch, CLI.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent

from ImgSimulation.types import (
    BeamParams,
    CameraParams,
    IntegrationParams,
    NoiseParams,
)


def test_psf_sigma_list_scalar_and_sequence():
    from ImgSimulation.batch import _psf_sigma_list

    assert _psf_sigma_list(2.0, 3) == [2.0, 2.0, 2.0]
    assert _psf_sigma_list([1.0, 1.5, 0.0], 3) == [1.0, 1.5, 0.0]
    with pytest.raises(ValueError, match="length"):
        _psf_sigma_list([1.0, 2.0], 3)


def test_noise_list_for_batch_overrides():
    from dataclasses import replace

    from ImgSimulation.json_config import _noise_list_for_batch

    base = NoiseParams(seed=0, shot_scale=1e6, readout_sigma=0.1)
    noises = _noise_list_for_batch(
        base,
        (10, 20, 30),
        (
            {"shot_scale": 2e6},
            {},
            {"readout_factor": 3.0, "seed": 99},
        ),
    )
    assert noises[0] == replace(base, seed=10, shot_scale=2e6)
    assert noises[1] == replace(base, seed=20)
    assert noises[2].seed == 99 and noises[2].readout_factor == 3.0 and noises[2].shot_scale == 1e6


def test_can_share_noise_only():
    from ImgSimulation.batch import can_share_dynamics_for_noise_only

    a = NoiseParams(seed=1, readout_sigma=0.5)
    b = NoiseParams(seed=99, readout_sigma=0.5)
    assert can_share_dynamics_for_noise_only([a, b])
    c = NoiseParams(seed=3, readout_sigma=0.6)
    assert not can_share_dynamics_for_noise_only([a, c])


def _tiny_scene():
    from FieldConfiguration.constants import init_from_config

    cfg, _ = init_from_config(str(_ROOT / "FieldConfiguration/configs/default.json"))
    r0 = np.zeros((1, 3), dtype=np.float64)
    v0 = np.zeros((1, 3), dtype=np.float64)
    q = np.ones(1, dtype=np.float64)
    m = np.ones(1, dtype=np.float64)
    camera = CameraParams(pixel_um=0.5, l=32, h=32, x0_um=0.0, y0_um=0.0)
    beam = BeamParams(w_um=50.0, I=1.0)
    integ = IntegrationParams(t_start_us=0.0, t_cum_us=2.0, n_step=40)
    return cfg, r0, v0, q, m, camera, beam, integ


def test_render_batch_fast_path_matches_sequential():
    pytest.importorskip("ionsim")
    from FieldParser.force import _zero_force
    from ImgSimulation.batch import render_batch
    from ImgSimulation.pipeline import render_single_frame

    cfg, r0, v0, q, m, camera, beam, integ = _tiny_scene()
    noises = [
        NoiseParams(seed=11, shot_scale=1e6),
        NoiseParams(seed=22, shot_scale=1e6),
        NoiseParams(seed=33, shot_scale=1e6),
    ]
    out_batch = render_batch(
        cfg,
        _zero_force,
        r0,
        v0,
        q,
        m,
        camera,
        beam,
        integ,
        noises,
        psf_sigma_px=1.5,
        use_cuda=False,
        use_zero_force=True,
        apply_sensor_noise=True,
        normalize_mode="none",
        share_dynamics=True,
    )
    out_seq = [
        render_single_frame(
            cfg,
            _zero_force,
            r0,
            v0,
            q,
            m,
            camera,
            beam,
            n,
            integ,
            psf_sigma_px=1.5,
            use_cuda=False,
            use_zero_force=True,
            apply_sensor_noise=True,
            normalize_mode="none",
            log_interval_sim_us=None,
        )
        for n in noises
    ]
    assert len(out_batch) == len(out_seq)
    for a, b in zip(out_batch, out_seq, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_compute_exposure_map_then_render_from_exposure_matches_single_frame():
    pytest.importorskip("ionsim")
    from FieldParser.force import _zero_force
    from ImgSimulation.pipeline import compute_exposure_map, render_from_exposure, render_single_frame

    cfg, r0, v0, q, m, camera, beam, integ = _tiny_scene()
    noise = NoiseParams(seed=7, shot_scale=1e6)
    psf = 1.2
    exposure = compute_exposure_map(
        cfg,
        _zero_force,
        r0,
        v0,
        q,
        m,
        camera,
        beam,
        integ,
        use_cuda=False,
        use_zero_force=True,
        log_interval_sim_us=None,
    )
    img_split = render_from_exposure(
        exposure,
        psf,
        noise,
        apply_sensor_noise=True,
        normalize_mode="none",
    )
    img_one = render_single_frame(
        cfg,
        _zero_force,
        r0,
        v0,
        q,
        m,
        camera,
        beam,
        noise,
        integ,
        psf_sigma_px=psf,
        use_cuda=False,
        use_zero_force=True,
        apply_sensor_noise=True,
        normalize_mode="none",
        log_interval_sim_us=None,
    )
    np.testing.assert_allclose(img_split, img_one, rtol=0, atol=0)


def test_render_batch_varying_psf_matches_sequential():
    pytest.importorskip("ionsim")
    from FieldParser.force import _zero_force
    from ImgSimulation.batch import render_batch
    from ImgSimulation.pipeline import render_single_frame

    cfg, r0, v0, q, m, camera, beam, integ = _tiny_scene()
    noises = [
        NoiseParams(seed=1, shot_scale=1e6),
        NoiseParams(seed=2, shot_scale=1e6),
        NoiseParams(seed=3, shot_scale=1e6),
    ]
    sigmas = [0.8, 1.5, 2.2]
    out_batch = render_batch(
        cfg,
        _zero_force,
        r0,
        v0,
        q,
        m,
        camera,
        beam,
        integ,
        noises,
        psf_sigma_px=sigmas,
        use_cuda=False,
        use_zero_force=True,
        apply_sensor_noise=True,
        normalize_mode="none",
    )
    out_seq = [
        render_single_frame(
            cfg,
            _zero_force,
            r0,
            v0,
            q,
            m,
            camera,
            beam,
            n,
            integ,
            psf_sigma_px=sig,
            use_cuda=False,
            use_zero_force=True,
            apply_sensor_noise=True,
            normalize_mode="none",
            log_interval_sim_us=None,
        )
        for n, sig in zip(noises, sigmas, strict=True)
    ]
    for a, b in zip(out_batch, out_seq, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_render_batch_varying_noise_magnitude_matches_sequential():
    """Different shot_scale / readout per item: shared blurred, same as full pipeline."""
    pytest.importorskip("ionsim")
    from FieldParser.force import _zero_force
    from ImgSimulation.batch import render_batch
    from ImgSimulation.pipeline import render_single_frame

    cfg, r0, v0, q, m, camera, beam, integ = _tiny_scene()
    noises = [
        NoiseParams(seed=1, shot_scale=1e6, readout_sigma=0.5),
        NoiseParams(seed=2, shot_scale=2e6, readout_factor=2.0, readout_sigma=0.3),
        NoiseParams(seed=3, shot_scale=0.5e6, bg_offset=1.0),
    ]
    out_batch = render_batch(
        cfg,
        _zero_force,
        r0,
        v0,
        q,
        m,
        camera,
        beam,
        integ,
        noises,
        psf_sigma_px=1.5,
        use_cuda=False,
        use_zero_force=True,
        apply_sensor_noise=True,
        normalize_mode="none",
    )
    out_seq = [
        render_single_frame(
            cfg,
            _zero_force,
            r0,
            v0,
            q,
            m,
            camera,
            beam,
            n,
            integ,
            psf_sigma_px=1.5,
            use_cuda=False,
            use_zero_force=True,
            apply_sensor_noise=True,
            normalize_mode="none",
            log_interval_sim_us=None,
        )
        for n in noises
    ]
    for a, b in zip(out_batch, out_seq, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_cli_json_batch_writes_pngs(tmp_path: Path):
    pytest.importorskip("ionsim")
    outp0 = tmp_path / "b0.png"
    outp1 = tmp_path / "b1.png"
    cfg = {
        "version": 1,
        "paths": {"field_config": "FieldConfiguration/configs/default.json"},
        "dynamics": {
            "force": "zero",
            "r0": [[0, 0, 0]],
            "v0": [[0, 0, 0]],
        },
        "camera": {"pixel_um": 0.5, "l": 24, "h": 24},
        "beam": {"w_um": 50.0, "I": 1.0},
        "noise": {"shot_factor": 1.0, "shot_scale": 1.0, "readout_factor": 0.0, "seed": 0},
        "integration": {"t_start_us": 0.0, "t_cum_us": 1.0, "n_step": 16},
        "imaging": {"psf_sigma_px": 1.5, "normalize_mode": "none"},
        "simulation": {
            "use_cuda": False,
            "calc_method": "VV",
            "use_zero_force": True,
            "apply_sensor_noise": False,
        },
        "display": {"show": False, "figure_path": None},
        "batch": {
            "seeds": [7, 8],
            "figure_paths": [str(outp0.name), str(outp1.name)],
            "max_workers": 1,
        },
    }
    jpath = tmp_path / "batch.json"
    jpath.write_text(json.dumps(cfg), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ImgSimulation",
            str(jpath),
            "--no-show",
        ],
        cwd=str(_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    assert outp0.is_file() and outp0.stat().st_size > 0
    assert outp1.is_file() and outp1.stat().st_size > 0
