"""
ImgSimulation smoke tests (require compiled ionsim).

Interactive figure: set environment variable ``ISM_SHOW_ION_FRAME=1`` and run,
for example::

    ISM_SHOW_ION_FRAME=1 pytest tests/test_imgsimulation.py -s

The test always writes a PNG under pytest's ``tmp_path`` (e.g. ``imgsim_smoke.png``).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent

pytest.importorskip("ionsim")

from FieldConfiguration.constants import init_from_config
from FieldParser.force import _zero_force
from ImgSimulation.pipeline import render_single_frame
from ImgSimulation.visualize import show_ion_frame
from ImgSimulation.types import (
    BeamParams,
    CameraParams,
    IntegrationParams,
    NoiseParams,
)


def test_single_ion_center_peak(tmp_path: Path):
    """Static ion at origin, zero external force; bright spot near image center."""
    cfg, _ = init_from_config(str(_ROOT / "FieldConfiguration/configs/default.json"))
    r0 = np.zeros((1, 3), dtype=np.float64)
    v0 = np.zeros((1, 3), dtype=np.float64)
    q = np.ones(1, dtype=np.float64)
    m = np.ones(1, dtype=np.float64)

    camera = CameraParams(pixel_um=0.5, l=64, h=64, x0_um=0.0, y0_um=0.0)
    beam = BeamParams(w_um=50.0, I=1.0)
    noise = NoiseParams(seed=42)
    integ = IntegrationParams(t_start_us=0.0, t_cum_us=2.0, n_step=60)

    img = render_single_frame(
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
        psf_sigma_px=2.0,
        use_cuda=False,
        use_zero_force=True,
        apply_sensor_noise=False,
    )

    assert img.shape == (64, 64)
    assert np.isfinite(img).all()
    assert float(img.max()) > 0
    # peak at image center (row, col) ≈ (31.5, 31.5) for 64×64
    jm, im = np.unravel_index(int(np.argmax(img)), img.shape)
    assert abs(jm - 31.5) < 6.0 and abs(im - 31.5) < 6.0

    # Visualize: PNG in tmp_path; interactive window if ISM_SHOW_ION_FRAME=1
    show_block = os.environ.get("ISM_SHOW_ION_FRAME", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    show_ion_frame(
        img,
        camera=camera,
        title="Ion image (smoke test)",
        save_path=tmp_path / "imgsim_smoke.png",
        block=show_block,
    )


def test_cli_module_json_no_show(tmp_path: Path):
    """python -m ImgSimulation <json> --no-show -o ... writes PNG."""
    pytest.importorskip("ionsim")
    cfg = _ROOT / "ImgSimulation/configs/example_ion_image.json"
    out = tmp_path / "cli_ion_image.png"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ImgSimulation",
            str(cfg),
            "--no-show",
            "-o",
            str(out),
        ],
        cwd=str(_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
