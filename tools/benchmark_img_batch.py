#!/usr/bin/env python3
"""
Compare wall time: :func:`ImgSimulation.batch.render_batch` (noise-only fast path)
vs a simple loop of :func:`ImgSimulation.pipeline.render_single_frame`.

Run from repo root::

    python tools/benchmark_img_batch.py

Uses CPU + ``use_zero_force`` and a small single-ion scene (same defaults as smoke tests).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from FieldConfiguration.constants import init_from_config
from FieldParser.force import _zero_force
from ImgSimulation.batch import render_batch
from ImgSimulation.pipeline import render_single_frame
from ImgSimulation.types import (
    BeamParams,
    CameraParams,
    IntegrationParams,
    NoiseParams,
)


def main() -> None:
    n_img = 8
    cfg, _ = init_from_config(str(_ROOT / "FieldConfiguration/configs/default.json"))
    r0 = np.zeros((1, 3), dtype=np.float64)
    v0 = np.zeros((1, 3), dtype=np.float64)
    q = np.ones(1, dtype=np.float64)
    m = np.ones(1, dtype=np.float64)
    camera = CameraParams(pixel_um=0.5, l=64, h=64, x0_um=0.0, y0_um=0.0)
    beam = BeamParams(w_um=50.0, I=1.0)
    integ = IntegrationParams(t_start_us=0.0, t_cum_us=2.0, n_step=60)
    noises = [NoiseParams(seed=i, shot_scale=1e6) for i in range(n_img)]

    t0 = time.perf_counter()
    render_batch(
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
        psf_sigma_px=2.0,
        use_cuda=False,
        use_zero_force=True,
        apply_sensor_noise=True,
        normalize_mode="none",
        share_dynamics=True,
    )
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    for n in noises:
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
            psf_sigma_px=2.0,
            use_cuda=False,
            use_zero_force=True,
            apply_sensor_noise=True,
            normalize_mode="none",
            log_interval_sim_us=None,
        )
    t_loop = time.perf_counter() - t0

    print(f"Images: {n_img}")
    print(f"render_batch (share_dynamics=True): {t_batch:.4f} s")
    print(f"sequential render_single_frame:      {t_loop:.4f} s")
    if t_batch > 0:
        print(f"speedup vs loop: {t_loop / t_batch:.2f}x")


if __name__ == "__main__":
    main()
