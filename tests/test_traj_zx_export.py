"""Batch export ion images from traj_zx-style plane NPZ files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ImgSimulation.plane_trajectory_io import save_plane_trajectory_npz
from ImgSimulation.traj_zx_export import export_ion_npy_from_traj_dir


def test_export_ion_npy_from_traj_dir_writes_npy_and_meta(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    imgs_root = tmp_path / "Imgs"
    meta_root = tmp_path / "meta"

    xy = np.zeros((4, 2, 2), dtype=np.float64)
    xy[:, 0, 0] = np.linspace(-1e-3, 1e-3, 4)
    xy[:, 0, 1] = 0.0
    xy[:, 1, 0] = np.linspace(0.5e-3, -0.5e-3, 4)
    xy[:, 1, 1] = 0.2e-3
    npz_path = traj_dir / "case_a.npz"
    save_plane_trajectory_npz(npz_path, xy, 1e-8, meta={"test": True}, write_mean_pos_zx=False)

    imaging = {
        "version": 1,
        "camera": {
            "pixel_um": 0.25,
            "l": 32,
            "h": 32,
            "x0_um": 0.0,
            "y0_um": 0.0,
        },
        "beam": {"w_um": 500.0, "xb_um": 0.0, "yb_um": 0.0, "I": 10000.0},
        "noise": {
            "shot_factor": 1.0,
            "shot_scale": 1e10,
            "readout_factor": 1e7,
            "readout_sigma": 1.0,
            "bg_offset": 0.0,
            "seed": 42,
        },
        "imaging": {
            "psf_sigma_px": 2.0,
            "normalize_mode": "max",
            "normalize_eps": 1e-12,
            "normalize_q_low": 0.1,
            "normalize_q_high": 99.9,
            "normalize_q_scale": 99.9,
        },
        "display": {"show": False, "show_block": True, "figure_path": None, "show_title": "t"},
        "simulation": {"apply_sensor_noise": True},
        "batch": {"seeds": [1001, 1002], "max_workers": 1},
    }
    img_json = tmp_path / "imaging.json"
    img_json.write_text(json.dumps(imaging), encoding="utf-8")

    done = export_ion_npy_from_traj_dir(
        img_json,
        traj_dir=traj_dir,
        imgs_root=imgs_root,
        meta_root=meta_root,
        project_root=tmp_path,
    )
    assert done == [npz_path.resolve()]

    stem = "case_a"
    img_sub = imgs_root / stem
    meta_sub = meta_root / stem
    assert (img_sub / f"{stem}_0001.npy").is_file()
    assert (img_sub / f"{stem}_0002.npy").is_file()
    assert (meta_sub / f"{stem}_0001.json").is_file()
    assert (meta_sub / f"{stem}_0002.json").is_file()

    a1 = np.load(img_sub / f"{stem}_0001.npy")
    a2 = np.load(img_sub / f"{stem}_0002.npy")
    assert a1.shape == (32, 32)
    assert a2.shape == (32, 32)
    assert not np.allclose(a1, a2)

    m1 = json.loads((meta_sub / f"{stem}_0001.json").read_text(encoding="utf-8"))
    m2 = json.loads((meta_sub / f"{stem}_0002.json").read_text(encoding="utf-8"))
    assert m1["noise"]["seed"] == 1001
    assert m2["noise"]["seed"] == 1002
    assert m1["frame_index"] == 1
    assert m2["frame_index"] == 2
    assert m1["psf_sigma_px"] == 2.0
    assert str(npz_path.resolve()) == m1["traj_npz"]


def test_export_dry_run_lists_without_writing(tmp_path: Path) -> None:
    traj_dir = tmp_path / "t"
    traj_dir.mkdir()
    xy = np.zeros((3, 1, 2), dtype=np.float64)
    save_plane_trajectory_npz(traj_dir / "only.npz", xy, 1e-8, meta={}, write_mean_pos_zx=False)

    imaging = {
        "version": 1,
        "camera": {"pixel_um": 1.0, "l": 8, "h": 8, "x0_um": 0.0, "y0_um": 0.0},
        "beam": {"w_um": 100.0, "I": 100.0},
        "noise": {"seed": 1},
        "imaging": {"psf_sigma_px": 1.0, "normalize_mode": "none"},
        "display": {"show": False},
    }
    img_json = tmp_path / "img.json"
    img_json.write_text(json.dumps(imaging), encoding="utf-8")

    imgs_root = tmp_path / "Imgs"
    meta_root = tmp_path / "meta"
    export_ion_npy_from_traj_dir(
        img_json,
        traj_dir=traj_dir,
        imgs_root=imgs_root,
        meta_root=meta_root,
        project_root=tmp_path,
        dry_run=True,
    )
    assert not imgs_root.exists()
    assert not meta_root.exists()
