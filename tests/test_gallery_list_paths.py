"""Tests for ImgSimulation.gallery path listing (no GUI)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def test_list_image_paths_sorts_and_filters(tmp_path: Path) -> None:
    from ImgSimulation.gallery import _IMAGE_EXTENSIONS, _list_image_paths

    d = tmp_path / "Imgs" / "a"
    d.mkdir(parents=True)
    (d / "z.png").write_bytes(b"")
    (d / "b.npy").parent.mkdir(exist_ok=True)
    np.save(d / "b.npy", np.zeros((2, 2), dtype=np.float32))
    (d / "skip.txt").write_text("x")
    (tmp_path / "Imgs" / "c.JPG").write_bytes(b"")
    sub = d / "sub"
    sub.mkdir()
    np.save(sub / "a.npy", np.zeros((1, 1)))

    paths = _list_image_paths(tmp_path / "Imgs")
    rel = [p.relative_to(tmp_path) for p in paths]
    assert rel == [
        Path("Imgs/a/b.npy"),
        Path("Imgs/a/sub/a.npy"),
        Path("Imgs/a/z.png"),
        Path("Imgs/c.JPG"),
    ]
    for p in paths:
        assert p.suffix.lower() in {e.lower() for e in _IMAGE_EXTENSIONS}

    assert _list_image_paths(tmp_path / "nope") == []


def test_traj_stem_for_image(tmp_path: Path) -> None:
    from ImgSimulation.gallery import _traj_stem_for_image

    root = tmp_path / "Imgs"
    (root / "run_a").mkdir(parents=True)
    f1 = root / "run_a" / "run_a_0003.npy"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"")
    assert _traj_stem_for_image(f1, root) == "run_a"

    f2 = root / "flat_0001.png"
    f2.write_bytes(b"")
    assert _traj_stem_for_image(f2, root) == "flat"
