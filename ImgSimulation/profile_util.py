"""Lightweight wall-clock profiling for ImgSimulation (env ``IMG_SIM_PROFILE``)."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator


def profiling_enabled() -> bool:
    v = os.environ.get("IMG_SIM_PROFILE", "").strip().lower()
    return v in ("1", "true", "yes")


@contextmanager
def profile_section(
    label: str,
    *,
    times: dict[str, float] | None = None,
) -> Iterator[None]:
    """Accumulate seconds under *label* in *times* if profiling is on."""
    if not profiling_enabled() or times is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        times[label] = times.get(label, 0.0) + (time.perf_counter() - t0)


def print_profile_summary(times: dict[str, float], *, prefix: str = "[ImgSimulation profile]") -> None:
    if not times:
        return
    total = sum(times.values())
    parts = [f"{k}={v:.4f}s" for k, v in sorted(times.items(), key=lambda x: -x[1])]
    print(f"{prefix} total={total:.4f}s " + " ".join(parts), flush=True)
