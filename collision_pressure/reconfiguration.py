"""结构重构检测：Zigzag flip 检测器 + 通用拓扑检测器"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ReconfigResult:
    reconfigured: bool
    topology_changed: bool = False
    geometry_changed: bool = False
    ssd_um2: float = 0.0


class ReconfigDetector(ABC):
    @abstractmethod
    def register_equilibrium(self, r0: np.ndarray) -> None: ...

    @abstractmethod
    def check(self, r_final: np.ndarray) -> ReconfigResult: ...


class ZigzagFlipDetector(ReconfigDetector):
    """Detect zigzag flip via sort-by-axial + sum-squared-distances.

    Based on Appendix C of Pagano et al., Quantum Sci. Technol. 4 (2019) 014004.
    Sort initial and final positions by the axial (chain) coordinate, then
    compute the sum of squared Euclidean distances between corresponding ions.
    A zigzag flip (zig → zag) produces a large SSD; no flip gives a small SSD.

    Parameters
    ----------
    flip_axis : int
        Transverse axis where zigzag oscillation occurs (0=x, 1=y, 2=z).
    sort_axis : int
        Axial (chain) direction used for sorting (0=x, 1=y, 2=z).
    threshold_um2 : float | None
        Manual SSD threshold (μm²). None → auto-compute from equilibrium.
    auto_threshold_factor : float
        Auto-threshold = factor × Σ(flip_axis_i²).
        Default 2.0 is midway between no-flip (~0) and full-flip (~4Σy²).
    """

    def __init__(
        self,
        flip_axis: int = 1,
        sort_axis: int = 2,
        threshold_um2: float | None = None,
        auto_threshold_factor: float = 2.0,
    ):
        self._flip_axis = flip_axis
        self._sort_axis = sort_axis
        self._threshold_um2 = threshold_um2
        self._auto_factor = auto_threshold_factor
        self._r0_sorted: np.ndarray | None = None
        self._threshold: float = 0.0

    @property
    def threshold(self) -> float:
        """Computed SSD threshold (μm²)."""
        return self._threshold

    def register_equilibrium(self, r0: np.ndarray) -> None:
        idx = np.argsort(r0[:, self._sort_axis])
        self._r0_sorted = r0[idx].copy()

        if self._threshold_um2 is not None:
            self._threshold = self._threshold_um2
        else:
            y_sq = self._r0_sorted[:, self._flip_axis] ** 2
            self._threshold = self._auto_factor * float(np.sum(y_sq))

    def check(self, r_final: np.ndarray) -> ReconfigResult:
        if self._r0_sorted is None:
            raise RuntimeError("register_equilibrium not called")

        idx_final = np.argsort(r_final[:, self._sort_axis])
        r_final_sorted = r_final[idx_final]

        diff = r_final_sorted - self._r0_sorted
        ssd = float(np.sum(diff ** 2))

        reconfigured = ssd > self._threshold
        return ReconfigResult(
            reconfigured=reconfigured,
            geometry_changed=reconfigured,
            ssd_um2=ssd,
        )


class TopologyDetector(ReconfigDetector):
    """General topology-based reconfiguration detector.

    Compares Delaunay-derived topology of initial and final configurations.
    """

    def __init__(self, plane: str = "xoz", edge_filter_factor: float = 1.5):
        self._plane = plane
        self._edge_filter = edge_filter_factor
        self._topo = None

    def register_equilibrium(self, r0: np.ndarray) -> None:
        from collision_pressure.topology import build_topology
        self._topo = build_topology(r0, plane=self._plane,
                                    edge_filter_factor=self._edge_filter)

    def check(self, r_final: np.ndarray) -> ReconfigResult:
        from collision_pressure.topology import build_topology, same_topology
        if self._topo is None:
            raise RuntimeError("register_equilibrium not called")
        topo_final = build_topology(r_final, plane=self._plane,
                                    edge_filter_factor=self._edge_filter)
        changed = not same_topology(self._topo, topo_final)
        return ReconfigResult(
            reconfigured=changed,
            topology_changed=changed,
        )
