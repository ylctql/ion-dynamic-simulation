"""
collision_pressure — H2 elastic collision pressure estimation

Subcommands:
  python -m collision_pressure scan          Build configuration library
  python -m collision_pressure simulate      Run collision simulation
"""

from collision_pressure.topology import (
    CrystalTopology,
    TopologyFingerprint,
    build_topology,
    same_topology,
)
from collision_pressure.config_scan import (
    ConfigResult,
    ConfigurationLibrary,
    setup_fit,
    setup_fit_harmonic,
)
from collision_pressure.species import Species, BA_135, H2_MOLECULE
from collision_pressure.collision import scattering_angle, post_collision_kick
from collision_pressure.sampling import sample_velocity
from collision_pressure.reconfiguration import (
    ReconfigDetector,
    ZigzagFlipDetector,
    TopologyDetector,
)
from collision_pressure.simulation import run_single_collision, run_collision_scan
from collision_pressure.pressure import estimate_pressure, langevin_rate_coefficient
from collision_pressure.visualize_collision import (
    plot_trajectory_snapshots,
    plot_before_after,
    plot_batch_statistics,
)

__all__ = [
    "CrystalTopology", "TopologyFingerprint", "build_topology", "same_topology",
    "ConfigResult", "ConfigurationLibrary", "setup_fit", "setup_fit_harmonic",
    "Species", "BA_135", "H2_MOLECULE",
    "scattering_angle", "post_collision_kick", "sample_velocity",
    "ReconfigDetector", "ZigzagFlipDetector", "TopologyDetector",
    "run_single_collision", "run_collision_scan",
    "estimate_pressure", "langevin_rate_coefficient",
    "plot_trajectory_snapshots", "plot_before_after", "plot_batch_statistics",
]
