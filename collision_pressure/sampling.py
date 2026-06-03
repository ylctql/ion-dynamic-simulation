"""蒙特卡洛采样：分子速度、碰撞参数、碰撞离子、入射方向"""
from __future__ import annotations

import numpy as np

from collision_pressure.species import KB, AMU


def sample_velocity(T: float, mass_amu: float, rng: np.random.Generator) -> float:
    """Maxwell-Boltzmann speed sampling (m/s)

    Uses the 3D Maxwell speed distribution:
        f(v) ~ v^2 * exp(-m*v^2 / (2*kB*T))
    Sampled via Gamma(3/2, kB*T/m) then sqrt.
    """
    m_kg = mass_amu * AMU
    v2 = rng.gamma(1.5, 2.0 * KB * T / m_kg)  # v^2 ~ Gamma(3/2, 2*kB*T/m)
    return float(np.sqrt(v2))


def sample_impact_parameter(b_max: float, rng: np.random.Generator) -> float:
    """Area-weighted sampling of impact parameter b in [0, b_max] (m).

    Differential cross-section dσ/db = 2πb, so the CDF is (b/b_max)².
    Sampling b = b_max * sqrt(U) with U ~ Uniform(0,1) gives the correct
    area-weighted distribution.
    """
    return float(b_max * np.sqrt(rng.uniform(0.0, 1.0)))


def sample_collision_ion(N: int, rng: np.random.Generator) -> int:
    """Randomly select an ion index (uniform)"""
    return int(rng.integers(0, N))


def sample_direction(rng: np.random.Generator) -> np.ndarray:
    """Uniform random direction on the unit sphere"""
    cos_theta = rng.uniform(-1.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
