"""碰撞力学：Langevin 碇撞速率、散射角、动量冲量"""
from __future__ import annotations

import numpy as np
from scipy.special import ellipk

from collision_pressure.species import (
    Species, reduced_mass, polarization_coefficient, EC, EPS0, KB,
)


def langevin_rate(ion: Species, mol: Species, T: float, n: float) -> float:
    """Langevin collision rate (1/s)

    Parameters
    ----------
    T : float — molecule temperature (K)
    n : float — number density of molecules (1/m^3)
    """
    mu = reduced_mass(ion, mol)
    return n * EC * np.sqrt(mol.polarizability * np.pi / (mu * EPS0))


def critical_impact_param(ion: Species, mol: Species, v0: float) -> float:
    """Critical impact parameter b_c (m)"""
    mu = reduced_mass(ion, mol)
    C4 = polarization_coefficient(ion, mol)
    return (8.0 * C4 / (mu * v0**2))**0.25


def scattering_angle(ion: Species, mol: Species, v0: float, b: float) -> float:
    """Scattering angle theta (rad)

    b < b_c: spiral orbit, theta = pi (backward scatter)
    b >= b_c: deflection orbit, theta from Binet equation
    """
    mu = reduced_mass(ion, mol)
    C4 = polarization_coefficient(ion, mol)
    bc = critical_impact_param(ion, mol, v0)

    if b < bc:
        return np.pi

    E = 0.5 * mu * v0**2
    L = mu * v0 * b
    x = np.sqrt(max(1.0 - 16.0 * C4 * mu**2 * E / L**4, 0.0))
    m_ellip = (1.0 - x) / (1.0 + x)
    theta = float(np.pi - 2.0 * np.sqrt(2.0) / np.sqrt(1.0 + x) * ellipk(m_ellip))
    return abs(theta)


def post_collision_kick(
    ion: Species,
    mol: Species,
    v0: float,
    theta: float,
    direction: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Velocity kick on the struck ion (m/s)

    Parameters
    ----------
    v0 : float — H2 speed (m/s)
    theta : float — scattering angle (rad)
    direction : (3,) unit vector — incoming molecule direction
    rng : Generator or None — if provided, azimuthal angle is randomized
          uniformly in [0, 2π); otherwise φ = 0 (backward compatible).

    Returns
    -------
    dv : (3,) velocity change in m/s
    """
    M_i = ion.mass_kg
    M_m = mol.mass_kg
    ratio = M_m / (M_i + M_m)

    dv_mag_parallel = v0 * (1.0 - np.cos(theta))
    dv_mag_perp = v0 * np.sin(theta)

    # Build orthonormal basis (e1, e2) in the plane perpendicular to direction
    z = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(direction, z)
    norm = np.linalg.norm(e1)
    if norm < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 /= norm
    e2 = np.cross(direction, e1)

    # Randomize azimuthal angle in the scattering plane
    phi = rng.uniform(0.0, 2.0 * np.pi) if rng is not None else 0.0
    perp = np.cos(phi) * e1 + np.sin(phi) * e2

    return ratio * (dv_mag_parallel * direction + dv_mag_perp * perp)
