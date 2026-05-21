"""从重构概率估算背景气压"""
from __future__ import annotations

import numpy as np

from collision_pressure.species import Species, EC, EPS0, KB


def langevin_rate_coefficient(ion: Species, mol: Species) -> float:
    """Langevin collision rate coefficient k_L (m^3/s).

    k_L = e * sqrt(alpha * pi / (mu * eps0))

    The collision rate per ion is: gamma_el = n * k_L
    where n is the neutral gas number density (1/m^3).
    """
    mu = ion.mass_kg * mol.mass_kg / (ion.mass_kg + mol.mass_kg)
    return EC * np.sqrt(mol.polarizability * np.pi / (mu * EPS0))


def estimate_pressure(
    reconfig_prob: float,
    ion: Species,
    mol: Species,
    T: float,
    reconfig_rate: float | None = None,
) -> float:
    """Estimate background gas pressure from simulation + experiment.

    The simulation gives P_flip (fraction of collisions causing reconfiguration).
    The experiment gives R_obs (observed reconfiguration rate in 1/s).

    Derivation:
        collision rate per ion:   gamma_el = n * k_L
        reconfiguration rate:     R_obs = gamma_el * P_flip = n * k_L * P_flip
        ideal gas:                P = n * kB * T

        => P = R_obs * kB * T / (P_flip * k_L)

    Parameters
    ----------
    reconfig_prob : float — P_flip from simulation (dimensionless)
    ion, mol : Species
    T : float — molecule temperature (K)
    reconfig_rate : float | None — R_obs from experiment (1/s)
        If None, returns the *inverse Langevin rate* kB*T / (P_flip * k_L)
        which has units of Pa*s. Multiply by R_obs to get pressure.

    Returns
    -------
    float — pressure in Pa (if reconfig_rate given), or Pa*s otherwise.
    """
    if reconfig_prob <= 0:
        return float("inf")

    k_L = langevin_rate_coefficient(ion, mol)
    # P = R_obs * kB * T / (P_flip * k_L)
    # If R_obs unknown, return the coefficient: kB * T / (P_flip * k_L) [Pa*s]
    coefficient = KB * T / (reconfig_prob * k_L)

    if reconfig_rate is not None:
        return reconfig_rate * coefficient
    return coefficient
