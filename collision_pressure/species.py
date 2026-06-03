"""离子和中性分子的物理参数"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from FieldConfiguration.constants import BA135_MASS_AMU

AMU = 1.66053906660e-27       # kg
EPS0 = 8.854187817e-12        # F/m
EC = 1.602176634e-19          # C
KB = 1.380649e-23             # J/K


@dataclass(frozen=True)
class Species:
    name: str
    mass_amu: float
    polarizability: float      # SI, m^3 (only for neutral molecules)
    charge_ec: float = 1.0

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * AMU

    @property
    def charge_C(self) -> float:
        return self.charge_ec * EC


H2_MOLECULE = Species("H2", 2.016, polarizability=8.04e-31, charge_ec=0.0)
HE_MOLECULE = Species("He", 4.003, polarizability=2.07e-31, charge_ec=0.0)

BA_135 = Species("Ba135+", BA135_MASS_AMU, polarizability=0.0, charge_ec=1.0)
BA_138 = Species("Ba138+", 138.0, polarizability=0.0, charge_ec=1.0)
YB_171 = Species("Yb171+", 171.0, polarizability=0.0, charge_ec=1.0)

ION_SPECIES: dict[str, Species] = {
    "Ba135+": BA_135, "Ba138+": BA_138, "Yb171+": YB_171,
}
MOL_SPECIES: dict[str, Species] = {
    "H2": H2_MOLECULE, "He": HE_MOLECULE,
}


def reduced_mass(ion: Species, mol: Species) -> float:
    return ion.mass_kg * mol.mass_kg / (ion.mass_kg + mol.mass_kg)


def polarization_coefficient(ion: Species, mol: Species) -> float:
    """C_4 = alpha * q^2 / (2 * (4*pi*eps0)^2)"""
    q = ion.charge_C
    return mol.polarizability * q**2 / (2.0 * (4.0 * np.pi * EPS0)**2)
