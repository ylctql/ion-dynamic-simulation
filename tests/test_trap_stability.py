"""
tests for trap_stability module
"""
import math

import numpy as np
import pytest
from scipy.constants import e as EC, pi, u as AMU

from FieldConfiguration.ion_species import BA_135, CA_40, BE_9, ION_SPECIES
from trap_stability.stability import (
    StabilityResult,
    check_stability_region,
    compute_stability_direct,
    _secular_freq_mhz,
)


# ============== Direct mode tests ==============

class TestDirectMode:
    """Tests for textbook formula (direct mode) computation."""

    def test_linear_trap_known_values(self):
        """Ba135+, f_RF=35.28 MHz, r0=700 um, V0=275 V, U=1 V"""
        r = compute_stability_direct(
            rf_freq_MHz=35.28, r0_um=700, V0=275, U=1.0,
            species=BA_135, geometry="linear",
        )
        Omega = 2 * pi * 35.28e6
        r0 = 700e-6
        m = BA_135.mass_kg

        expected_q_x = 2 * EC * 275 / (m * r0**2 * Omega**2)
        assert math.isclose(r.q_x, expected_q_x, rel_tol=1e-10)
        assert math.isclose(r.q_y, -expected_q_x, rel_tol=1e-10)
        assert r.q_z == 0.0

        expected_a_z = 4 * EC * 1.0 / (m * r0**2 * Omega**2)
        assert math.isclose(r.a_z, expected_a_z, rel_tol=1e-10)
        assert math.isclose(r.a_x, -expected_a_z / 2, rel_tol=1e-10)

    def test_3d_trap(self):
        """3D Paul trap geometry"""
        r = compute_stability_direct(
            rf_freq_MHz=35.28, r0_um=700, V0=275, U=1.0,
            species=BA_135, geometry="3d",
        )
        Omega = 2 * pi * 35.28e6
        r0 = 700e-6
        m = BA_135.mass_kg

        expected_q_r = EC * 275 / (m * r0**2 * Omega**2)
        assert math.isclose(r.q_x, expected_q_r, rel_tol=1e-10)
        assert math.isclose(r.q_y, expected_q_r, rel_tol=1e-10)
        assert r.q_z == 0.0

    def test_zero_rf_unstable(self):
        """V0=0 -> q=0, a_x=a_y<0 -> unstable"""
        r = compute_stability_direct(
            rf_freq_MHz=35.28, r0_um=700, V0=0, U=1.0,
            species=BA_135,
        )
        assert r.q_x == 0.0
        assert not r.is_stable

    def test_species_mass_scaling(self):
        """Be9+ ~15x lighter than Ba135+ -> q ~15x larger"""
        r_ba = compute_stability_direct(
            rf_freq_MHz=35.28, r0_um=700, V0=275,
            species=BA_135,
        )
        r_be = compute_stability_direct(
            rf_freq_MHz=35.28, r0_um=700, V0=275,
            species=BE_9,
        )
        ratio = abs(r_be.q_x) / abs(r_ba.q_x)
        expected_ratio = BA_135.mass_amu / BE_9.mass_amu
        assert math.isclose(ratio, expected_ratio, rel_tol=1e-6)

    def test_zero_dc_stable_radial(self):
        """U=0 -> a_z=0, a_x=a_y=0: radial confinement from RF pseudo-potential"""
        r = compute_stability_direct(
            rf_freq_MHz=35.28, r0_um=700, V0=275, U=0,
            species=BA_135,
        )
        # With U=0: a_x=a_y=a_z=0, q_z=0
        # z: a+q^2/2 = 0 -> unstable (no axial confinement)
        assert not r.is_stable
        # x should have valid secular freq from q only
        assert not math.isnan(r.f_sec_x)

    def test_default_species(self):
        """species=None should default to Ba135+"""
        r = compute_stability_direct(35.28, 700, 275)
        assert r.species_name == "Ba135+"

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError):
            compute_stability_direct(35.28, 700, 275, geometry="cylinder")


# ============== Stability check tests ==============

class TestStabilityCheck:
    """Tests for check_stability_region."""

    def test_stable_point(self):
        is_stable, note = check_stability_region(
            a_x=0.0, a_y=0.0, a_z=0.001,
            q_x=0.3, q_y=-0.3, q_z=0.0,
        )
        assert is_stable
        assert "stable" in note

    def test_q_too_large_unstable(self):
        is_stable, note = check_stability_region(
            a_x=0.0, a_y=0.0, a_z=0.001,
            q_x=1.0, q_y=-1.0, q_z=0.0,
        )
        assert not is_stable

    def test_negative_a_plus_q2_unstable(self):
        """a + q^2/2 < 0 should be unstable"""
        is_stable, note = check_stability_region(
            a_x=-0.1, a_y=-0.1, a_z=0.2,
            q_x=0.3, q_y=-0.3, q_z=0.0,
        )
        # a_x + q_x^2/2 = -0.1 + 0.045 = -0.055 < 0
        assert not is_stable


# ============== Secular frequency helper tests ==============

class TestSecularFreq:
    def test_positive(self):
        Omega = 2 * pi * 35.28e6
        f = _secular_freq_mhz(a=0.001, q=0.3, Omega=Omega)
        assert f > 0
        assert not math.isnan(f)

    def test_negative_arg_returns_nan(self):
        f = _secular_freq_mhz(a=-0.1, q=0.3, Omega=1.0)
        # a + q^2/2 = -0.1 + 0.045 < 0
        assert math.isnan(f)


# ============== CLI parsing tests ==============

class TestCLI:
    """Tests for CLI argument parsing."""

    def test_create_parser(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        assert p is not None

    def test_direct_mode_parse(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args([
            "--direct", "--rf-freq", "35.28", "--r0", "700", "--V0", "275",
        ])
        assert args.direct is True
        assert args.rf_freq == 35.28
        assert args.r0 == 700.0
        assert args.V0 == 275.0

    def test_csv_mode_parse(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args(["--csv", "test.csv", "--config", "test.json"])
        assert args.csv == "test.csv"
        assert args.config == "test.json"
        assert args.direct is False

    def test_species_parse(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args([
            "--direct", "--rf-freq", "35.28", "--r0", "700", "--V0", "275",
            "--species", "Ca40+",
        ])
        assert args.species == "Ca40+"

    def test_default_species(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args([
            "--direct", "--rf-freq", "35.28", "--r0", "700", "--V0", "275",
        ])
        assert args.species == "Ba135+"

    def test_default_ranges(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args(["--csv", "test.csv"])
        assert args.x_range == "-50,50"
        assert args.y_range == "-20,20"
        assert args.z_range == "-150,150"

    def test_out_parse(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args([
            "--direct", "--rf-freq", "35.28", "--r0", "700", "--V0", "275",
            "--out", "result.json",
        ])
        assert args.out == "result.json"


# ============== Integration: direct mode consistency ==============

class TestDirectConsistency:
    """Verify secular frequencies are consistent with a, q parameters."""

    def test_secular_freq_matches_a_q(self):
        """f_sec from result should match Omega/2*sqrt(a+q^2/2)/(2*pi)"""
        r = compute_stability_direct(
            rf_freq_MHz=35.28, r0_um=700, V0=275, U=1.0,
            species=BA_135,
        )
        Omega = r.omega_rf
        for axis in ("x", "y", "z"):
            a = getattr(r, f"a_{axis}")
            q = getattr(r, f"q_{axis}")
            f_sec = getattr(r, f"f_sec_{axis}")
            arg = a + q**2 / 2
            if arg > 0:
                expected = (Omega / 2) * math.sqrt(arg) / (2 * pi * 1e6)
                assert math.isclose(f_sec, expected, rel_tol=1e-10), (
                    f"axis {axis}: f_sec={f_sec}, expected={expected}"
                )
            else:
                assert math.isnan(f_sec)
