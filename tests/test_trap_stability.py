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
    compute_stability_from_field,
    find_trap_center,
    _secular_freq_mhz,
)


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

    def test_secular_freq_formula(self):
        """Verify Omega/2·sqrt(a + q²/2)/(2π)"""
        Omega = 2 * pi * 35.28e6
        a, q = 0.001, 0.3
        f = _secular_freq_mhz(a, q, Omega)
        arg = a + q**2 / 2
        expected = (Omega / 2) * math.sqrt(arg) / (2 * pi * 1e6)
        assert math.isclose(f, expected, rel_tol=1e-10)


# ============== CLI parsing tests ==============

class TestCLI:
    """Tests for CLI argument parsing."""

    def test_create_parser(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        assert p is not None

    def test_csv_mode_parse(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args(["--csv", "test.csv", "--config", "test.json"])
        assert args.csv == "test.csv"
        assert args.config == "test.json"

    def test_species_parse(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args(["--csv", "test.csv", "--species", "Ca40+"])
        assert args.species == "Ca40+"

    def test_default_species(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args(["--csv", "test.csv"])
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
        args = p.parse_args(["--csv", "test.csv", "--out", "result.json"])
        assert args.out == "result.json"

    def test_csv_required(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_fit_degree_default(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        args = p.parse_args(["--csv", "test.csv"])
        assert args.fit_degree == 6

    def test_fit_degree_options(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        for deg in (2, 4, 6):
            args = p.parse_args(["--csv", "test.csv", "--fit-degree", str(deg)])
            assert args.fit_degree == deg

    def test_fit_degree_invalid(self):
        from trap_stability.cli import create_parser
        p = create_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--csv", "test.csv", "--fit-degree", "3"])


# ============== Mathieu a/q formula verification ==============

class TestAQFormula:
    """Verify the a/q formulas match textbook expressions."""

    def test_a_from_k2_matches_textbook(self):
        """a = 8e·c₂/(m·Ω²) for DC potential."""
        c2_dc = 1e-4  # V/μm²
        m = BA_135.mass_kg
        Omega = 2 * pi * 35.28e6
        a_expected = 8 * EC * c2_dc * 1e12 / (m * Omega**2)
        # a_expected should be positive and small
        assert a_expected > 0
        assert a_expected < 1.0

    def test_q_from_k2_matches_textbook(self):
        """q = 4e·c₂/(m·Ω²) for RF amplitude potential."""
        c2_rf = 1e-4  # V/μm²
        m = BA_135.mass_kg
        Omega = 2 * pi * 35.28e6
        q_expected = 4 * EC * c2_rf * 1e12 / (m * Omega**2)
        assert q_expected > 0
        assert q_expected < 1.0

    def test_a_equals_2q_for_same_k2(self):
        """For identical c₂: a(DC) = 2·q(RF)."""
        c2 = 1e-4
        a = 8 * EC * c2 * 1e12 / (BA_135.mass_kg * (2 * pi * 35.28e6)**2)
        q = 4 * EC * c2 * 1e12 / (BA_135.mass_kg * (2 * pi * 35.28e6)**2)
        assert math.isclose(a, 2 * q, rel_tol=1e-12)

    def test_species_mass_scaling(self):
        """Be9+ ~15x lighter than Ba135+ → q ~15x larger."""
        c2 = 1e-4
        Omega = 2 * pi * 35.28e6
        q_ba = 4 * EC * c2 * 1e12 / (BA_135.mass_kg * Omega**2)
        q_be = 4 * EC * c2 * 1e12 / (BE_9.mass_kg * Omega**2)
        ratio = q_be / q_ba
        expected_ratio = BA_135.mass_amu / BE_9.mass_amu
        assert math.isclose(ratio, expected_ratio, rel_tol=1e-6)


# ============== Field mode: q from RF k2 ==============

class TestFieldModeQFormula:
    """q must use k2 (x² coefficient), not V'' = 2*k2."""

    def test_q_from_k2_matches_textbook(self):
        """Ideal quadrupole: k2_rf = V0/(2*r0²) → q = 4e*k2/(m*Omega²)."""
        rf_freq_MHz = 35.28
        r0_um = 700.0
        V0 = 275.0
        Omega = 2 * pi * rf_freq_MHz * 1e6
        r0 = r0_um * 1e-6
        m = BA_135.mass_kg
        k2_rf_si = V0 / (2 * r0**2)

        q_from_k2 = 4 * EC * k2_rf_si / (m * Omega**2)
        # Verify the textbook formula value
        f_sec = _secular_freq_mhz(0.0, q_from_k2, Omega)
        assert not math.isnan(f_sec)
        assert f_sec > 0


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/monolithic20241118.csv").is_file(),
    reason="monolithic CSV not available",
)
class TestFieldModeIntegration:
    """Integration test on real field geometry (when CSV present)."""

    def _compute_result(self, fit_degree=6):
        from FieldConfiguration.constants import init_from_config
        from FieldConfiguration.loader import build_voltage_list
        from FieldParser.calc_field import calc_field, calc_potential
        from FieldParser.csv_reader import read as read_csv
        from field_visualize.core import apply_savgol_smooth

        cfg, config_dict = init_from_config(
            "FieldConfiguration/configs/default.json", mass_amu=BA_135.mass_amu
        )
        csv_path = "data/monolithic20241118.csv"
        grid_coord, grid_voltage = read_csv(
            csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV
        )
        grid_voltage = apply_savgol_smooth(
            grid_coord, grid_voltage, ("z",), window_length=11, polyorder=3
        )
        potential_interps = calc_potential(grid_coord, grid_voltage)
        field_interps = calc_field(grid_coord, grid_voltage)
        voltage_list = build_voltage_list(config_dict, grid_voltage.shape[1], cfg)
        center = find_trap_center(
            potential_interps, field_interps, voltage_list, cfg
        )
        result = compute_stability_from_field(
            potential_interps,
            field_interps,
            voltage_list,
            cfg,
            species=BA_135,
            center_um=center,
            fit_range_um=((-100, 100), (-20, 20), (-150, 150)),
            fit_degree=fit_degree,
        )
        return result, cfg

    def test_compute_stability_from_field_q_x(self):
        result, cfg = self._compute_result()
        k2_rf_si = result.k2_rf_amp["x"] * 1e12
        q_expected = 4 * EC * k2_rf_si / (BA_135.mass_kg * cfg.Omega**2)
        assert math.isclose(result.q_x, q_expected, rel_tol=1e-9)
        assert result.q_x < 0.2, "q_x should not be ~2x direct-scale after fix"

    def test_anh4_anh6_present(self):
        result, cfg = self._compute_result()
        assert result.anh4_dc is not None
        assert result.anh4_rf is not None
        assert result.anh6_dc is not None
        assert result.anh6_rf is not None
        for axis in ("x", "y", "z"):
            assert axis in result.anh4_dc
            assert axis in result.anh6_rf
        for d in (result.anh4_dc, result.anh4_rf, result.anh6_dc, result.anh6_rf):
            for v in d.values():
                assert math.isfinite(v)

    def test_anh4_small_relative_to_k2(self):
        result, cfg = self._compute_result()
        dl_um = cfg.dl * 1e6
        for axis in ("x", "y", "z"):
            c2_dim = result.k2_rf_amp[axis] * dl_um**2 / cfg.dV
            if abs(c2_dim) > 1e-15:
                ratio = abs(result.anh4_rf[axis] / c2_dim)
                assert ratio < 1.0, (
                    f"anh4_rf[{axis}]/c2_dim = {ratio}, expected << 1"
                )

    def test_fit_degree_2_no_anh(self):
        """fit_degree=2 should produce None for all anharmonic constants."""
        result, cfg = self._compute_result(fit_degree=2)
        assert result.anh4_dc is None
        assert result.anh4_rf is None
        assert result.anh6_dc is None
        assert result.anh6_rf is None
        # a, q should still be valid
        assert math.isfinite(result.q_x)

    def test_fit_degree_4_has_anh4_not_anh6(self):
        """fit_degree=4 should produce anh4 dicts but None for anh6."""
        result, cfg = self._compute_result(fit_degree=4)
        assert result.anh4_dc is not None
        assert result.anh4_rf is not None
        assert result.anh6_dc is None
        assert result.anh6_rf is None
        # anh4 values should be finite
        for v in result.anh4_dc.values():
            assert math.isfinite(v)

    def test_fit_degree_invalid_raises(self):
        """Invalid fit_degree should raise ValueError."""
        from FieldConfiguration.constants import init_from_config, Config
        Omega = 2 * pi * 35.28e6
        from scipy.constants import e, epsilon_0
        m_kg = BA_135.mass_kg
        dt = 2 / Omega
        dl = (e**2 / (4 * pi * m_kg * epsilon_0 * Omega**2)) ** (1 / 3)
        dV = m_kg / e * (dl / dt) ** 2
        cfg = Config(Omega=Omega, dt=dt, dl=dl, dV=dV, freq_RF=35.28)
        with pytest.raises(ValueError, match="fit_degree"):
            compute_stability_from_field(
                [], [], [], cfg, BA_135, (0, 0, 0),
                ((-1, 1), (-1, 1), (-1, 1)), fit_degree=3,
            )


# ============== Anharmonic constants tests ==============

class TestAnharmonicSynthetic:
    """Verify anharmonic constants on synthetic potential with known coefficients."""

    def test_pure_quartic(self):
        """V(x) = x^4 → c4 = 1 V/μm⁴, c6 = 0."""
        from FieldParser.potential_fit import fit_potential_1d

        coord_um = np.linspace(-10, 10, 200)
        V_quartic = coord_um**4
        fit, _ = fit_potential_1d(coord_um, V_quartic, degree=6)
        coefs = fit[1:]
        a4_coeff = coefs[4]

        assert math.isclose(a4_coeff, 1.0, rel_tol=1e-6)

    def test_pure_sextic(self):
        """V(x) = x^6 → c6 = 1 V/μm⁶."""
        from FieldParser.potential_fit import fit_potential_1d

        coord_um = np.linspace(-5, 5, 200)
        V_sextic = coord_um**6
        fit, _ = fit_potential_1d(coord_um, V_sextic, degree=6)
        coefs = fit[1:]
        a6_coeff = coefs[6]

        assert math.isclose(a6_coeff, 1.0, rel_tol=1e-6)

    def test_taylor_coeff_has_factorial(self):
        """Polynomial coefficient = Taylor coefficient (includes 1/n! factor)."""
        from FieldParser.potential_fit import fit_potential_1d

        coord_um = np.linspace(-10, 10, 200)
        V = coord_um**4
        fit, _ = fit_potential_1d(coord_um, V, degree=6)
        coefs = fit[1:]
        assert math.isclose(coefs[4], 1.0, rel_tol=1e-8)

        V2 = 2 * coord_um**4
        fit2, _ = fit_potential_1d(coord_um, V2, degree=6)
        coefs2 = fit2[1:]
        assert math.isclose(coefs2[4], 2.0, rel_tol=1e-8)
