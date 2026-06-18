"""
tests for field_visualize trap-frequency / harmonicity output.

Covers quadratic_fit_r2 (the harmonicity indicator) with synthetic data, and the
end-to-end enrichment of compute_trap_freqs_at_point on real field geometry.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from field_visualize.trap_freq import (
    compute_trap_freqs_at_point,
    quadratic_fit_r2,
)


# ============== quadratic_fit_r2 unit tests (no CSV) ==============

class TestQuadraticFitR2:
    """Synthetic-data tests for the harmonicity indicator."""

    def test_pure_quadratic_is_perfect(self):
        # 纯抛物线 → 纯二次模型完美拟合，R² ≈ 1
        x = np.linspace(-50.0, 50.0, 200)
        V = 0.5 + 0.02 * x + 3.0e-3 * x**2
        r2 = quadratic_fit_r2(x, V)
        assert math.isclose(r2, 1.0, rel_tol=1e-9, abs_tol=1e-12)

    def test_quartic_perturbation_lowers_r2(self):
        # 加入四次非谐项后 R² 应下降，且四次项越强 R² 越低
        x = np.linspace(-50.0, 50.0, 200)
        base = 3.0e-3 * x**2
        r2_pure = quadratic_fit_r2(x, base)
        r2_weak = quadratic_fit_r2(x, base + 1.0e-7 * x**4)
        r2_strong = quadratic_fit_r2(x, base + 1.0e-6 * x**4)
        assert r2_weak < r2_pure
        assert r2_strong < r2_weak
        # 非谐性不至极端时 R² 仍为正常概率比（≤1）
        assert r2_weak <= 1.0 + 1e-9

    def test_failure_returns_nan(self):
        # 点数不足以做二次拟合 → ValueError → 返回 NaN
        x = np.array([0.0, 1.0])
        V = np.array([0.0, 1.0])
        r2 = quadratic_fit_r2(x, V)
        assert math.isnan(r2)


# ============== compute_trap_freqs_at_point enrichment ==============

@pytest.mark.skipif(
    not Path("data/monolithic20241118.csv").is_file(),
    reason="monolithic CSV not available",
)
class TestFreqHarmonicityIntegration:
    """Integration test on real field geometry (when CSV present)."""

    def _compute(self):
        from FieldConfiguration.constants import init_from_config
        from FieldConfiguration.ion_species import BA_135
        from FieldConfiguration.loader import build_voltage_list
        from FieldParser.calc_field import calc_field, calc_potential
        from FieldParser.csv_reader import read as read_csv
        from field_visualize.core import apply_savgol_smooth

        cfg, config_dict = init_from_config(
            "FieldConfiguration/configs/default.json", mass_amu=BA_135.mass_amu
        )
        grid_coord, grid_voltage = read_csv(
            "data/monolithic20241118.csv", None, normalize=True, dl=cfg.dl, dV=cfg.dV
        )
        grid_voltage = apply_savgol_smooth(
            grid_coord, grid_voltage, ("z",), window_length=11, polyorder=3
        )
        potential_interps = calc_potential(grid_coord, grid_voltage)
        field_interps = calc_field(grid_coord, grid_voltage)
        voltage_list = build_voltage_list(config_dict, grid_voltage.shape[1], cfg)
        freqs = compute_trap_freqs_at_point(
            potential_interps,
            field_interps,
            voltage_list,
            cfg,
            0.0, 0.0, 0.0,
            x_range_um=(-100.0, 100.0),
            y_range_um=(-20.0, 20.0),
            z_range_um=(-150.0, 150.0),
            n_pts=200,
            fit_degree=2,
        )
        return freqs, cfg

    def test_harmonicity_keys_present_and_finite(self):
        freqs, _ = self._compute()
        for axis in ("x", "y", "z"):
            assert f"r2_quad_{axis}" in freqs
            r2 = freqs[f"r2_quad_{axis}"]
            assert math.isfinite(r2)
            # R² 为概率比指标，正常拟合不超过 1（留少量数值余量）
            assert r2 <= 1.0 + 1e-6

    def test_trap_freq_keys_still_present(self):
        # 加键不得破坏既有 f_x/f_y/f_z
        freqs, _ = self._compute()
        for axis in ("x", "y", "z"):
            assert f"f_{axis}" in freqs

    def test_narrower_range_more_harmonic(self):
        # 范围越小，非谐边缘效应越弱，纯二次 R² 越接近 1（range-aware 验证）
        from FieldConfiguration.constants import init_from_config
        from FieldConfiguration.ion_species import BA_135
        from FieldConfiguration.loader import build_voltage_list
        from FieldParser.calc_field import calc_field, calc_potential
        from FieldParser.csv_reader import read as read_csv
        from field_visualize.core import apply_savgol_smooth

        cfg, config_dict = init_from_config(
            "FieldConfiguration/configs/default.json", mass_amu=BA_135.mass_amu
        )
        grid_coord, grid_voltage = read_csv(
            "data/monolithic20241118.csv", None, normalize=True, dl=cfg.dl, dV=cfg.dV
        )
        grid_voltage = apply_savgol_smooth(
            grid_coord, grid_voltage, ("z",), window_length=11, polyorder=3
        )
        potential_interps = calc_potential(grid_coord, grid_voltage)
        field_interps = calc_field(grid_coord, grid_voltage)
        voltage_list = build_voltage_list(config_dict, grid_voltage.shape[1], cfg)

        common = dict(
            potential_interps=potential_interps,
            field_interps=field_interps,
            voltage_list=voltage_list,
            cfg=cfg,
            yc_um=0.0, zc_um=0.0,
            y_range_um=(-20.0, 20.0), z_range_um=(-150.0, 150.0),
            n_pts=200, fit_degree=2,
        )
        wide = compute_trap_freqs_at_point(xc_um=0.0, x_range_um=(-100.0, 100.0), **common)
        narrow = compute_trap_freqs_at_point(xc_um=0.0, x_range_um=(-20.0, 20.0), **common)
        # 窄范围的 x 轴应比宽范围更接近纯谐性
        assert narrow["r2_quad_x"] >= wide["r2_quad_x"] - 1e-9
