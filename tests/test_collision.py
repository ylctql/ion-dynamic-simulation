"""Tests for collision_pressure Phase 1 modules"""
import numpy as np
import pytest

from collision_pressure.species import (
    BA_135, H2_MOLECULE, reduced_mass, polarization_coefficient,
)
from collision_pressure.collision import (
    critical_impact_param, scattering_angle, post_collision_kick,
)
from collision_pressure.sampling import (
    sample_velocity, sample_impact_parameter, sample_collision_ion, sample_direction,
)


class TestSpecies:
    def test_reduced_mass(self):
        mu = reduced_mass(BA_135, H2_MOLECULE)
        m1, m2 = BA_135.mass_kg, H2_MOLECULE.mass_kg
        expected = m1 * m2 / (m1 + m2)
        assert abs(mu - expected) < 1e-30

    def test_polarization_coefficient_positive(self):
        C4 = polarization_coefficient(BA_135, H2_MOLECULE)
        assert C4 > 0


class TestCollision:
    def test_scattering_spiral(self):
        """b << b_c should give theta ~ pi"""
        v0 = 100.0  # m/s (slow molecule at ~10K)
        bc = critical_impact_param(BA_135, H2_MOLECULE, v0)
        theta = scattering_angle(BA_135, H2_MOLECULE, v0, 0.1 * bc)
        assert abs(theta - np.pi) < 0.01

    def test_scattering_forward(self):
        """b >> b_c should give small theta (weak deflection)"""
        v0 = 100.0
        bc = critical_impact_param(BA_135, H2_MOLECULE, v0)
        theta_large = scattering_angle(BA_135, H2_MOLECULE, v0, 10.0 * bc)
        theta_vlarge = scattering_angle(BA_135, H2_MOLECULE, v0, 100.0 * bc)
        assert theta_large > theta_vlarge  # deflection decreases with b
        assert theta_vlarge < theta_large

    def test_scattering_monotone(self):
        """theta should decrease as b increases"""
        v0 = 100.0
        bc = critical_impact_param(BA_135, H2_MOLECULE, v0)
        bs = np.linspace(0.5 * bc, 5.0 * bc, 20)
        thetas = [scattering_angle(BA_135, H2_MOLECULE, v0, b) for b in bs]
        for i in range(len(thetas) - 1):
            assert thetas[i] >= thetas[i + 1]

    def test_kick_zero_angle(self):
        """theta=0 should give zero kick"""
        direction = np.array([1.0, 0.0, 0.0])
        dv = post_collision_kick(BA_135, H2_MOLECULE, 100.0, 0.0, direction)
        assert np.allclose(dv, 0.0, atol=1e-15)

    def test_kick_backward(self):
        """theta=pi should give non-zero kick"""
        direction = np.array([1.0, 0.0, 0.0])
        dv = post_collision_kick(BA_135, H2_MOLECULE, 100.0, np.pi, direction)
        assert np.linalg.norm(dv) > 0


class TestSampling:
    def test_velocity_positive(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            v = sample_velocity(10.0, 2.016, rng)
            assert v > 0

    def test_velocity_scale(self):
        """At T=10K, H2 mean speed should be ~few hundred m/s"""
        rng = np.random.default_rng(42)
        vs = [sample_velocity(10.0, 2.016, rng) for _ in range(1000)]
        mean_v = np.mean(vs)
        # Maxwell mean speed = sqrt(8*kB*T/(pi*m))
        m = 2.016 * 1.66053906660e-27
        expected = np.sqrt(8 * 1.380649e-23 * 10.0 / (np.pi * m))
        assert abs(mean_v - expected) / expected < 0.1

    def test_impact_parameter_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            b = sample_impact_parameter(1e-6, rng)
            assert 0 <= b <= 1e-6

    def test_collision_ion_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            idx = sample_collision_ion(10, rng)
            assert 0 <= idx < 10

    def test_direction_unit(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            d = sample_direction(rng)
            assert abs(np.linalg.norm(d) - 1.0) < 1e-12
