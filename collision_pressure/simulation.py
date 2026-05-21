"""单次/批量碰撞模拟：采样 → 碰撞力学 → 轨迹积分 → 重构检测"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.constants import e as EC
from scipy.integrate import solve_ivp

from collision_pressure.species import Species, AMU, reduced_mass
from collision_pressure.collision import (
    critical_impact_param, scattering_angle, post_collision_kick,
)
from collision_pressure.sampling import (
    sample_velocity, sample_impact_parameter, sample_collision_ion, sample_direction,
)
from collision_pressure.reconfiguration import ReconfigDetector


def thermalize(
    r0_um: np.ndarray,
    fit,
    mass_amu: float = 135.0,
    softening_um: float = 0.001,
    t_thermalize_us: float = 10.0,
    gamma_damping_per_s: float = 5e5,
    perturbation_um: float = 0.01,
    rng: np.random.Generator | None = None,
    n_steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve crystal with damping to let it settle from approximate equilibrium.

    Adds a small random perturbation to break symmetry, then integrates the
    equations of motion with Doppler-cooling damping.  The crystal relaxes
    into a true dynamical equilibrium with small but nonzero thermal motion.

    Returns (r_thermalized, v_thermalized) in um and um/us (= m/s).
    """
    from equilibrium.energy import total_energy_and_grad

    N = r0_um.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    r_start = r0_um + rng.normal(0, perturbation_um, r0_um.shape)
    charge_ec = np.ones(N)
    mass_kg = mass_amu * AMU
    accel_factor = EC / mass_kg
    gamma_us = gamma_damping_per_s * 1e-6

    y0 = np.zeros(6 * N)
    y0[:3 * N] = r_start.ravel()

    def rhs(t, y):
        r = y[:3 * N].reshape(N, 3)
        v = y[3 * N:].reshape(N, 3)
        _, grad = total_energy_and_grad(fit, r, charge_ec, softening_um)
        a = -accel_factor * grad - gamma_us * v
        return np.concatenate([v.ravel(), a.ravel()])

    t_span = (0.0, t_thermalize_us)
    t_eval = np.linspace(0.0, t_thermalize_us, n_steps)
    sol = solve_ivp(rhs, t_span, y0, method="RK45", t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)

    r_final = sol.y[:3 * N, -1].reshape(N, 3)
    v_final = sol.y[3 * N:, -1].reshape(N, 3)
    return r_final, v_final


@dataclass
class CollisionResult:
    reconfigured: bool
    v0: float               # H2 speed (m/s)
    b: float                 # impact parameter (m)
    theta: float             # scattering angle (rad)
    hit_ion: int             # struck ion index
    dv: np.ndarray           # (3,) kick velocity (m/s)
    r_final: np.ndarray | None = None      # (N, 3) final positions in um
    trajectory: np.ndarray | None = None   # (6N, n_steps) [r; v] in um, um/us
    time_us: np.ndarray | None = None      # (n_steps,) time in us


@dataclass
class CollisionScanResult:
    n_total: int
    n_reconfigured: int
    reconfig_prob: float
    collisions: list[CollisionResult]
    r0_um: np.ndarray
    v0_um_us: np.ndarray | None = None   # (N, 3) initial ion velocities


def run_single_collision(
    r0_um: np.ndarray,
    fit,
    ion: Species,
    mol: Species,
    T: float,
    rng: np.random.Generator,
    detector: ReconfigDetector,
    mass_amu: float = 135.0,
    softening_um: float = 0.001,
    t_integrate_us: float = 50.0,
    n_steps: int = 2000,
    save_trajectory: bool = False,
    v_init_um_us: np.ndarray | None = None,
    gamma_damping_per_s: float = 0.0,
) -> CollisionResult:
    """Run a single collision event + trajectory integration.

    Parameters
    ----------
    r0_um : (N, 3) equilibrium positions in um
    fit : FitResult3D from setup_fit / setup_fit_harmonic
    T : molecule temperature (K)
    t_integrate_us : integration time in microseconds
    n_steps : number of ODE output steps
    v_init_um_us : (N, 3) initial ion velocities in um/us (= m/s numerically).
                   If None, ions start at rest.
    gamma_damping_per_s : Doppler cooling damping rate in 1/s (e.g. 1e5).
                          0.0 = no damping (conservative dynamics).
    """
    from equilibrium.energy import total_energy_and_grad, UM_TO_M

    N = r0_um.shape[0]
    charge_ec = np.ones(N)

    # 1. Sample
    v0 = sample_velocity(T, mol.mass_amu, rng)
    b_max = 3.0 * critical_impact_param(ion, mol, v0)
    b = sample_impact_parameter(b_max, rng)
    hit_ion = sample_collision_ion(N, rng)
    direction = sample_direction(rng)

    # 2. Collision mechanics
    theta = scattering_angle(ion, mol, v0, b)
    dv_m_s = post_collision_kick(ion, mol, v0, theta, direction)

    # 3. Initial conditions: r = r0, v = v_init + collision kick on hit ion
    # m/s and um/us are numerically equal (1 m/s = 1 um/us)
    r_flat = r0_um.ravel().copy()
    v_init = np.zeros((N, 3)) if v_init_um_us is None else v_init_um_us.copy()
    v_init[hit_ion] += dv_m_s
    v_flat = v_init.ravel()

    y0 = np.concatenate([r_flat, v_flat])

    # 4. ODE RHS: dr/dt = v, dv/dt = -grad(E)/m
    mass_kg = mass_amu * AMU
    # grad is in eV/um (from total_energy_and_grad).
    # F [N] = grad * EC / UM_TO_M;  a [m/s^2] = F / mass;
    # a [um/us^2] = a * 1e-6 (since 1 m/s^2 = 1e-6 um/us^2).
    # With UM_TO_M = 1e-6: accel_factor = EC / (mass * UM_TO_M) * 1e-6 = EC / mass.
    accel_factor = EC / mass_kg  # eV/um → um/us^2
    gamma_us = gamma_damping_per_s * 1e-6             # 1/s → 1/us

    def rhs(t, y):
        r = y[:3*N].reshape(N, 3)
        v = y[3*N:].reshape(N, 3)
        _, grad = total_energy_and_grad(fit, r, charge_ec, softening_um)
        a = -accel_factor * grad - gamma_us * v
        return np.concatenate([v.ravel(), a.ravel()])

    # 5. Integrate
    t_span = (0.0, t_integrate_us)
    t_eval = np.linspace(0.0, t_integrate_us, n_steps)
    sol = solve_ivp(rhs, t_span, y0, method="RK45", t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)

    r_final = sol.y[:3*N, -1].reshape(N, 3)

    # 6. Detect reconfiguration
    result = detector.check(r_final)

    return CollisionResult(
        reconfigured=result.reconfigured,
        v0=v0, b=b, theta=theta,
        hit_ion=hit_ion, dv=dv_m_s,
        r_final=r_final,
        trajectory=sol.y if save_trajectory else None,
        time_us=sol.t if save_trajectory else None,
    )


def run_collision_scan(
    r0_um: np.ndarray,
    fit,
    ion: Species,
    mol: Species,
    T: float,
    n_simulations: int,
    detector: ReconfigDetector,
    seed: int = 42,
    v_init_um_us: np.ndarray | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    **kwargs,
) -> CollisionScanResult:
    """Run multiple collision simulations and collect statistics.

    Parameters
    ----------
    progress_callback : callable(done, total, elapsed_s) or None
        Called after each simulation with the number completed so far,
        total count, and elapsed seconds since start.
    """
    import time as _time

    rng = np.random.default_rng(seed)
    results: list[CollisionResult] = []
    t0 = _time.perf_counter()

    for i in range(n_simulations):
        cr = run_single_collision(
            r0_um, fit, ion, mol, T, rng, detector,
            v_init_um_us=v_init_um_us, **kwargs,
        )
        results.append(cr)
        if progress_callback is not None:
            progress_callback(i + 1, n_simulations, _time.perf_counter() - t0)

    n_reconfig = sum(1 for r in results if r.reconfigured)
    return CollisionScanResult(
        n_total=n_simulations,
        n_reconfigured=n_reconfig,
        reconfig_prob=n_reconfig / n_simulations if n_simulations > 0 else 0.0,
        collisions=results,
        r0_um=r0_um,
        v0_um_us=v_init_um_us,
    )
