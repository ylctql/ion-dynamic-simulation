"""python -m collision_pressure [scan|simulate]"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="collision_pressure",
        description="H2 collision pressure estimation",
    )
    sub = parser.add_subparsers(dest="command")

    # --- scan subcommand (existing build_config_library) ---
    p_scan = sub.add_parser("scan", help="Build configuration library")
    p_scan.add_argument("--n-ions", type=int, default=54)
    p_scan.add_argument("--n-scans", type=int, default=200)
    p_scan.add_argument("--seed", type=int, default=42)
    p_scan.add_argument("--maxiter", type=int, default=500)
    p_scan.add_argument("--fit-mode", type=str, default="quartic",
                        choices=["none", "even", "quartic", "quartic_even", "quadratic"])
    p_scan.add_argument("--output-dir", type=str, default=None)
    p_scan.add_argument("--csv", type=str, default=None)
    p_scan.add_argument("--trap-freq", nargs=3, type=float, metavar=("FX", "FY", "FZ"))
    p_scan.add_argument("--config", type=str, default="FieldConfiguration/configs/collision.json")
    p_scan.add_argument("--mass-amu", type=float, default=135.0)
    p_scan.add_argument("--softening-um", type=float, default=0.001)
    p_scan.add_argument("--edge-filter", type=float, default=1.5)
    p_scan.add_argument("--plane", type=str, default="xoz")

    # --- simulate subcommand ---
    p_sim = sub.add_parser("simulate", help="Run collision simulation")
    p_sim.add_argument("--n-ions", type=int, default=10)
    p_sim.add_argument("--n-simulations", type=int, default=100)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--mass-amu", type=float, default=135.0)
    p_sim.add_argument("--ion-species", type=str, default="Ba135+",
                       choices=["Ba135+", "Ba138+", "Yb171+"])
    p_sim.add_argument("--molecule", type=str, default="H2", choices=["H2", "He"])
    p_sim.add_argument("--molecule-temp", type=float, default=10.0, help="K")
    p_sim.add_argument("--detector", type=str, default="topology",
                       choices=["zigzag", "topology"])
    p_sim.add_argument("--flip-axis", type=int, default=0,
                       help="Transverse axis for zigzag oscillation (0=x, 1=y, 2=z)")
    p_sim.add_argument("--sort-axis", type=int, default=2,
                       help="Axial (chain) direction for ion sorting (0=x, 1=y, 2=z)")
    p_sim.add_argument("--threshold-um2", type=float, default=None,
                       help="Manual SSD threshold in um^2 (auto if not set)")
    p_sim.add_argument("--auto-threshold-factor", type=float, default=2.0,
                       help="Auto-threshold multiplier (default 2.0)")
    p_sim.add_argument("--t-integrate-us", type=float, default=50.0)
    p_sim.add_argument("--maxiter", type=int, default=1000)
    p_sim.add_argument("--softening-um", type=float, default=0.001)
    p_sim.add_argument("--gamma-damping", type=float, default=0.0,
                       help="Doppler cooling damping rate (1/s), 0=no damping")
    p_sim.add_argument("--thermalize-us", type=float, default=20.0,
                       help="Thermalize crystal for this many us before collisions (0=skip)")
    p_sim.add_argument("--thermalize-gamma", type=float, default=2e5,
                       help="Damping rate during thermalization (1/s)")
    p_sim.add_argument("--output", type=str, default=None)
    p_sim.add_argument("--init-file", type=str, default=None,
                       help="npz with r (N,3) in um and v (N,3) in m/s")
    p_sim.add_argument("--visualize", type=str, default="none",
                       choices=["none", "trajectory", "before-after", "statistics", "all"])
    p_sim.add_argument("--viz-output", type=str, default=None)
    p_sim.add_argument("--viz-n-trajectories", type=int, default=3)
    p_sim.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers (1=sequential)")

    # Potential field (mutually exclusive)
    field_group = p_sim.add_mutually_exclusive_group()
    field_group.add_argument("--csv", type=str, default=None)
    field_group.add_argument("--trap-freq", nargs=3, type=float, metavar=("FX", "FY", "FZ"))
    p_sim.add_argument("--config", type=str, default="FieldConfiguration/configs/collision.json")
    p_sim.add_argument("--fit-mode", type=str, default="quartic",
                       choices=["none", "even", "quartic", "quartic_even", "quadratic"])

    args = parser.parse_args()

    if args.command == "scan":
        from collision_pressure.build_config_library import main as scan_main
        # Re-parse with scan's own parser for full compatibility
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip "scan"
        return scan_main()

    if args.command == "simulate":
        return _run_simulate(args)

    parser.print_help()
    return 0


# --- Multiprocessing worker globals ---
_POOL_STATE = {}


def _init_pool(r0, fit, ion, mol, T, det, mass_amu, softening_um,
               t_integrate_us, v_init_um_us, gamma_damping_per_s):
    global _POOL_STATE
    _POOL_STATE.update(
        r0=r0, fit=fit, ion=ion, mol=mol, T=T, det=det,
        mass_amu=mass_amu, softening_um=softening_um,
        t_integrate_us=t_integrate_us, v_init_um_us=v_init_um_us,
        gamma_damping_per_s=gamma_damping_per_s,
    )


def _pool_worker(seed: int):
    import numpy as np
    from collision_pressure.simulation import run_single_collision
    p = _POOL_STATE
    rng = np.random.default_rng(seed)
    return run_single_collision(
        p['r0'], p['fit'], p['ion'], p['mol'], p['T'], rng, p['det'],
        mass_amu=p['mass_amu'], softening_um=p['softening_um'],
        t_integrate_us=p['t_integrate_us'], save_trajectory=True,
        v_init_um_us=p['v_init_um_us'], gamma_damping_per_s=p['gamma_damping_per_s'],
    )


def _run_simulate(args) -> int:
    from datetime import datetime

    import numpy as np
    from pathlib import Path
    import time

    from collision_pressure.species import ION_SPECIES, MOL_SPECIES, BA_135, H2_MOLECULE, reduced_mass
    from collision_pressure.config_scan import setup_fit, setup_fit_harmonic, find_equilibrium
    from collision_pressure.reconfiguration import ZigzagFlipDetector, TopologyDetector
    from collision_pressure.simulation import run_collision_scan, run_single_collision, thermalize
    from collision_pressure.pressure import estimate_pressure, langevin_rate_coefficient

    do_viz = args.visualize != "none"
    viz_traj = do_viz and args.visualize in ("trajectory", "all")
    viz_ba = do_viz and args.visualize in ("before-after", "all")
    viz_stats = do_viz and args.visualize in ("statistics", "all")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("collision_pressure/results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = Path(args.viz_output) if args.viz_output else results_dir

    ion = ION_SPECIES.get(args.ion_species, BA_135)
    mol = MOL_SPECIES.get(args.molecule, H2_MOLECULE)

    print("=== H2 Collision Simulation ===")
    print(f"  Ions:       N={args.n_ions}, {ion.name}")
    print(f"  Molecule:   {mol.name}, T={args.molecule_temp} K")
    print(f"  Detector:   {args.detector}")
    print(f"  Simulations: {args.n_simulations}")
    if args.gamma_damping > 0:
        print(f"  Damping:     gamma={args.gamma_damping:.2e} /s")
    if do_viz:
        print(f"  Visualize:  {args.visualize} -> {viz_dir}")
    print()

    # 1. Setup potential
    print("[1] Setting up potential...")
    t0 = time.time()
    if args.trap_freq is not None:
        fit, _ = setup_fit_harmonic(
            args.trap_freq[0]*1e6, args.trap_freq[1]*1e6, args.trap_freq[2]*1e6,
            mass_amu=args.mass_amu,
        )
        print(f"  Harmonic trap: fx={args.trap_freq[0]} fy={args.trap_freq[1]} fz={args.trap_freq[2]} MHz")
    else:
        csv = args.csv or "data/monolithic20241118.csv"
        fit, _ = setup_fit(csv_path=csv, config_path=args.config, fit_mode=args.fit_mode)
        print(f"  CSV+Config fit done ({time.time()-t0:.1f}s)")

    # 2. Load initial state or find equilibrium
    v_init = None  # (N, 3) in m/s ≡ um/us
    if args.init_file:
        print("\n[2] Loading initial state...")
        data = np.load(args.init_file)
        r0 = data["r"]  # (N, 3) um
        if "v" in data:
            v_init = data["v"]  # (N, 3) m/s
        args.n_ions = r0.shape[0]
        print(f"  Loaded {args.n_ions} ions from {args.init_file}")
        if v_init is not None:
            print(f"  Velocity range: [{v_init.min():.4f}, {v_init.max():.4f}] m/s")
        else:
            print("  No velocities in file, ions start at rest")
    else:
        print("\n[2] Finding equilibrium...")
        rng = np.random.default_rng(args.seed)
        eq = find_equilibrium(
            fit, args.n_ions, rng,
            x_range=(-50, 50), y_range=(-50, 50), z_range=(-150, 150),
            softening_um=args.softening_um, mass_amu=args.mass_amu,
            maxiter=args.maxiter,
        )
        r0 = eq.r_um
        print(f"  E = {eq.energy_total_eV:.6f} eV, success = {eq.success}")

        # Thermalize: evolve with damping so crystal settles properly
        if args.thermalize_us > 0:
            print(f"\n[3] Thermalizing crystal for {args.thermalize_us} us "
                  f"(gamma={args.thermalize_gamma:.1e} /s)...")
            t0_th = time.time()
            r0, v_init = thermalize(
                r0, fit,
                mass_amu=args.mass_amu,
                softening_um=args.softening_um,
                t_thermalize_us=args.thermalize_us,
                gamma_damping_per_s=args.thermalize_gamma,
                perturbation_um=0.01,
                rng=rng,
            )
            print(f"  Done ({time.time()-t0_th:.1f}s)")
            print(f"  v range: [{v_init.min():.6f}, {v_init.max():.6f}] m/s")
        else:
            print("  (no thermalization, starting from static equilibrium)")

    y = r0[:, args.flip_axis]
    print(f"  Flip-axis range: [{y.min():.4f}, {y.max():.4f}] um")

    # 3. Run collision scan
    if args.detector == "zigzag":
        det = ZigzagFlipDetector(
            flip_axis=args.flip_axis,
            sort_axis=args.sort_axis,
            threshold_um2=args.threshold_um2,
            auto_threshold_factor=args.auto_threshold_factor,
        )
    else:
        det = TopologyDetector()
    det.register_equilibrium(r0)
    if args.detector == "zigzag":
        print(f"  SSD threshold: {det.threshold:.2f} um^2")

    # --- Write simulation log header ---
    log_path = results_dir / "simulation.log"
    traj_dir = results_dir / "trajectories"

    pot_desc = (
        f"harmonic (fx={args.trap_freq[0]} fy={args.trap_freq[1]} fz={args.trap_freq[2]} MHz)"
        if args.trap_freq is not None
        else f"csv={args.csv or 'data/monolithic20241118.csv'}, config={args.config}"
    )
    det_desc = (
        f"zigzag (flip_axis={args.flip_axis}, sort_axis={args.sort_axis}, "
        f"threshold={args.threshold_um2 or 'auto'})"
        if args.detector == "zigzag"
        else "topology"
    )
    mu = reduced_mass(ion, mol)
    v_init_desc = (
        "thermalized" if v_init is not None and args.thermalize_us > 0
        else ("from file" if args.init_file else "zero (static equilibrium)")
    )

    header_lines = [
        "# ISM Collision Pressure Simulation Log",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "#",
        "# === Species ===",
        f"# ion: {ion.name} (mass={ion.mass_amu} amu, charge={ion.charge_ec} e)",
        f"# molecule: {mol.name} (mass={mol.mass_amu} amu, "
        f"polarizability={mol.polarizability:.3e} m^3)",
        f"# reduced_mass: {mu:.6e} kg",
        "#",
        "# === Potential ===",
        f"# source: {pot_desc}",
        f"# fit_mode: {args.fit_mode}",
        "#",
        "# === Ion Crystal ===",
        f"# n_ions: {args.n_ions}",
        f"# mass_amu: {args.mass_amu}",
        f"# softening_um: {args.softening_um}",
        f"# init_file: {args.init_file or '(none)'}",
        "# equilibrium_range: x=(-50,50) y=(-50,50) z=(-150,150) um",
        f"# equilibrium_maxiter: {args.maxiter}",
        "#",
        "# === Thermalization ===",
        f"# thermalize_us: {args.thermalize_us}",
        f"# thermalize_gamma: {args.thermalize_gamma:.6e} /s",
        "# perturbation_um: 0.01",
        "# thermalize_n_steps: 1000",
        f"# initial_velocity: {v_init_desc}",
        "#",
        "# === Collision Scan ===",
        f"# n_simulations: {args.n_simulations}",
        f"# seed: {args.seed}",
        f"# molecule_temp: {args.molecule_temp} K",
        f"# t_integrate_us: {args.t_integrate_us}",
        "# n_steps: 2000",
        f"# gamma_damping: {args.gamma_damping:.6e} /s",
        "# b_max_factor: 3.0",
        f"# workers: {args.workers}",
        "#",
        "# === Detector ===",
        f"# type: {det_desc}",
        "#",
        "# event  hit_ion     v0[m/s]        b[m]     theta[rad]     |dv|[m/s]     dv_x[m/s]     dv_y[m/s]     dv_z[m/s]  reconfigured  trajectory",
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")

    print(f"\n[4] Running {args.n_simulations} collision simulations...")
    if args.workers > 1:
        print(f"  Workers: {args.workers}")
    print(f"  Log: {log_path}")
    t0 = time.time()

    def _print_progress(done: int, total: int, elapsed: float):
        bar_w = 30
        pct = done / total
        filled = int(bar_w * pct)
        bar = '=' * filled + '>' * min(1, bar_w - filled) + ' ' * max(0, bar_w - filled - 1)
        eta = elapsed / done * (total - done) if done > 0 else 0
        print(f"\r  [{bar}] {done}/{total} ({100*pct:.0f}%) "
              f"elapsed {elapsed:.0f}s ETA {eta:.0f}s", end='', flush=True)

    results = []
    n_traj = min(args.viz_n_trajectories, args.n_simulations) if viz_traj else 0
    n_reconfig = 0
    from collision_pressure.visualize_collision import plot_trajectory_snapshots
    import matplotlib.pyplot as plt

    # Build result iterator: Pool.imap preserves order → log stays sequential
    pool = None
    try:
        if args.workers > 1:
            from multiprocessing import Pool
            _init_args = (r0, fit, ion, mol, args.molecule_temp, det,
                          args.mass_amu, args.softening_um, args.t_integrate_us,
                          v_init, args.gamma_damping)
            seeds = [args.seed + 1 + i for i in range(args.n_simulations)]
            chunksize = max(1, args.n_simulations // (args.workers * 4))
            pool = Pool(args.workers, initializer=_init_pool, initargs=_init_args)
            result_iter = pool.imap(_pool_worker, seeds, chunksize=chunksize)
        else:
            scan_rng = np.random.default_rng(args.seed + 1)
            result_iter = (
                run_single_collision(
                    r0, fit, ion, mol, args.molecule_temp, scan_rng, det,
                    mass_amu=args.mass_amu, t_integrate_us=args.t_integrate_us,
                    softening_um=args.softening_um, save_trajectory=True,
                    v_init_um_us=v_init, gamma_damping_per_s=args.gamma_damping,
                )
                for _ in range(args.n_simulations)
            )

        with open(log_path, "a", encoding="utf-8") as log_f:
            for i, cr in enumerate(result_iter):
                results.append(cr)
                n_reconfig += int(cr.reconfigured)

                traj_file = "-"
                if cr.reconfigured:
                    traj_dir.mkdir(exist_ok=True)
                    traj_file = f"trajectories/event_{i:04d}.npz"
                    np.savez(
                        results_dir / traj_file,
                        trajectory=cr.trajectory,
                        time_us=cr.time_us,
                        r0=r0,
                        r_final=cr.r_final,
                        hit_ion=cr.hit_ion,
                        v0=cr.v0, b=cr.b, theta=cr.theta, dv=cr.dv,
                    )
                    img_path = results_dir / f"trajectories/event_{i:04d}.png"
                    fig = plot_trajectory_snapshots(
                        cr.trajectory, cr.time_us, r0, cr.hit_ion,
                        output=img_path,
                        v0=cr.v0, b=cr.b, theta=cr.theta, dv=cr.dv,
                        reconfigured=cr.reconfigured,
                    )
                    plt.close(fig)

                dv_mag = float(np.linalg.norm(cr.dv))
                reconfig_flag = 1 if cr.reconfigured else 0
                log_f.write(
                    f"{i:>5}  {cr.hit_ion:>7}  {cr.v0:>13.6e}  {cr.b:>13.6e}"
                    f"  {cr.theta:>12.6f}  {dv_mag:>13.6e}"
                    f"  {cr.dv[0]:>13.6e}  {cr.dv[1]:>13.6e}  {cr.dv[2]:>13.6e}"
                    f"  {reconfig_flag:>12}  {traj_file}\n"
                )
                log_f.flush()

                # Free trajectory memory if not needed for viz
                if not (viz_traj and i < n_traj):
                    cr.trajectory = None
                    cr.time_us = None

                _print_progress(i + 1, args.n_simulations, time.time() - t0)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print()

    scan = type("CollisionScanResult", (), {
        "n_total": args.n_simulations,
        "n_reconfigured": n_reconfig,
        "reconfig_prob": n_reconfig / args.n_simulations if args.n_simulations > 0 else 0.0,
        "collisions": results,
        "r0_um": r0,
        "v0_um_us": v_init,
    })()

    print(f"  Done ({time.time()-t0:.1f}s)")
    print(f"\n  Results: {scan.n_reconfigured}/{scan.n_total} reconfigured")
    print(f"  P_flip = {scan.reconfig_prob:.4f}")

    k_L = langevin_rate_coefficient(ion, mol)
    print(f"  Langevin rate coefficient k_L = {k_L:.4e} m^3/s")

    pressure_coeff = estimate_pressure(scan.reconfig_prob, ion, mol, args.molecule_temp)
    print(f"  Pressure coefficient (P / R_obs) = {pressure_coeff:.4e} Pa*s")
    print(f"  => P = {pressure_coeff:.4e} * R_obs  (need experimental R_obs to get Pa)")

    # Append results summary to log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("#\n")
        f.write("# === Results ===\n")
        f.write(f"# reconfig_prob: {scan.reconfig_prob:.6f}\n")
        f.write(f"# k_L: {k_L:.6e} m^3/s\n")
        f.write(f"# pressure_coeff: {pressure_coeff:.6e} Pa*s\n")

    # 4. Visualization
    if do_viz:
        import matplotlib.pyplot as plt
        from collision_pressure.visualize_collision import (
            plot_trajectory_snapshots, plot_before_after, plot_batch_statistics,
        )
        print(f"\n[5] Generating visualizations in {viz_dir}...")

        if viz_traj:
            for i, c in enumerate(scan.collisions):
                if c.trajectory is None:
                    continue
                out = viz_dir / f"trajectory_{i:03d}.png"
                plot_trajectory_snapshots(
                    c.trajectory, c.time_us, r0, c.hit_ion, output=out,
                    v0=c.v0, b=c.b, theta=c.theta, dv=c.dv,
                    reconfigured=c.reconfigured,
                )
                plt.close("all")
                print(f"  -> {out}")

        if viz_ba:
            # Pick one reconfigured and one unchanged
            reconfig_idx = next((i for i, c in enumerate(scan.collisions) if c.reconfigured), None)
            unchanged_idx = next((i for i, c in enumerate(scan.collisions) if not c.reconfigured), None)
            for label, idx in [("reconfig", reconfig_idx), ("unchanged", unchanged_idx)]:
                if idx is None:
                    continue
                c = scan.collisions[idx]
                if c.r_final is None:
                    continue
                out = viz_dir / f"before_after_{label}.png"
                plot_before_after(
                    r0, c.r_final, c.hit_ion,
                    flip_axis=args.flip_axis, reconfigured=c.reconfigured, output=out,
                    v0=c.v0, b=c.b, theta=c.theta, dv=c.dv,
                )
                plt.close("all")
                print(f"  -> {out}")

        if viz_stats:
            out = viz_dir / "statistics.png"
            plot_batch_statistics(scan.collisions, args.n_ions, output=out)
            plt.close("all")
            print(f"  -> {out}")

    # Save
    if args.output:
        np.savez(
            args.output,
            r0=scan.r0_um,
            reconfigured=np.array([c.reconfigured for c in scan.collisions]),
            v0=np.array([c.v0 for c in scan.collisions]),
            b=np.array([c.b for c in scan.collisions]),
            theta=np.array([c.theta for c in scan.collisions]),
            hit_ion=np.array([c.hit_ion for c in scan.collisions]),
            reconfig_prob=scan.reconfig_prob,
            langevin_coeff=k_L,
            pressure_coeff_Pa_s=pressure_coeff,
        )
        print(f"\n  Saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
