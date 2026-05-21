"""
构型预扫描：多起点 L-BFGS-B 能量最小化 + 拓扑合并

从随机初始位置出发多次求解平衡构型，用拓扑指纹合并等价构型，
构建给定离子数和陷阱参数下的构型库。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from collision_pressure.topology import CrystalTopology, build_topology, same_topology


@dataclass
class ConfigResult:
    """单个构型的扫描结果"""
    r_um: np.ndarray                    # (N, 3) 平衡位置
    energy_total_eV: float
    energy_trap_eV: float
    energy_coulomb_eV: float
    success: bool
    topology: CrystalTopology
    nit: int = 0


class ConfigurationLibrary:
    """构型库：持有同一陷阱条件下所有不等价的平衡构型"""

    def __init__(self, n_ions: int):
        self.n_ions = n_ions
        self.configs: list[ConfigResult] = []

    def add_or_merge(self, candidate: ConfigResult) -> str:
        """尝试将候选构型加入库

        Returns
        -------
        'new' | 'merged' | 'replaced'
        """
        for i, existing in enumerate(self.configs):
            if same_topology(existing.topology, candidate.topology):
                if candidate.energy_total_eV < existing.energy_total_eV:
                    self.configs[i] = candidate
                    return "replaced"
                return "merged"

        self.configs.append(candidate)
        return "new"

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        summary = []
        for idx, cfg in enumerate(self.configs):
            fname = f"config_{idx:03d}.npz"
            t = cfg.topology
            np.savez(
                directory / fname,
                r=cfg.r_um,
                energy_total_eV=cfg.energy_total_eV,
                energy_trap_eV=cfg.energy_trap_eV,
                energy_coulomb_eV=cfg.energy_coulomb_eV,
                success=cfg.success,
                nit=cfg.nit,
                coord_numbers=t.coord_numbers,
                adj_matrix=t.adj_matrix,
                boundary=t.boundary,
            )
            summary.append({
                "idx": idx,
                "npz_file": fname,
                "energy_total_eV": cfg.energy_total_eV,
                "energy_trap_eV": cfg.energy_trap_eV,
                "energy_coulomb_eV": cfg.energy_coulomb_eV,
                "fingerprint_coord_seq": list(t.fingerprint.coord_seq),
                "n_boundary": t.fingerprint.n_boundary,
                "success": cfg.success,
            })

        with open(directory / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"n_ions": self.n_ions, "configurations": summary}, f,
                       indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path) -> ConfigurationLibrary:
        directory = Path(directory)
        with open(directory / "summary.json", encoding="utf-8") as f:
            meta = json.load(f)

        lib = cls(n_ions=meta["n_ions"])
        for entry in meta["configurations"]:
            data = np.load(directory / entry["npz_file"])
            r_um = data["r"]
            adj_matrix = data["adj_matrix"]
            n = adj_matrix.shape[0]
            adjacency = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if adj_matrix[i, j]:
                        adjacency[i].append(j)
            coord_numbers = data["coord_numbers"]
            boundary = data["boundary"]

            from collision_pressure.topology import TopologyFingerprint
            fingerprint = TopologyFingerprint(
                coord_seq=tuple(sorted(coord_numbers.tolist())),
                n_boundary=int(len(boundary)),
            )
            topology = CrystalTopology(
                adj_matrix=adj_matrix,
                adjacency=adjacency,
                coord_numbers=coord_numbers,
                boundary=boundary,
                fingerprint=fingerprint,
            )
            lib.configs.append(ConfigResult(
                r_um=r_um,
                energy_total_eV=float(data["energy_total_eV"]),
                energy_trap_eV=float(data["energy_trap_eV"]),
                energy_coulomb_eV=float(data["energy_coulomb_eV"]),
                success=bool(data["success"]),
                topology=topology,
                nit=int(data["nit"]),
            ))
        return lib

    def summary(self) -> str:
        lines = [f"构型库: N={self.n_ions}, 共 {len(self.configs)} 种不等价构型"]
        lines.append(f"{'#':>3}  {'E_total(eV)':>14}  {'E_trap(eV)':>14}  "
                     f"{'E_coulomb(eV)':>14}  {'边界':>4}  {'配位数指纹'}")
        for i, cfg in enumerate(self.configs):
            fp = cfg.topology.fingerprint
            lines.append(
                f"{i:>3}  {cfg.energy_total_eV:>14.6f}  "
                f"{cfg.energy_trap_eV:>14.6f}  "
                f"{cfg.energy_coulomb_eV:>14.6f}  "
                f"{fp.n_boundary:>4}  {fp.coord_seq}"
            )
        return "\n".join(lines)


def setup_fit(
    csv_path: str = "data/monolithic20241118.csv",
    config_path: str = "FieldConfiguration/configs/collision.json",
    fit_mode: str = "quartic",
    smooth_axes: tuple[str, ...] = ("z",),
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-20.0, 20.0),
    z_range: tuple[float, float] = (-150.0, 150.0),
    n_pts_per_axis: tuple[int, int, int] = (100, 40, 300),
) -> "tuple":
    """加载电场数据并拟合 3D 势场多项式

    Returns
    -------
    (FitResult3D, potential_offset_V)
    """
    from FieldConfiguration.constants import init_from_config
    from FieldParser.csv_reader import read as read_csv
    from FieldParser.calc_field import calc_field, calc_potential
    from FieldConfiguration.loader import field_settings_from_config
    from field_visualize.core import (
        apply_savgol_smooth,
        compute_potentials,
        um_to_norm,
    )
    from equilibrium.potential_fit_3d import fit_potential_3d_quartic

    cfg, config = init_from_config(config_path)

    grid_coord, grid_voltage = read_csv(csv_path, None, normalize=True,
                                         dl=cfg.dl, dV=cfg.dV)
    n_voltage = grid_voltage.shape[1]

    if smooth_axes:
        grid_voltage = apply_savgol_smooth(
            grid_coord, grid_voltage, smooth_axes,
            window_length=smooth_window, polyorder=smooth_polyorder,
        )

    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)

    fs = field_settings_from_config(csv_path, config_path, n_voltage, cfg)
    voltage_list = fs.voltage_list

    def compute_V_total(r_norm: np.ndarray) -> np.ndarray:
        _, _, _, v_total = compute_potentials(
            potential_interps, field_interps, voltage_list, cfg, r_norm,
        )
        return v_total

    v_grid = compute_V_total(grid_coord)
    v_min_grid = float(np.min(v_grid[np.isfinite(v_grid)]))

    fit = fit_potential_3d_quartic(
        compute_V_total=compute_V_total,
        um_to_norm=lambda v: um_to_norm(v, cfg.dl),
        center_um=(0.0, 0.0, 0.0),
        range_um=(x_range, y_range, z_range),
        n_pts_per_axis=n_pts_per_axis,
        potential_offset_V=v_min_grid,
        fit_mode=fit_mode,
    )
    return fit, v_min_grid


def setup_fit_harmonic(
    fx_Hz: float,
    fy_Hz: float,
    fz_Hz: float,
    mass_amu: float = 135.0,
    scale_um: float = 100.0,
) -> "tuple":
    """从三个方向的阱频构造理想谐振势

    V(x,y,z) = 0.5 * (d2V_dx2 * x^2 + d2V_dy2 * y^2 + d2V_dz2 * z^2)

    Parameters
    ----------
    fx_Hz, fy_Hz, fz_Hz : float
        各轴阱频 (Hz)
    mass_amu : float
        离子质量 (amu), 默认 135.0 (Ba-135+)
    scale_um : float
        多项式坐标缩放半跨度 (um)

    Returns
    -------
    (FitResult3D, 0.0)
    """
    from equilibrium.potential_fit_3d import FitResult3D, QUADRATIC_FIT_EXPS

    AMU = 1.66053906660e-27   # kg
    EC = 1.602176634e-19      # C

    m_kg = mass_amu * AMU

    omega = np.array([2.0 * np.pi * fx_Hz,
                      2.0 * np.pi * fy_Hz,
                      2.0 * np.pi * fz_Hz])

    # d2V/dxi^2 = m * omega_i^2 / q  (V/m^2)
    # c2i = 0.5 * d2V/dxi^2, convert to V/um^2
    c2 = 0.5 * m_kg * omega**2 / EC * 1e-12   # shape (3,)

    L = scale_um
    coeffs = np.zeros((5, 5, 5))
    coeffs[2, 0, 0] = c2[0] * L**2
    coeffs[0, 2, 0] = c2[1] * L**2
    coeffs[0, 0, 2] = c2[2] * L**2

    fit = FitResult3D(
        coeffs=coeffs,
        center_um=(0.0, 0.0, 0.0),
        scale_um=L,
        potential_offset_V=0.0,
        r_squared=1.0,
        fit_mode="quadratic",
        basis_exps=QUADRATIC_FIT_EXPS,
    )
    return fit, 0.0


def find_equilibrium(
    fit,
    n_ions: int,
    rng: np.random.Generator,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-20.0, 20.0),
    z_range: tuple[float, float] = (-150.0, 150.0),
    softening_um: float = 0.001,
    mass_amu: float = 135.0,
    maxiter: int = 500,
    ftol: float = 1e-10,
    plane: str = "xoz",
    edge_filter_factor: float = 1.5,
) -> ConfigResult:
    """从随机初始位置出发，求一个平衡构型

    Parameters
    ----------
    fit : FitResult3D
        由 setup_fit 返回的势场拟合结果
    """
    from equilibrium.energy import total_energy_and_grad

    charge_ec = np.ones(n_ions)

    r0 = np.zeros((n_ions, 3))
    r0[:, 0] = rng.uniform(x_range[0], x_range[1], size=n_ions)
    r0[:, 1] = rng.uniform(y_range[0], y_range[1], size=n_ions)
    r0[:, 2] = rng.uniform(z_range[0], z_range[1], size=n_ions)

    bounds = [x_range, y_range, z_range] * n_ions

    def objective(x_flat: np.ndarray):
        r = x_flat.reshape(n_ions, 3)
        breakdown, grad = total_energy_and_grad(fit, r, charge_ec, softening_um)
        return breakdown.total_eV, grad.ravel()

    res = minimize(
        fun=lambda x: objective(x)[0],
        x0=r0.ravel(),
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    r_eq = res.x.reshape(n_ions, 3)
    breakdown, _ = total_energy_and_grad(fit, r_eq, charge_ec, softening_um)

    topology = build_topology(r_eq, plane=plane, edge_filter_factor=edge_filter_factor)

    return ConfigResult(
        r_um=r_eq,
        energy_total_eV=breakdown.total_eV,
        energy_trap_eV=breakdown.trap_eV,
        energy_coulomb_eV=breakdown.coulomb_eV,
        success=bool(res.success),
        topology=topology,
        nit=int(res.nit),
    )


def scan_configurations(
    fit,
    n_ions: int,
    n_scans: int = 200,
    seed: int = 42,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-20.0, 20.0),
    z_range: tuple[float, float] = (-150.0, 150.0),
    softening_um: float = 0.001,
    mass_amu: float = 135.0,
    maxiter: int = 500,
    ftol: float = 1e-10,
    plane: str = "xoz",
    edge_filter_factor: float = 1.5,
    verbose: bool = True,
) -> ConfigurationLibrary:
    """多次随机初猜 + 拓扑合并，构建构型库"""
    rng = np.random.default_rng(seed)
    lib = ConfigurationLibrary(n_ions)
    stats = {"new": 0, "merged": 0, "replaced": 0, "failed": 0}

    for i in range(n_scans):
        result = find_equilibrium(
            fit, n_ions, rng,
            x_range=x_range, y_range=y_range, z_range=z_range,
            softening_um=softening_um, mass_amu=mass_amu,
            maxiter=maxiter, ftol=ftol,
            plane=plane, edge_filter_factor=edge_filter_factor,
        )

        if not result.success:
            stats["failed"] += 1
            continue

        action = lib.add_or_merge(result)
        stats[action] += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_scans}] 新={stats['new']} "
                  f"合并={stats['merged']} 替换={stats['replaced']} "
                  f"失败={stats['failed']}")

    if verbose:
        print(f"\n扫描完成: {n_scans} 次优化 → {len(lib.configs)} 种不等价构型")
        print(f"  新={stats['new']} 合并={stats['merged']} "
              f"替换={stats['replaced']} 失败={stats['failed']}")
        print(lib.summary())

    return lib
