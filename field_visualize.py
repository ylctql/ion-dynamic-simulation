"""
电场可视化：静电势、RF 赝势、总电势的空间分布
支持 1D（单坐标）与 2D（热力图/三维）绘图，可指定坐标变量与绘图区间
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np

# 需在 import 其他模块前设置路径
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from FieldConfiguration.constants import Config, init_from_config, m as ION_MASS
from FieldConfiguration.loader import build_voltage_list, field_settings_from_config
from FieldParser.csv_reader import read as read_csv
from FieldParser.calc_field import calc_field, calc_potential
from utils import Voltage

CoordAxis = Literal["x", "y", "z"]
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# SI 单位换算：长度 μm ↔ m，内部使用归一化坐标 r_norm = r_si / dl
UM = 1e-6  # 1 μm = 1e-6 m


def _um_to_norm(val_um: float, dl: float) -> float:
    """μm → 归一化坐标"""
    return val_um * UM / dl


def _norm_to_um(val_norm: float, dl: float) -> float:
    """归一化坐标 → μm"""
    return val_norm * dl / UM


def _is_rf(voltage: Voltage) -> bool:
    """判断是否为 RF 电极（V0 非零）"""
    return abs(voltage.V0) > 1e-12


def _compute_potentials(
    potential_interps: list[Callable],
    field_interps: list[Callable],
    voltage_list: list[Voltage],
    cfg: Config,
    r: np.ndarray,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算静电势、RF 赝势、总电势（SI 单位 V）

    potential_interps 输出 V_raw/dV（无量纲），需乘 dV 得 SI 电势
    field_interps 输出 (dl/dV)*E_si，即 E_si = field_interp * (dV/dl)
    """
    r = np.atleast_2d(r)
    dl, dV = cfg.dl, cfg.dV
    V_dc = np.zeros(r.shape[0])
    V_rf_amp = np.zeros(r.shape[0])
    E_rf = np.zeros((r.shape[0], 3))

    for v_interp, e_interp, v in zip(potential_interps, field_interps, voltage_list):
        V_basis_norm = np.atleast_1d(v_interp(r)).ravel()  # V_raw/dV，无量纲
        E_basis = e_interp(r)  # (dl/dV)*E_si
        V_dc += v.V_bias * V_basis_norm * dV
        if _is_rf(v):
            V_rf_amp += v.V0 * V_basis_norm * dV
            E_rf += v.V0 * E_basis

    # E_rf 为 (dl/dV)*E_si 的叠加，E_rf_si = E_rf * (dV/dl)
    E_rf_si = E_rf * (dV / dl)
    from scipy.constants import e
    Omega = cfg.Omega
    q = e
    # 赝势 U = q²/(4mΩ²)|E_RF|² 为势能 (J)，除以 q 得电势 (V)
    coeff = (q ** 2) / (4 * ION_MASS * Omega ** 2)
    E_rf_sq = np.sum(E_rf_si ** 2, axis=1)
    V_pseudo_J = coeff * E_rf_sq
    V_pseudo = V_pseudo_J / q  # J → V

    V_total = V_dc + V_pseudo
    return V_dc, V_rf_amp, V_pseudo, V_total


def _apply_offset_min(
    V_dc: np.ndarray, V_pseudo: np.ndarray, V_total: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """静电势与赝势各减去其最小值，总电势 = 偏置后静电势 + 偏置后赝势"""
    valid_dc = V_dc[~np.isnan(V_dc)]
    valid_pseudo = V_pseudo[~np.isnan(V_pseudo)]
    V_dc_off = V_dc - (np.min(valid_dc) if valid_dc.size else 0)
    V_pseudo_off = V_pseudo - (np.min(valid_pseudo) if valid_pseudo.size else 0)
    V_total_off = V_dc_off + V_pseudo_off
    return V_dc_off, V_pseudo_off, V_total_off


def _set_ylim_from_data(ax, data: np.ndarray, pad_frac: float = 0.1) -> None:
    """根据数据范围设置 y 轴，避免数值过小时曲线不可见"""
    data = np.asarray(data)
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return
    y_min, y_max = np.min(valid), np.max(valid)
    span = y_max - y_min
    if span < 1e-30:
        ax.set_ylim(y_min - 0.1, y_min + 0.1)
    else:
        pad = max(span * pad_frac, span * 0.05)
        ax.set_ylim(y_min - pad, y_max + pad)


def _build_grid_1d(
    vary_axis: CoordAxis,
    vary_range: tuple[float, float],
    x_const: float,
    y_const: float,
    z_const: float,
    n_pts: int,
) -> np.ndarray:
    """构建 1D 采样点，vary_range 为变化坐标的范围"""
    coords = np.linspace(vary_range[0], vary_range[1], n_pts)
    r = np.zeros((n_pts, 3))
    r[:, 0] = x_const
    r[:, 1] = y_const
    r[:, 2] = z_const
    r[:, AXIS_INDEX[vary_axis]] = coords
    return r


def _build_grid_2d(
    vary_axes: tuple[CoordAxis, CoordAxis],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    const_val: float,
    n_pts: tuple[int, int],
) -> np.ndarray:
    """构建 2D 采样网格，返回 (n1*n2, 3)"""
    a1, a2 = vary_axes
    i1, i2 = AXIS_INDEX[a1], AXIS_INDEX[a2]
    c1 = np.linspace(x_range[0], x_range[1], n_pts[0])
    c2 = np.linspace(y_range[0], y_range[1], n_pts[1])
    cc1, cc2 = np.meshgrid(c1, c2, indexing="ij")
    n_tot = cc1.size
    r = np.full((n_tot, 3), const_val)
    r[:, i1] = cc1.ravel()
    r[:, i2] = cc2.ravel()
    return r, (cc1, cc2)


def plot_1d(
    potential_interps: list,
    field_interps: list,
    voltage_list: list[Voltage],
    cfg: Config,
    vary_axis: CoordAxis,
    vary_range: tuple[float, float],
    x_const: float = 0.0,
    y_const: float = 0.0,
    z_const: float = 0.0,
    n_pts: int = 500,
    offset_min: bool = False,
    show_rf_amp: bool = False,
    out_path: str | None = None,
) -> None:
    """1D 绘图：电势随单坐标变化"""
    import matplotlib.pyplot as plt

    r = _build_grid_1d(vary_axis, vary_range, x_const, y_const, z_const, n_pts)
    idx = AXIS_INDEX[vary_axis]
    coord_norm = r[:, idx]
    coord_um = _norm_to_um(coord_norm, cfg.dl)

    V_dc, V_rf_amp, V_pseudo, V_total = _compute_potentials(
        potential_interps, field_interps, voltage_list, cfg, r
    )
    if offset_min:
        V_dc, V_pseudo, V_total = _apply_offset_min(V_dc, V_pseudo, V_total)

    x_um = _norm_to_um(x_const, cfg.dl)
    y_um = _norm_to_um(y_const, cfg.dl)
    z_um = _norm_to_um(z_const, cfg.dl)
    title_base = f"Potential vs {vary_axis} (x={x_um:.1f} μm, y={y_um:.1f} μm, z={z_um:.1f} μm)"

    def _save_or_show_1d(path: str | None, suffix: str, fig) -> None:
        if path:
            base, ext = os.path.splitext(path)
            p = f"{base}{suffix}{ext}"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved: {p}")
        else:
            plt.show()

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(coord_um, V_dc, label="Static potential (DC)", color="blue")
    ax1.plot(coord_um, V_pseudo, label="RF pseudopotential", color="green")
    ax1.plot(coord_um, V_total, label="Total potential", color="red", linestyle="--")
    ax1.set_xlabel(f"{vary_axis} (μm)")
    ax1.set_ylabel("Potential (V)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{title_base} — Potentials")
    _set_ylim_from_data(ax1, np.concatenate([V_dc, V_pseudo, V_total]))
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _save_or_show_1d(out_path, "_potentials", fig1)
    plt.close()

    if show_rf_amp:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(coord_um, V_rf_amp, label="RF amplitude", color="orange")
        ax2.set_xlabel(f"{vary_axis} (μm)")
        ax2.set_ylabel("Potential (V)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"{title_base} — RF amplitude")
        _set_ylim_from_data(ax2, V_rf_amp)
        plt.tight_layout()
        _save_or_show_1d(out_path, "_rf_amp", fig2)
        plt.close()


def plot_2d(
    potential_interps: list,
    field_interps: list,
    voltage_list: list[Voltage],
    cfg: Config,
    vary_axes: tuple[CoordAxis, CoordAxis],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    const_val: float = 0.0,
    n_pts: tuple[int, int] = (100, 100),
    mode: Literal["heatmap", "3d"] = "heatmap",
    offset_min: bool = False,
    show_rf_amp: bool = False,
    out_path: str | None = None,
) -> None:
    """2D 绘图：热力图或三维曲面"""
    import matplotlib.pyplot as plt

    r, (cc1, cc2) = _build_grid_2d(vary_axes, x_range, y_range, const_val, n_pts)
    V_dc, V_rf_amp, V_pseudo, V_total = _compute_potentials(
        potential_interps, field_interps, voltage_list, cfg, r
    )
    if offset_min:
        V_dc, V_pseudo, V_total = _apply_offset_min(V_dc, V_pseudo, V_total)
    a1, a2 = vary_axes
    cc1_um = _norm_to_um(cc1, cfg.dl)
    cc2_um = _norm_to_um(cc2, cfg.dl)
    const_um = _norm_to_um(const_val, cfg.dl)
    V_dc_2d = V_dc.reshape(cc1.shape)
    V_rf_2d = V_rf_amp.reshape(cc1.shape)
    V_pseudo_2d = V_pseudo.reshape(cc1.shape)
    V_total_2d = V_total.reshape(cc1.shape)

    other = next(c for c in "xyz" if c not in vary_axes)
    suptitle_base = f"Potential distribution ({other}={const_um:.1f} μm)"

    def _save_or_show(path: str | None, suffix: str, fig) -> None:
        if path:
            base, ext = os.path.splitext(path)
            p = f"{base}{suffix}{ext}"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved: {p}")
        else:
            plt.show()

    if mode == "heatmap":
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0]]
        axes[1, 1].set_visible(False)
        for ax, data, title in zip(
            axes_flat,
            [V_dc_2d, V_pseudo_2d, V_total_2d],
            ["Static potential", "RF pseudopotential", "Total potential"],
        ):
            im = ax.pcolormesh(cc1_um, cc2_um, data, shading="auto", cmap="RdBu_r")
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
            cbar.set_label("V")
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_aspect("equal")
            ax.set_title(title)
        fig.suptitle(suptitle_base)
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _save_or_show(out_path, "_potentials", fig)
        plt.close(fig)

        if show_rf_amp:
            fig2, ax = plt.subplots(figsize=(6, 5))
            im = ax.pcolormesh(cc1_um, cc2_um, V_rf_2d, shading="auto", cmap="RdBu_r")
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
            cbar.set_label("V")
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_aspect("equal")
            ax.set_title("RF amplitude")
            fig2.suptitle(suptitle_base)
            plt.tight_layout()
            _save_or_show(out_path, "_rf_amp", fig2)
            plt.close(fig2)
    else:
        fig = plt.figure(figsize=(10, 10))
        for i, (data, title) in enumerate(
            [(V_dc_2d, "Static potential"), (V_pseudo_2d, "RF pseudopotential"), (V_total_2d, "Total potential")]
        ):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            ax.plot_surface(cc1_um, cc2_um, data, cmap="RdBu_r", alpha=0.9)
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_zlabel("Potential (V)")
            ax.set_title(title)
        fig.add_subplot(2, 2, 4).set_visible(False)
        fig.suptitle(suptitle_base)
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _save_or_show(out_path, "_potentials", fig)
        plt.close(fig)

        if show_rf_amp:
            fig2 = plt.figure(figsize=(7, 5))
            ax = fig2.add_subplot(111, projection="3d")
            ax.plot_surface(cc1_um, cc2_um, V_rf_2d, cmap="RdBu_r", alpha=0.9)
            ax.set_xlabel(f"{a1} (μm)")
            ax.set_ylabel(f"{a2} (μm)")
            ax.set_zlabel("Potential (V)")
            ax.set_title("RF amplitude")
            fig2.suptitle(suptitle_base)
            plt.tight_layout()
            _save_or_show(out_path, "_rf_amp", fig2)
            plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="电场电势可视化")
    parser.add_argument("--csv", type=str, default="", help="电场 CSV 路径")
    parser.add_argument("--config", type=str, default="", help="电压配置 JSON 路径")
    parser.add_argument(
        "--vary",
        type=str,
        default="x",
        help="变化坐标：单坐标 (x/y/z) 为 1D；逗号分隔 (如 x,y) 为 2D",
    )
    parser.add_argument(
        "--x_range",
        type=str,
        default="-100,100",
        help="主变化方向范围 (μm)，逗号分隔，须在网格范围内",
    )
    parser.add_argument(
        "--y_range",
        type=str,
        default="-100,100",
        help="2D 时第二坐标范围 (μm)",
    )
    parser.add_argument(
        "--const",
        type=str,
        default="0,0,0",
        help="固定坐标 x,y,z (μm)，逗号分隔，默认 0,0,0",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["heatmap", "3d"],
        default="heatmap",
        help="2D 绘图模式：heatmap 或 3d",
    )
    parser.add_argument(
        "--n_pts",
        type=str,
        default="",
        help="采样点数：1D 时为单个整数 (如 500)；2D 时为逗号分隔 (如 100,100)。与 --vary 维数须一致",
    )
    parser.add_argument("--out", type=str, default=None, help="输出图片路径")
    parser.add_argument(
        "--offset",
        action="store_true",
        help="静电势与赝势均减去各自最小值作为偏置（总电势 = 偏置后静电势 + 偏置后赝势）",
    )
    parser.add_argument(
        "--show-rf-amp",
        action="store_true",
        help="显示 RF 幅度图像（默认不显示）",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    from Interface.cli import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_CSV_PATH,
        DEFAULT_CSV_DIR,
        DEFAULT_CONFIG_DIR,
    )

    def _resolve_path(arg: str, default_full: str, default_dir: str) -> str:
        if not arg:
            return str(root / default_full)
        p = Path(arg)
        if not p.is_absolute() and "/" not in arg and "\\" not in arg:
            return str(root / default_dir / arg)
        return str(root / arg) if not p.is_absolute() else arg

    config_path = _resolve_path(args.config, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_DIR)
    csv_path = _resolve_path(args.csv, DEFAULT_CSV_PATH, DEFAULT_CSV_DIR)
    if not Path(csv_path).is_absolute():
        csv_path = str(root / csv_path)
    if not Path(config_path).is_absolute():
        config_path = str(root / config_path)

    cfg, config = init_from_config(config_path)
    grid_coord, grid_voltage = read_csv(
        csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV
    )
    n_voltage = grid_voltage.shape[1]
    if config:
        field_settings = field_settings_from_config(csv_path, config_path, n_voltage, cfg)
    else:
        from FieldConfiguration.field_settings import FieldSettings
        field_settings = FieldSettings(csv_filename=csv_path, voltage_list=[])
        from FieldConfiguration.loader import build_voltage_list
        field_settings.voltage_list = build_voltage_list(
            {"voltage_list": []}, n_voltage, cfg
        )

    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)
    voltage_list = field_settings.voltage_list

    def parse_range(s: str) -> tuple[float, float]:
        a, b = s.split(",")
        return float(a.strip()), float(b.strip())

    def parse_const(s: str) -> tuple[float, float, float]:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError("--const 须为 x,y,z 三个数")
        return parts[0], parts[1], parts[2]

    xc_um, yc_um, zc_um = parse_const(args.const)
    xr_um = parse_range(args.x_range)
    dl = cfg.dl

    def parse_n_pts(s: str, dim: int):
        if not s.strip():
            return 500 if dim == 1 else (100, 100)
        parts = [p.strip() for p in s.split(",")]
        if dim == 1:
            if len(parts) != 1:
                raise ValueError("1D 时 --n_pts 须为单个整数 (如 500)")
            return int(parts[0])
        if dim == 2:
            if len(parts) != 2:
                raise ValueError("2D 时 --n_pts 须为两个整数 (如 100,100)")
            return int(parts[0]), int(parts[1])
        raise ValueError("--vary 须为单坐标或两个坐标")

    xc = _um_to_norm(xc_um, dl)
    yc = _um_to_norm(yc_um, dl)
    zc = _um_to_norm(zc_um, dl)
    xr = (_um_to_norm(xr_um[0], dl), _um_to_norm(xr_um[1], dl))

    vary_parts = [p.strip().lower() for p in args.vary.split(",")]
    if len(vary_parts) == 1:
        vary = vary_parts[0]
        if vary not in "xyz":
            raise ValueError("--vary 须为 x, y 或 z")
        n_pts = parse_n_pts(args.n_pts, 1)
        plot_1d(
            potential_interps,
            field_interps,
            voltage_list,
            cfg,
            vary_axis=vary,
            vary_range=xr,
            x_const=xc,
            y_const=yc,
            z_const=zc,
            n_pts=n_pts,
            offset_min=args.offset,
            show_rf_amp=args.show_rf_amp,
            out_path=args.out,
        )
    elif len(vary_parts) == 2:
        a1, a2 = vary_parts[0], vary_parts[1]
        if a1 not in "xyz" or a2 not in "xyz" or a1 == a2:
            raise ValueError("--vary 须为两个不同坐标 x,y,z")
        yr_um = parse_range(args.y_range)
        yr = (_um_to_norm(yr_um[0], dl), _um_to_norm(yr_um[1], dl))
        n_pts = parse_n_pts(args.n_pts, 2)
        const_idx = next(i for i, c in enumerate("xyz") if c not in (a1, a2))
        const_val = [xc, yc, zc][const_idx]
        plot_2d(
            potential_interps,
            field_interps,
            voltage_list,
            cfg,
            vary_axes=(a1, a2),
            x_range=xr,
            y_range=yr,
            const_val=const_val,
            n_pts=n_pts,
            mode=args.mode,
            offset_min=args.offset,
            show_rf_amp=args.show_rf_amp,
            out_path=args.out,
        )
    else:
        raise ValueError("--vary 须为单坐标 (x/y/z) 或两个坐标 (如 x,y)")


if __name__ == "__main__":
    main()
