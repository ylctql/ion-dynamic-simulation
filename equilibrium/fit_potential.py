"""
三轴四次拟合入口脚本

加载场配置与势场，沿 x,y,z 分别做四次多项式拟合。
用法: python -m equilibrium.fit_potential [options]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="沿 x,y,z 三轴对总势场做四次多项式拟合"
    )
    parser.add_argument("--csv", type=str, default="", help="电场 CSV 路径")
    parser.add_argument("--config", type=str, default="", help="电压配置 JSON 路径")
    parser.add_argument(
        "--center",
        type=str,
        default="0,0,0",
        help="参考中心 x,y,z (μm)，逗号分隔",
    )
    parser.add_argument(
        "--x_range",
        type=str,
        default="-50,50",
        help="x 轴拟合范围 x_min,x_max (μm)，默认 -50,50",
    )
    parser.add_argument(
        "--y_range",
        type=str,
        default="-20,20",
        help="y 轴拟合范围 y_min,y_max (μm)，默认 -20,20",
    )
    parser.add_argument(
        "--z_range",
        type=str,
        default="-100,100",
        help="z 轴拟合范围 z_min,z_max (μm)，默认 -100,100",
    )
    parser.add_argument(
        "--n-pts",
        type=int,
        default=100,
        help="沿每轴采样点数，默认 100",
    )
    parser.add_argument(
        "--smooth-axes",
        type=str,
        default="z",
        help="势场平滑方向，默认 z；用 none 关闭",
    )
    parser.add_argument(
        "--smooth-sg",
        type=str,
        default="11,3",
        help="Savitzky-Golay 参数 窗口,阶数，默认 11,3",
    )
    args = parser.parse_args()

    from Interface.cli import (
        DEFAULT_CONFIG_DIR,
        DEFAULT_CONFIG_PATH,
        DEFAULT_CSV_DIR,
        DEFAULT_CSV_PATH,
    )
    from FieldConfiguration.constants import init_from_config
    from FieldConfiguration.loader import build_voltage_list, field_settings_from_config
    from FieldParser.calc_field import calc_field, calc_potential
    from FieldParser.csv_reader import read as read_csv
    from field_visualize.core import apply_savgol_smooth, compute_potentials, um_to_norm

    def _resolve_path(arg: str, default_full: str, default_dir: str) -> str:
        if not arg:
            return str(_ROOT / default_full)
        p = Path(arg)
        if not p.is_absolute() and "/" not in arg and "\\" not in arg:
            return str(_ROOT / default_dir / arg)
        return str(_ROOT / arg) if not p.is_absolute() else arg

    config_path = _resolve_path(args.config, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_DIR)
    csv_path = _resolve_path(args.csv, DEFAULT_CSV_PATH, DEFAULT_CSV_DIR)

    cfg, config = init_from_config(config_path)
    grid_coord, grid_voltage = read_csv(
        csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV
    )
    n_voltage = grid_voltage.shape[1]
    if config:
        field_settings = field_settings_from_config(
            csv_path, config_path, n_voltage, cfg
        )
    else:
        from FieldConfiguration.field_settings import FieldSettings

        field_settings = FieldSettings(csv_filename=csv_path, voltage_list=[])
        field_settings.voltage_list = build_voltage_list(
            {"voltage_list": []}, n_voltage, cfg
        )

    if args.smooth_axes.strip().lower() != "none":
        axes = tuple(a.strip().lower() for a in args.smooth_axes.split(",") if a.strip() in "xyz")
        if axes:
            parts = [p.strip() for p in args.smooth_sg.split(",")]
            wl = int(parts[0]) if parts else 11
            poly = int(parts[1]) if len(parts) >= 2 else 3
            grid_voltage = apply_savgol_smooth(
                grid_coord, grid_voltage, axes, window_length=wl, polyorder=poly
            )

    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)
    voltage_list = field_settings.voltage_list

    def compute_V_total(r):
        _, _, _, V_total = compute_potentials(
            potential_interps, field_interps, voltage_list, cfg, r
        )
        return V_total

    def um_to_norm_fn(val_um: float) -> float:
        return um_to_norm(val_um, cfg.dl)

    center_parts = [float(x.strip()) for x in args.center.split(",")]
    if len(center_parts) != 3:
        parser.error("--center 须为 x,y,z 三个数")
    center_um: tuple[float, float, float] = (
        center_parts[0],
        center_parts[1],
        center_parts[2],
    )

    def parse_range_2(s: str, arg_name: str) -> tuple[float, float]:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 2:
            parser.error(f"--{arg_name} 须为两个数: min,max")
        return (parts[0], parts[1])

    range_um = (
        parse_range_2(args.x_range, "x_range"),
        parse_range_2(args.y_range, "y_range"),
        parse_range_2(args.z_range, "z_range"),
    )

    from equilibrium.potential_fit_3d import (
        eval_fit_3d,
        fit_potential_3d_quartic,
        grad_fit_3d,
    )

    fit = fit_potential_3d_quartic(
        compute_V_total=compute_V_total,
        um_to_norm=um_to_norm_fn,
        center_um=center_um,
        range_um=range_um,
        n_pts_per_axis=args.n_pts,
    )

    print("=" * 60)
    print("总势场 3D 四次多项式拟合")
    print("=" * 60)
    print(f"参考中心: ({center_um[0]:.1f}, {center_um[1]:.1f}, {center_um[2]:.1f}) μm")
    print(f"拟合范围: x∈[{range_um[0][0]}, {range_um[0][1]}], "
          f"y∈[{range_um[1][0]}, {range_um[1][1]}], "
          f"z∈[{range_um[2][0]}, {range_um[2][1]}] μm")
    print()
    print(f"拟合优度 R²: {fit.r_squared:.6f}")
    print()
    print("模型: V(x,y,z) = Σ c_ijk u^i v^j w^k, u=(x-x0)/L, v=(y-y0)/L, w=(z-z0)/L")
    print(f"  缩放半跨度 L = {fit.scale_um:.1f} μm, 系数数组 shape (5,5,5)")
    print(f"  势能零点平移: V_shifted = V_true - ({fit.potential_offset_V:.6e} V)")
    print()
    # 验证：在中心点求值
    r_center = np.array([list(center_um)])
    V_fit_center = eval_fit_3d(fit, r_center)[0]
    grad_center = grad_fit_3d(fit, r_center)[0]
    print("中心点验证:")
    print(f"  V_fit(中心) = {V_fit_center:.6e} V")
    print(f"  grad V(中心) = ({grad_center[0]:.4e}, {grad_center[1]:.4e}, {grad_center[2]:.4e}) V/μm")
    print()
    print("系数数组 (部分，低阶项):")
    print("  c[0,0,0] (常数项):", fit.coeffs[0, 0, 0])
    print("  c[1,0,0], c[0,1,0], c[0,0,1] (一次项):", fit.coeffs[1, 0, 0], fit.coeffs[0, 1, 0], fit.coeffs[0, 0, 1])
    print("  c[2,0,0], c[0,2,0], c[0,0,2] (二次项):", fit.coeffs[2, 0, 0], fit.coeffs[0, 2, 0], fit.coeffs[0, 0, 2])
    print("=" * 60)


if __name__ == "__main__":
    main()
