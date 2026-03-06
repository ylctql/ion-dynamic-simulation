"""
电场可视化 CLI：argparse 与主流程路由
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import apply_savgol_smooth, um_to_norm
from .plots import plot_1d, plot_2d, plot_freq_scan_1d, plot_freq_scan_2d
from .trap_freq import (
    compute_freq_scan_1d,
    compute_freq_scan_2d,
    compute_trap_freqs_at_point,
)


def main() -> None:
    # 需在 import 项目模块前设置路径
    root = Path(__file__).resolve().parent.parent
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from FieldConfiguration.constants import init_from_config
    from FieldConfiguration.loader import build_voltage_list, field_settings_from_config
    from FieldParser.calc_field import calc_field, calc_potential
    from FieldParser.csv_reader import read as read_csv

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
    parser.add_argument(
        "--fit",
        type=int,
        default=None,
        choices=[2, 4],
        metavar="DEGREE",
        help="1D 时对势场做多项式拟合：2=二次，4=四次；叠加虚线并显示 R²",
    )
    parser.add_argument(
        "--freq",
        action="store_true",
        help="计算并输出阱频分布 f_x, f_y, f_z (MHz)，在 --const 指定点沿各轴拟合总势",
    )
    parser.add_argument(
        "--z_range",
        type=str,
        default="-100,100",
        help="--freq 时 z 轴拟合范围 (μm)，逗号分隔",
    )
    parser.add_argument(
        "--freq-fit-degree",
        type=int,
        default=2,
        choices=[2, 4],
        metavar="DEGREE",
        help="--freq 时拟合阶数，默认 2",
    )
    parser.add_argument(
        "--freq-n-pts",
        type=int,
        default=200,
        help="--freq 时每轴拟合采样点数，默认 200",
    )
    parser.add_argument(
        "--freq-scan",
        type=str,
        default=None,
        metavar="AXES",
        help="阱频沿轴扫描：单轴 (x/y/z) 绘曲线；双轴 (如 x,y) 绘 heatmap/3d；指定后不绘势场",
    )
    parser.add_argument(
        "--freq-scan-n",
        type=str,
        default="50",
        help="--freq-scan 扫描点数：1D 为整数 (如 50)；2D 为逗号分隔 (如 30,30)",
    )
    parser.add_argument(
        "--smooth-axes",
        type=str,
        default="z",
        metavar="AXES",
        help="势场平滑方向：默认 z；可指定 x,y,z 或 x,y 或 x；指定 none 关闭滤波",
    )
    parser.add_argument(
        "--smooth-sg",
        type=str,
        default="11,3",
        metavar="WINDOW,POLY",
        help="Savitzky-Golay 滤波器参数：窗口长度与多项式阶数，逗号分隔，默认 11,3",
    )
    args = parser.parse_args()

    from Interface.cli import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_CSV_PATH,
        DEFAULT_CONFIG_DIR,
        DEFAULT_CSV_DIR,
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
        field_settings.voltage_list = build_voltage_list(
            {"voltage_list": []}, n_voltage, cfg
        )

    # 势场平滑（默认沿 z；--smooth-axes none 关闭）
    raw_smooth = args.smooth_axes or ""
    if raw_smooth.strip().lower() != "none":
        axes_parts = [a.strip().lower() for a in raw_smooth.split(",") if a.strip()]
        valid_axes = [a for a in axes_parts if a in "xyz"]
        if valid_axes:
            try:
                sg_parts = [p.strip() for p in args.smooth_sg.split(",")]
                wl = int(sg_parts[0]) if sg_parts else 11
                poly = int(sg_parts[1]) if len(sg_parts) >= 2 else 3
            except (ValueError, IndexError):
                wl, poly = 11, 3
            grid_voltage = apply_savgol_smooth(
                grid_coord, grid_voltage, tuple(valid_axes), window_length=wl, polyorder=poly
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
    yr_um = parse_range(args.y_range)
    zr_um = parse_range(args.z_range)
    dl = cfg.dl

    def parse_freq_scan_n(s: str, dim: int):
        parts = [p.strip() for p in s.split(",")]
        if dim == 1:
            return int(parts[0]) if parts else 50
        return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (50, 50)

    if args.freq_scan:
        scan_parts = [p.strip().lower() for p in args.freq_scan.split(",")]
        if len(scan_parts) == 1:
            axis = scan_parts[0]
            if axis not in "xyz":
                raise ValueError("--freq-scan 须为 x, y, z 或 x,y 等形式")
            scan_range = xr_um if axis == "x" else (yr_um if axis == "y" else zr_um)
            n_scan = int(parse_freq_scan_n(args.freq_scan_n, 1))
            coord_um, f_x, f_y, f_z = compute_freq_scan_1d(
                potential_interps,
                field_interps,
                voltage_list,
                cfg,
                axis,
                scan_range,
                xc_um,
                yc_um,
                zc_um,
                x_range_um=xr_um,
                y_range_um=yr_um,
                z_range_um=zr_um,
                n_scan=n_scan,
                n_fit_pts=args.freq_n_pts,
                fit_degree=args.freq_fit_degree,
            )
            plot_freq_scan_1d(
                coord_um, f_x, f_y, f_z, axis, (xc_um, yc_um, zc_um), args.out
            )
        elif len(scan_parts) == 2:
            a1, a2 = scan_parts[0], scan_parts[1]
            if a1 not in "xyz" or a2 not in "xyz" or a1 == a2:
                raise ValueError("--freq-scan 双轴须为两个不同坐标 (如 x,y)")
            n_scan = parse_freq_scan_n(args.freq_scan_n, 2)
            cc1_um, cc2_um, f_x_2d, f_y_2d, f_z_2d = compute_freq_scan_2d(
                potential_interps,
                field_interps,
                voltage_list,
                cfg,
                (a1, a2),
                xr_um,
                yr_um,
                zr_um,
                xc_um,
                yc_um,
                zc_um,
                n_scan=n_scan,
                n_fit_pts=args.freq_n_pts,
                fit_degree=args.freq_fit_degree,
            )
            const_idx = next(i for i, c in enumerate("xyz") if c not in (a1, a2))
            const_axis = "xyz"[const_idx]
            const_val = [xc_um, yc_um, zc_um][const_idx]
            plot_freq_scan_2d(
                cc1_um,
                cc2_um,
                f_x_2d,
                f_y_2d,
                f_z_2d,
                (a1, a2),
                const_val,
                const_axis,
                args.mode,
                args.out,
            )
        else:
            raise ValueError("--freq-scan 须为单轴 (x/y/z) 或双轴 (如 x,y)")
        return

    if args.freq:
        freqs = compute_trap_freqs_at_point(
            potential_interps,
            field_interps,
            voltage_list,
            cfg,
            xc_um,
            yc_um,
            zc_um,
            x_range_um=xr_um,
            y_range_um=yr_um,
            z_range_um=zr_um,
            n_pts=args.freq_n_pts,
            fit_degree=args.freq_fit_degree,
        )
        print(
            "Trap frequencies at (x=%.1f, y=%.1f, z=%.1f) μm:"
            % (xc_um, yc_um, zc_um)
        )
        for k, v in freqs.items():
            if np.isnan(v):
                print("  %s: N/A (k2<=0 or fit failed)" % k)
            else:
                print("  %s: %.4f MHz" % (k, v))
        return

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

    xc = um_to_norm(xc_um, dl)
    yc = um_to_norm(yc_um, dl)
    zc = um_to_norm(zc_um, dl)
    xr = (um_to_norm(xr_um[0], dl), um_to_norm(xr_um[1], dl))

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
            fit_degree=args.fit,
            out_path=args.out,
        )
    elif len(vary_parts) == 2:
        a1, a2 = vary_parts[0], vary_parts[1]
        if a1 not in "xyz" or a2 not in "xyz" or a1 == a2:
            raise ValueError("--vary 须为两个不同坐标 x,y,z")
        yr = (um_to_norm(yr_um[0], dl), um_to_norm(yr_um[1], dl))
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
