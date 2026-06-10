"""
trap_stability CLI：计算 Mathieu 稳定性参数 (a, q)

用法:
    python -m trap_stability --csv <csv> --config <json> [options]
    python -m trap_stability --direct --rf-freq <MHz> --r0 <um> --V0 <V> [options]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# 默认路径（与 Interface/cli.py 一致）
DEFAULT_CSV_DIR = "data"
DEFAULT_CONFIG_DIR = "FieldConfiguration/configs"


def _resolve_path(arg: str, default_dir: str, root: Path) -> str:
    """路径解析：仅文件名时在默认目录查找"""
    if not arg:
        return ""
    p = Path(arg)
    if not p.is_absolute() and "/" not in arg and "\\" not in arg:
        return str(root / default_dir / arg)
    return str(root / arg) if not p.is_absolute() else arg


def _parse_range(s: str) -> tuple[float, float]:
    """解析 'lo,hi' 格式的范围字符串"""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"范围格式应为 'lo,hi'，收到 '{s}'")
    return float(parts[0]), float(parts[1])


def _parse_3floats(s: str) -> tuple[float, float, float]:
    """解析 'x,y,z' 格式的三浮点数"""
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"格式应为 'x,y,z'，收到 '{s}'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算离子阱 Mathieu 稳定性参数 (a, q) 和 secular 频率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # CSV + JSON 模式（从实际场几何计算）
  python -m trap_stability --csv data/monolithic20241118.csv --config default.json

  # 仅传文件名时自动在默认目录查找
  python -m trap_stability --csv default.csv --config default.json

  # 指定陷阱中心和离子种类
  python -m trap_stability --csv default.csv --config default.json --center 0,0,0 --species Ca40+

  # Direct 模式（理想四极阱教科书公式）
  python -m trap_stability --direct --rf-freq 35.28 --r0 700 --V0 275 --U 1

  # 输出 JSON
  python -m trap_stability --direct --rf-freq 35.28 --r0 700 --V0 275 --out result.json
""",
    )

    # ---- 互斥输入模式 ----
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--csv",
        type=str,
        default=None,
        help="电场 CSV 文件路径；可仅传文件名(如 monolithic20241118.csv)则自动在 data/ 下查找",
    )
    mode_group.add_argument(
        "--direct",
        action="store_true",
        default=False,
        help="使用理想四极阱教科书公式（需配合 --rf-freq, --r0, --V0）",
    )

    # ---- Direct 模式参数 ----
    direct_group = parser.add_argument_group("Direct 模式参数")
    direct_group.add_argument(
        "--rf-freq",
        type=float,
        default=None,
        metavar="MHz",
        help="RF 驱动频率 (MHz)，direct 模式必填",
    )
    direct_group.add_argument(
        "--r0",
        type=float,
        default=None,
        metavar="UM",
        help="特征尺寸 r₀ (μm)，direct 模式必填",
    )
    direct_group.add_argument(
        "--V0",
        type=float,
        default=None,
        metavar="V",
        help="RF 零-峰值电压幅度 (V)，direct 模式必填",
    )
    direct_group.add_argument(
        "--U",
        type=float,
        default=0.0,
        metavar="V",
        help="DC 端帽电压 (V)，默认 0",
    )
    direct_group.add_argument(
        "--geometry",
        type=str,
        default="linear",
        choices=["linear", "3d"],
        help="阱几何类型: linear(默认) 或 3d",
    )

    # ---- CSV 模式参数 ----
    csv_group = parser.add_argument_group("CSV+JSON 模式参数")
    csv_group.add_argument(
        "--config",
        type=str,
        default="",
        help="电极电压 JSON 路径；可仅传文件名(如 default.json)则自动在 configs/ 下查找",
    )
    csv_group.add_argument(
        "--center",
        type=str,
        default=None,
        metavar="X,Y,Z",
        help="陷阱中心坐标 (μm)，如 '0,0,0'；不指定时自动检测",
    )
    csv_group.add_argument(
        "--x-range",
        type=str,
        default="-50,50",
        metavar="LO,HI",
        help="x 轴拟合范围 (μm)，默认 -50,50",
    )
    csv_group.add_argument(
        "--y-range",
        type=str,
        default="-20,20",
        metavar="LO,HI",
        help="y 轴拟合范围 (μm)，默认 -20,20",
    )
    csv_group.add_argument(
        "--z-range",
        type=str,
        default="-150,150",
        metavar="LO,HI",
        help="z 轴拟合范围 (μm)，默认 -150,150",
    )
    csv_group.add_argument(
        "--fit-degree",
        type=int,
        default=2,
        help="多项式拟合阶数 (2 或 4)，默认 2",
    )
    csv_group.add_argument(
        "--n-fit-pts",
        type=int,
        default=200,
        help="每轴采样点数，默认 200",
    )
    csv_group.add_argument(
        "--smooth-axes",
        type=str,
        default="z",
        metavar="AXES",
        help="势场平滑方向: z(默认); none 关闭",
    )
    csv_group.add_argument(
        "--smooth-sg",
        type=str,
        default="11,3",
        metavar="WINDOW,POLY",
        help="Savitzky-Golay 滤波参数: 窗口,阶数，默认 11,3",
    )

    # ---- 共享参数 ----
    parser.add_argument(
        "--species",
        type=str,
        default="Ba135+",
        help="离子种类，默认 Ba135+；支持 Ba135+, Ba138+, Yb171+, Ca40+, Sr88+, Mg24+, Be9+",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON 输出路径（可选）",
    )

    return parser


def _print_report(result: "StabilityResult") -> None:
    """打印人类可读的分析报告"""
    print("=" * 60)
    print("  Ion Trap Stability Analysis")
    print("=" * 60)
    print(f"  Species:       {result.species_name} ({result.mass_amu:.3f} amu)")
    print(f"  RF frequency:  {result.freq_rf_MHz:.2f} MHz "
          f"(Omega = {result.omega_rf:.3e} rad/s)")
    print()
    print("  Mathieu Parameters:")
    print(f"     axis       a              q")
    print(f"     x     {result.a_x:14.6e}  {result.q_x:14.6e}")
    print(f"     y     {result.a_y:14.6e}  {result.q_y:14.6e}")
    print(f"     z     {result.a_z:14.6e}  {result.q_z:14.6e}")
    print()
    print("  Secular Frequencies (adiabatic approximation):")
    print(f"     f_x = {result.f_sec_x:8.4f} MHz    "
          f"f_y = {result.f_sec_y:8.4f} MHz    "
          f"f_z = {result.f_sec_z:8.4f} MHz")
    print()
    if any(k2 != 0.0 for k2 in result.k2_dc.values()):
        print("  Trap Frequencies (total potential curvature):")
        print(f"     f_x = {result.f_trap_x:8.4f} MHz    "
              f"f_y = {result.f_trap_y:8.4f} MHz    "
              f"f_z = {result.f_trap_z:8.4f} MHz")
        print()
    status = "[STABLE]" if result.is_stable else "[UNSTABLE]"
    print(f"  Stability: {status} ({result.stability_note})")
    print("=" * 60)


def _result_to_dict(result: "StabilityResult", mode: str, **extra: object) -> dict:
    """将 StabilityResult 转为可序列化字典"""
    return {
        "species": result.species_name,
        "mass_amu": result.mass_amu,
        "omega_rf_rad_s": result.omega_rf,
        "freq_rf_MHz": result.freq_rf_MHz,
        "a": {"x": result.a_x, "y": result.a_y, "z": result.a_z},
        "q": {"x": result.q_x, "y": result.q_y, "z": result.q_z},
        "f_secular_MHz": {
            "x": result.f_sec_x, "y": result.f_sec_y, "z": result.f_sec_z,
        },
        "f_trap_MHz": {
            "x": result.f_trap_x, "y": result.f_trap_y, "z": result.f_trap_z,
        },
        "k2_dc_V_per_um2": result.k2_dc,
        "k2_rf_amp_V_per_um2": result.k2_rf_amp,
        "is_stable": result.is_stable,
        "stability_note": result.stability_note,
        "mode": mode,
        **extra,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI 入口"""
    # 确保 sys.path 包含项目根
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    if (root / "build").exists():
        sys.path.insert(0, str(root / "build"))

    parser = create_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, os.environ.get("ISM_LOG_LEVEL", "WARNING")),
        format="%(levelname)s: %(message)s",
    )

    # 解析物种
    from FieldConfiguration.ion_species import ION_SPECIES
    if args.species not in ION_SPECIES:
        parser.error(
            f"未知物种: {args.species}，可选: {', '.join(ION_SPECIES.keys())}"
        )
    species = ION_SPECIES[args.species]

    # ---- Direct 模式 ----
    if args.direct:
        if args.rf_freq is None or args.r0 is None or args.V0 is None:
            parser.error("--direct 模式需要 --rf-freq, --r0, --V0")
        from .stability import compute_stability_direct

        result = compute_stability_direct(
            rf_freq_MHz=args.rf_freq,
            r0_um=args.r0,
            V0=args.V0,
            U=args.U,
            species=species,
            geometry=args.geometry,
        )
        _print_report(result)

        if args.out:
            d = _result_to_dict(
                result,
                mode="direct",
                rf_freq_MHz=args.rf_freq,
                r0_um=args.r0,
                V0=args.V0,
                U=args.U,
                geometry=args.geometry,
            )
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.out}")
        return

    # ---- CSV + JSON 模式 ----
    csv_path = _resolve_path(args.csv, DEFAULT_CSV_DIR, root)
    config_path = _resolve_path(args.config, DEFAULT_CONFIG_DIR, root) if args.config else ""

    if not Path(csv_path).exists():
        parser.error(f"CSV 文件不存在: {csv_path}")

    from FieldConfiguration.constants import init_from_config
    from FieldConfiguration.loader import load_field_config, build_voltage_list
    from FieldParser.csv_reader import read as read_csv
    from FieldParser.calc_field import calc_potential, calc_field
    from field_visualize.core import apply_savgol_smooth
    from .stability import compute_stability_from_field, find_trap_center

    # 加载 Config
    cfg, config_dict = init_from_config(config_path, mass_amu=species.mass_amu)

    # 解析 CSV
    grid_coord, grid_voltage = read_csv(
        csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV
    )

    # 平滑
    raw_smooth = getattr(args, "smooth_axes", "z")
    smooth_axes = None
    if raw_smooth and raw_smooth.strip().lower() != "none":
        axes_parts = [a.strip().lower() for a in raw_smooth.split(",") if a.strip()]
        smooth_axes = tuple(a for a in axes_parts if a in "xyz") or None

    if smooth_axes:
        win, poly = args.smooth_sg.split(",")
        grid_voltage = apply_savgol_smooth(
            grid_coord, grid_voltage, smooth_axes,
            window_length=int(win), polyorder=int(poly),
        )

    # 构建插值器
    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)

    # 构建电压列表
    if config_dict:
        n_voltage = grid_voltage.shape[1]
        voltage_list = build_voltage_list(config_dict, n_voltage, cfg)
    else:
        from utils import Voltage, constant
        voltage_list = [Voltage(f"U{i+1}", 0.0, constant(1.0), 0.0)
                        for i in range(grid_voltage.shape[1])]

    # 陷阱中心
    if args.center:
        xc, yc, zc = _parse_3floats(args.center)
    else:
        print("Auto-detecting trap center...")
        xc, yc, zc = find_trap_center(
            potential_interps, field_interps, voltage_list, cfg,
        )
        print(f"  Found center: ({xc:.2f}, {yc:.2f}, {zc:.2f}) um")

    # 拟合范围
    x_range = _parse_range(args.x_range)
    y_range = _parse_range(args.y_range)
    z_range = _parse_range(args.z_range)

    result = compute_stability_from_field(
        potential_interps=potential_interps,
        field_interps=field_interps,
        voltage_list=voltage_list,
        cfg=cfg,
        species=species,
        center_um=(xc, yc, zc),
        fit_range_um=(x_range, y_range, z_range),
        n_pts=args.n_fit_pts,
        fit_degree=args.fit_degree,
    )
    _print_report(result)

    if args.out:
        d = _result_to_dict(
            result,
            mode="csv_json",
            csv=str(csv_path),
            config=str(config_path),
            center_um=[xc, yc, zc],
            fit_degree=args.fit_degree,
            n_fit_pts=args.n_fit_pts,
        )
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.out}")
