"""
field_optimize CLI：从目标阱频反推电极电压

用法:
    python -m field_optimize --csv <csv> --config <json> --target-freq fx fy fz [options]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


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
        description="优化电极电压以匹配目标阱频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python -m field_optimize --csv data/default.csv --config FieldConfiguration/configs/default.json \\
      --target-freq 2.0 3.0 0.1

  # 仅传文件名时自动在默认目录查找
  python -m field_optimize --csv default.csv --config default.json --target-freq 2.0 3.0 0.1

  # 同时优化 RF 幅值，增加对称性惩罚权重
  python -m field_optimize --csv default.csv --config default.json \\
      --target-freq 2.0 3.0 0.1 --optimize-rf-v0 --w-parity 0.5
""",
    )

    # 必选
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="电场 CSV 路径；可仅传文件名自动在 data/ 下查找",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="电压配置 JSON 路径；可仅传文件名自动在 FieldConfiguration/configs/ 下查找",
    )
    parser.add_argument(
        "--target-freq",
        type=float,
        nargs=3,
        required=True,
        metavar=("FX", "FY", "FZ"),
        help="目标阱频 (MHz): fx fy fz",
    )

    # 评估参数
    parser.add_argument(
        "--center", type=str, default="0,0,0",
        help="评估中心 x,y,z (µm)，默认 0,0,0",
    )
    parser.add_argument(
        "--x-range", type=str, default="-50,50",
        help="x 轴拟合范围 (µm)，默认 -50,50",
    )
    parser.add_argument(
        "--y-range", type=str, default="-20,20",
        help="y 轴拟合范围 (µm)，默认 -20,20",
    )
    parser.add_argument(
        "--z-range", type=str, default="-150,150",
        help="z 轴拟合范围 (µm)，默认 -150,150",
    )
    parser.add_argument(
        "--fit-degree", type=int, choices=[2, 4], default=2,
        help="多项式拟合阶数，默认 2",
    )
    parser.add_argument(
        "--n-fit-pts", type=int, default=200,
        help="每轴 1D 采样点数，默认 200",
    )

    # 权重
    parser.add_argument(
        "--w-freq", type=float, default=1.0,
        help="频率误差权重，默认 1.0",
    )
    parser.add_argument(
        "--w-parity", type=float, default=0.1,
        help="多项式奇偶性惩罚权重，默认 0.1",
    )
    parser.add_argument(
        "--w-offdiag", type=float, default=0.1,
        help="Hessian 离轴比惩罚权重，默认 0.1",
    )

    # 优化模式
    parser.add_argument(
        "--optimize-rf-v0", action="store_true",
        help="同时优化 RF 幅值 V0",
    )

    # 电压上下界
    parser.add_argument(
        "--v-bias-bounds", type=str, default="-100,100",
        help="DC 偏置电压上下界 (V)，默认 -100,100",
    )
    parser.add_argument(
        "--v0-rf-bounds", type=str, default="50,500",
        help="RF 幅值上下界 (V)，默认 50,500",
    )

    # 优化器参数
    parser.add_argument(
        "--maxiter", type=int, default=200,
        help="最大迭代次数，默认 200",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-8,
        help="收敛容限，默认 1e-8",
    )
    parser.add_argument(
        "--method", type=str, choices=["L-BFGS-B", "Nelder-Mead"], default="L-BFGS-B",
        help="优化方法，默认 L-BFGS-B",
    )

    # 对称性评估
    parser.add_argument(
        "--symmetry-fit-mode", type=str, default="quartic",
        help="3D 多项式拟合模式，默认 quartic",
    )
    parser.add_argument(
        "--symmetry-n-pts", type=int, default=6,
        help="3D 对称性评估每轴采样点数，默认 6",
    )

    # 势场平滑
    parser.add_argument(
        "--smooth-axes", type=str, default="z",
        help="SG 平滑方向 (x/y/z 或逗号组合)，默认 z；'none' 关闭",
    )
    parser.add_argument(
        "--smooth-sg", type=str, default="11,3",
        help="SG 平滑参数 window_length,polyorder，默认 11,3",
    )

    # 输出
    parser.add_argument(
        "--out", type=str, default="",
        help="输出 JSON 路径；默认自动生成",
    )

    return parser


def _print_report(result) -> None:
    """打印优化结果摘要"""
    print("\n" + "=" * 60)
    print("  电极电压优化结果")
    print("=" * 60)

    # 频率对比表
    print(f"\n{'':>12} {'目标':>10} {'优化前':>10} {'优化后':>10} {'误差%':>8}")
    print("-" * 52)
    target = result.target_freqs_MHz
    initial = result.initial_freqs_MHz
    final = result.optimized_freqs_MHz
    for axis in "xyz":
        key = f"f_{axis}"
        t = target[key]
        fi = initial[key]
        ff = final[key]
        err = abs(ff - t) / (t + 1e-30) * 100
        fi_str = f"{fi:.4f}" if not np.isnan(fi) else "  NaN  "
        ff_str = f"{ff:.4f}" if not np.isnan(ff) else "  NaN  "
        print(f"  f_{axis} (MHz) {t:>10.4f} {fi_str:>10} {ff_str:>10} {err:>7.3f}%")

    # 电压对比
    print(f"\n{'电极':>8} {'类型':>4} {'初始 V':>12} {'优化 V':>12} {'变化':>10}")
    print("-" * 52)
    for iv, ov in zip(result.initial_voltages, result.optimized_voltages):
        name = iv["name"]
        tp = iv["type"].upper()
        if tp == "DC":
            v_i = iv["V_bias"]
            v_o = ov["V_bias"]
            delta = v_o - v_i
            print(f"  {name:>6} {tp:>4} {v_i:>12.4f} {v_o:>12.4f} {delta:>+10.4f}")
        else:
            v_i = iv["V0"]
            v_o = ov["V0"]
            delta = v_o - v_i
            vb_i = iv["V_bias"]
            vb_o = ov["V_bias"]
            print(
                f"  {name:>6} {tp:>4} V0={v_i:.2f}→{v_o:.2f}"
                f"  V_bias={vb_i:.2f}→{vb_o:.2f}"
            )

    # 收敛信息
    print(f"\n收敛: {'成功' if result.success else '未收敛'} ({result.n_iterations} iter, {result.n_evaluations} eval)")
    print(f"目标函数: {result.initial_objective:.6e} → {result.final_objective:.6e}")
    print(f"信息: {result.message}")

    # 对称性
    if result.optimized_symmetry:
        sym = result.optimized_symmetry
        print("\n对称性指标:")
        if "s_parity_yz" in sym:
            print(
                f"  S_parity: yz={sym['s_parity_yz']:.4f}"
                f"  xz={sym['s_parity_xz']:.4f}"
                f"  xy={sym['s_parity_xy']:.4f}"
            )
        if "offdiag_ratio" in sym:
            print(f"  Hessian 离轴比: {sym['offdiag_ratio']:.6f}")

    print("=" * 60)


def _write_output_json(result, out_path: str, original_config: dict | None) -> None:
    """将优化结果写为可复用的 JSON 配置"""
    output: dict = {
        "_comment": "Optimized voltage configuration from field_optimize",
    }

    # 保留原始 g 值
    if original_config and "g" in original_config:
        output["g"] = original_config["g"]
    else:
        output["g"] = 0.1

    # voltage_list
    vlist = []
    for v in result.optimized_voltages:
        entry: dict = {"type": v["type"], "name": v["name"]}
        if v["type"] == "dc":
            entry["V_bias"] = round(v["V_bias"], 6)
        else:
            entry["V0"] = round(v["V0"], 4)
            entry["V_bias"] = round(v["V_bias"], 6)
            # 频率信息需要从原始配置获取
            if original_config:
                for orig_v in original_config.get("voltage_list", []):
                    if orig_v.get("name") == v["name"] and orig_v.get("type") == "rf":
                        entry["frequency"] = orig_v.get("frequency", 35.28)
                        break
        vlist.append(entry)
    output["voltage_list"] = vlist

    # 优化元数据
    output["_optimization"] = {
        "target_freq_MHz": [
            result.target_freqs_MHz["f_x"],
            result.target_freqs_MHz["f_y"],
            result.target_freqs_MHz["f_z"],
        ],
        "achieved_freq_MHz": [
            result.optimized_freqs_MHz["f_x"],
            result.optimized_freqs_MHz["f_y"],
            result.optimized_freqs_MHz["f_z"],
        ],
        "iterations": result.n_iterations,
        "n_evaluations": result.n_evaluations,
        "success": result.success,
        "timestamp": datetime.now().isoformat(),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n输出已保存到: {out_path}")


def main() -> None:
    # 路径设置
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import numpy as np

    from FieldConfiguration.constants import init_from_config
    from FieldConfiguration.loader import build_voltage_list, field_settings_from_config
    from FieldParser.calc_field import calc_field, calc_potential
    from FieldParser.csv_reader import read as read_csv
    from Interface.cli import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH, DEFAULT_CSV_DIR, DEFAULT_CSV_PATH
    from field_visualize.core import apply_savgol_smooth

    from .optimizer import optimize_voltages
    from .types import OptimizationConfig

    # 日志
    log_level = os.environ.get("ISM_LOG_LEVEL", "INFO")
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(levelname)s: %(message)s")

    # 解析参数
    parser = create_parser()
    args = parser.parse_args()

    # 路径解析
    def _resolve_path(arg: str, default_full: str, default_dir: str) -> str:
        p = Path(arg)
        if not p.is_absolute() and "/" not in arg and "\\" not in arg:
            return str(root / default_dir / arg)
        return str(root / arg) if not p.is_absolute() else arg

    config_path = _resolve_path(args.config, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_DIR)
    csv_path = _resolve_path(args.csv, DEFAULT_CSV_PATH, DEFAULT_CSV_DIR)

    # 初始化
    cfg, config_dict = init_from_config(config_path)
    grid_coord, grid_voltage = read_csv(
        csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV
    )
    n_voltage = grid_voltage.shape[1]

    # 势场平滑
    smooth_axes = tuple(a.strip() for a in args.smooth_axes.split(",") if a.strip()) if args.smooth_axes.lower() != "none" else ()
    if smooth_axes:
        sg_parts = args.smooth_sg.split(",")
        wl, po = int(sg_parts[0]), int(sg_parts[1])
        grid_voltage = apply_savgol_smooth(grid_coord, grid_voltage, smooth_axes, wl, po)

    # 构建插值器（一次）
    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)

    # 构建 voltage_list
    if config_dict:
        field_settings = field_settings_from_config(csv_path, config_path, n_voltage, cfg)
        voltage_list = field_settings.voltage_list
    else:
        voltage_list = build_voltage_list({"voltage_list": []}, n_voltage, cfg)

    # 解析 CLI 参数为 OptimizationConfig
    opt_config = OptimizationConfig(
        target_freq_MHz=tuple(args.target_freq),
        center_um=_parse_3floats(args.center),
        fit_range_um=(
            _parse_range(args.x_range),
            _parse_range(args.y_range),
            _parse_range(args.z_range),
        ),
        n_fit_pts=args.n_fit_pts,
        fit_degree=args.fit_degree,
        w_freq=args.w_freq,
        w_parity=args.w_parity,
        w_offdiag=args.w_offdiag,
        maxiter=args.maxiter,
        tol=args.tol,
        method=args.method,
        v_bias_bounds=_parse_range(args.v_bias_bounds),
        v0_rf_bounds=_parse_range(args.v0_rf_bounds),
        optimize_rf_v0=args.optimize_rf_v0,
        symmetry_fit_mode=args.symmetry_fit_mode,
        symmetry_n_pts=args.symmetry_n_pts,
    )

    # 运行优化
    result = optimize_voltages(
        potential_interps, field_interps, cfg, voltage_list, opt_config
    )

    # 报告
    _print_report(result)

    # 输出 JSON
    out_path = args.out
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"field_optimize/results/optimized_{ts}.json"
    _write_output_json(result, out_path, config_dict)
