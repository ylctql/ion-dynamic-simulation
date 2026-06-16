"""
motion_analysis CLI：从 continuous_sampling 轨迹测量 micromotion。

用法:
    python -m motion_analysis <run_dir> --csv <csv> --config <json> [options]

示例:
    python -m motion_analysis continuous_sampling/t030.00_interval0.08_step10 \\
        --csv monolithic20241118.csv --config default.json --out mm.json --plot-dir plots

采集（产生 run_dir）:
    python main.py --N 1 --time 20 --continuous-sampling \\
        --continuous-sampling-frames 3000 --interval 0.08 --step 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CSV_DIR = "data"
DEFAULT_CONFIG_DIR = "FieldConfiguration/configs"


def _resolve_path(arg: str, default_dir: str, root: Path) -> str:
    """路径解析：仅文件名时在默认目录查找（与 trap_stability.cli 一致）。"""
    if not arg:
        return ""
    p = Path(arg)
    if not p.is_absolute() and "/" not in arg and "\\" not in arg:
        return str(root / default_dir / arg)
    return str(root / arg) if not p.is_absolute() else arg


def _parse_axes(s: str) -> tuple[str, ...]:
    parts = [a.strip().lower() for a in s.split(",") if a.strip()]
    valid = [a for a in parts if a in ("x", "y", "z")]
    if not valid:
        raise argparse.ArgumentTypeError(f"--axes 无有效项，收到 '{s}'")
    return tuple(valid)


def _parse_ions(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError(f"--ions 解析为空: '{s}'")
    return out


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 continuous_sampling 轨迹数值测量 RF micromotion 与 q_eff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=str, help="continuous_sampling 输出目录")
    parser.add_argument(
        "--csv", type=str, required=True,
        help="电场 CSV（RF 频率与交叉验证用）；可仅传文件名自动在 data/ 下查找",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="电压 JSON（解析 RF 频率）；可仅传文件名自动在 configs/ 下查找",
    )
    parser.add_argument("--species", type=str, default="Ba135+",
                        help="离子种类，默认 Ba135+")
    parser.add_argument("--axes", type=_parse_axes, default=("x", "y", "z"),
                        help="分析轴，逗号分隔，默认 x,y,z")
    parser.add_argument("--ions", type=_parse_ions, default=None,
                        help="指定离子索引（逗号分隔），默认全部")
    parser.add_argument("--window-us", type=float, default=None,
                        help="phase-folding 滑窗长度 (us)，默认 max(3·T_RF, 0.3)")
    parser.add_argument("--n-phase-bins", type=int, default=32,
                        help="相位 bin 数，默认 32")
    parser.add_argument("--secular-freq", type=float, default=0.5,
                        help="secular 频率估计 (MHz)，用于采样总时长校验，默认 0.5")
    parser.add_argument("--warmup-us", type=float, default=None,
                        help="手动丢弃前 N µs（相对 t[0]）；优先级低于 --trim-start-us")
    parser.add_argument("--trim-start-us", type=float, default=None,
                        help="手动指定稳态起点（绝对时刻 µs）；优先级最高")
    parser.add_argument("--no-auto-trim", action="store_true",
                        help="关闭自动瞬态检测（不裁剪）；默认自动检测开")
    parser.add_argument("--warmup-tol", type=float, default=0.1,
                        help="自动检测相对收敛容差，默认 0.1")
    parser.add_argument("--warmup-periods", type=int, default=3,
                        help="自动检测窗覆盖的 secular 周期数，默认 3")
    parser.add_argument("--center", type=str, default=None,
                        help="阱中心 (um)，如 '0,0,0'；不指定时自动检测")
    parser.add_argument("--smooth-axes", type=str, default="z",
                        help="交叉验证中场平滑方向，默认 z；none 关闭")
    parser.add_argument("--smooth-sg", type=str, default="11,3",
                        help="Savitzky-Golay 窗口,阶数，默认 11,3")
    parser.add_argument("--no-cross-check", action="store_true",
                        help="跳过 trap_stability 交叉验证")
    parser.add_argument("--lattice-show-theory", action="store_true",
                        help="在晶格 micromotion 图上叠加理论 β=|q_th|/2·|x−x_null| "
                             "比对竖线（绿色虚线），需 cross-check 未关闭")
    parser.add_argument("--out", type=str, default=None,
                        help="JSON 输出路径")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="图输出目录（无头保存 PNG）")
    parser.add_argument("--show", action="store_true",
                        help="弹出交互窗口显示图（可缩放/平移/另存）；与 --plot-dir "
                             "可共存；需 GUI 后端（PyQt5/tkinter + WSLg/X11）")
    return parser


def _parse_smooth(args) -> tuple[tuple[str, ...] | None, tuple[int, int]]:
    raw = args.smooth_axes.strip().lower()
    if not raw or raw == "none":
        return None, (11, 3)
    axes = tuple(a for a in raw.split(",") if a in "xyz") or None
    sg = args.smooth_sg.split(",")
    return axes, (int(sg[0]), int(sg[1]) if len(sg) > 1 else 3)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    if (root / "build").exists():
        sys.path.insert(0, str(root / "build"))

    parser = create_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, os.environ.get("ISM_LOG_LEVEL", "INFO")),
        format="%(levelname)s: %(message)s",
    )

    from FieldConfiguration.ion_species import ION_SPECIES
    if args.species not in ION_SPECIES:
        parser.error(f"未知物种: {args.species}，可选: {', '.join(ION_SPECIES.keys())}")

    csv_path = _resolve_path(args.csv, DEFAULT_CSV_DIR, root)
    config_path = _resolve_path(args.config, DEFAULT_CONFIG_DIR, root)
    if not Path(csv_path).exists():
        parser.error(f"CSV 不存在: {csv_path}")
    if not Path(config_path).exists():
        parser.error(f"config 不存在: {config_path}")

    from .micromotion import analyze_run, cross_check_q, report_to_dict

    report = analyze_run(
        args.run_dir,
        csv_path=csv_path,
        config_path=config_path,
        species=args.species,
        axes=args.axes,
        ions=args.ions,
        window_us=args.window_us,
        n_phase_bins=args.n_phase_bins,
        secular_freq_MHz=args.secular_freq,
        warmup_us=args.warmup_us,
        trim_start_us=args.trim_start_us,
        no_auto_trim=args.no_auto_trim,
        warmup_tol=args.warmup_tol,
        warmup_periods=args.warmup_periods,
    )

    cross = None
    if not args.no_cross_check:
        center = None
        if args.center:
            parts = [float(x) for x in args.center.split(",")]
            if len(parts) != 3:
                parser.error("--center 需 'x,y,z'")
            center = (parts[0], parts[1], parts[2])
        smooth_axes, smooth_sg = _parse_smooth(args)
        cross = cross_check_q(
            report, csv_path=csv_path, config_path=config_path,
            species=args.species, center_um=center,
            smooth_axes=smooth_axes, smooth_sg=smooth_sg,
        )

    # 终端摘要
    print("=" * 60)
    print("  Micromotion Analysis")
    print("=" * 60)
    print(f"  Run:        {report.trajectory.run_dir}")
    print(f"  RF:         {report.trajectory.freq_rf_MHz:.3f} MHz "
          f"(dt={report.trajectory.dt_us*1e3:.2f} ns)")
    print(f"  Ions:       {len(report.ions)}  axes: {','.join(report.axes)}")
    print(f"  window:     {report.window_us:.4f} us, {report.n_phase_bins} phase bins")
    for ax in report.axes:
        ai = {"x": 0, "y": 1, "z": 2}[ax]
        col = report.q_eff[:, ai]
        col = col[np.isfinite(col)]
        if col.size:
            print(f"  axis {ax}: q_eff median={np.median(col):.5f} "
                  f"(min={col.min():.5f}, max={col.max():.5f})")
    # warmup 裁剪摘要
    trimmed = [res for res in report.results.values() if res.dropped_frames > 0]
    if trimmed:
        max_tstar = max(res.t_star_us for res in trimmed if np.isfinite(res.t_star_us))
        print(f"  warmup: {len(trimmed)}/{len(report.results)} (ion,axis) trimmed, "
              f"global max t*={max_tstar:.3f} us")
    else:
        reasons = sorted({res.warmup_reason for res in report.results.values()})
        print(f"  warmup: no trim ({', '.join(reasons)})")
    if cross is not None:
        print("  Cross-check (trap_stability q_theory):")
        for ax in report.axes:
            print(f"    {ax}: q_theory={cross.q_theory.get(ax, float('nan')):.5f}, "
                  f"q_meas_median={cross.q_measured_median.get(ax, float('nan')):.5f}, "
                  f"ratio={cross.ratio.get(ax, float('nan')):.3f}")
        print(f"    center=({cross.center_um[0]:.2f},{cross.center_um[1]:.2f},"
              f"{cross.center_um[2]:.2f}) um  stable={cross.is_stable}")
    print("=" * 60)

    # resid_rms 诊断：方法 C 残差过大提示 RF 频率与轨迹不匹配
    resid = [r.residual_rms_um for r in report.results.values()
             if np.isfinite(r.residual_rms_um)]
    betas = [float(np.median(r.beta_t)) for r in report.results.values()
             if np.all(np.isfinite(r.beta_t))]
    if resid and betas:
        mr, mb = float(np.median(resid)), float(np.median(betas))
        if mb > 0 and mr / mb > 0.3:
            logger.warning(
                "方法 C 残差偏大 (median resid_rms=%.3g um vs median β=%.3g um, "
                "ratio=%.2f)：可能 RF 频率与轨迹不匹配（请确认 --config 为采集时所用的"
                "配置），或运动偏离乘性模型",
                mr, mb, mr / mb,
            )

    if args.out:
        d = report_to_dict(report, cross)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.out}")

    if args.plot_dir or args.show:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")          # 仅保存：无头后端；--show 用默认 GUI 后端
        import matplotlib.pyplot as plt
        from .plots import (
            plot_qeff_histogram, plot_qeff_vs_displacement, plot_beta_vs_secular,
            plot_lattice_micromotion,
        )
        figures = [
            (plot_qeff_histogram(report), "qeff_histogram.png"),
            (plot_qeff_vs_displacement(report, cross), "qeff_vs_displacement.png"),
            (plot_beta_vs_secular(report, cross), "beta_vs_secular.png"),
            (plot_lattice_micromotion(report, cross=cross,
                                      show_theory=args.lattice_show_theory),
             "lattice_micromotion_x.png"),
        ]
        if args.plot_dir:
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            for fig, name in figures:
                p = plot_dir / name
                fig.savefig(p, dpi=150, bbox_inches="tight")
                print(f"Plot saved: {p}")
        if args.show:
            backend = matplotlib.get_backend()
            if backend.lower() == "agg":
                logger.warning(
                    "--show 当前后端为 Agg（无 GUI）无法弹窗；请安装 GUI 后端"
                    "（PyQt5/tkinter 等）并用 MPLBACKEND 指定，例如 "
                    "MPLBACKEND=Qt5Agg python -m motion_analysis ... --show"
                )
            print("弹出交互窗口，关闭窗口后程序退出 ...")
            plt.show()
        else:
            for fig, _ in figures:
                plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
