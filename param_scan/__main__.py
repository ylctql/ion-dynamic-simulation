"""
CLI 派发：``python -m param_scan scan|plot``

  scan  —— 跑 x²/x⁴ 扫描（动力学，写 .npy + scan.log）
  plot  —— 仅读 scan.log 画热力图（与动力学解耦）
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="param_scan")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="跑 x²/x⁴ 扫描")
    s.add_argument("--x2", required=True, help="x² 系数：min,max,n 或 v1,v2,...")
    s.add_argument("--x4", required=True, help="x⁴ 系数：min,max,n 或 v1,v2,...")
    s.add_argument("--out-dir", default=None, help="覆盖默认输出目录")
    s.add_argument("--base-seed", type=int, default=None)
    s.add_argument("--fixed-seed", action="store_true",
                   help="整网格共用一个初态（受控扫描）；默认每轮 seed=base_seed+idx")
    s.add_argument("--force", action="store_true", help="覆盖已存在 .npy（log 仍追加）")
    s.add_argument("--new-log", action="store_true",
                   help="截断 scan.log 重开一轮，并强制重跑所有点（扫描不同参数集时用）")
    s.add_argument("--init-range", default=None, help="X,Y,Z 随机盒 ±µm，覆盖默认常量")

    pl = sub.add_parser("plot", help="由 scan.log 画热力图")
    pl.add_argument("--log", required=True)
    pl.add_argument("--out", default=None)
    pl.add_argument("--value-col", default="sigma_y_final_um")
    pl.add_argument("--log-color", action="store_true")
    pl.add_argument("--title", default=None)

    args = p.parse_args()
    if args.cmd == "scan":
        from .scan import (
            BASE_SEED, FIXED_SEED, INIT_RANGE_UM, OUT_DIR, parse_axis_spec, run_scan,
        )

        init_range = INIT_RANGE_UM
        if args.init_range:
            parts = [float(v) for v in args.init_range.split(",")]
            if len(parts) != 3:
                raise SystemExit("--init-range 须为 X,Y,Z 三个数")
            init_range = (parts[0], parts[1], parts[2])

        run_scan(
            parse_axis_spec(args.x2),
            parse_axis_spec(args.x4),
            args.out_dir or OUT_DIR,
            force=args.force,
            new_log=args.new_log,
            base_seed=args.base_seed if args.base_seed is not None else BASE_SEED,
            fixed_seed=args.fixed_seed or FIXED_SEED,
            init_range_um=init_range,
        )
    elif args.cmd == "plot":
        from .heatmap import plot_heatmap

        plot_heatmap(
            args.log, args.out,
            value_col=args.value_col, log_color=args.log_color, title=args.title,
        )


if __name__ == "__main__":
    main()
