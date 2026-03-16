"""
批量运行 CPU-CUDA 轨迹距离比较并保存结果。

复用 tests/test_cpu_cuda_trajectory_distance.py 中的演化逻辑：
- _build_force_from_parsed
- run_comparison
- plot_distance_vs_time（可选）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from setup_path import ensure_build_in_path

ensure_build_in_path(_ROOT)

try:
    import ionsim
except ImportError as e:
    raise ImportError("请先编译 ionsim: cd ism-main && cmake -B build && cmake --build build") from e

from Interface import cli
from tests.test_cpu_cuda_trajectory_distance import (
    _build_force_from_parsed,
    plot_distance_vs_time,
    run_comparison,
)


def _parse_number_list(raw: str, cast_type, arg_name: str) -> list:
    values = []
    for x in raw.split(","):
        item = x.strip()
        if not item:
            continue
        try:
            values.append(cast_type(item))
        except ValueError as exc:
            raise ValueError(f"{arg_name} 中存在非法值: {item}") from exc
    if not values:
        raise ValueError(f"{arg_name} 不能为空")
    return values


def _build_main_args(
    n_ions: int,
    time_us: float,
    csv_path: str,
    config_path: str,
    alpha: float,
    isotope: str | None,
):
    arg_list = ["--N", str(n_ions), "--time", str(time_us), "--csv", csv_path]
    if config_path:
        arg_list += ["--config", config_path]
    if alpha != 0.0:
        arg_list += ["--alpha", str(alpha)]
    if isotope is not None:
        arg_list += ["--isotope", isotope]
    return cli.create_parser().parse_args(arg_list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按给定 N 与 time 列表批量运行 CPU-CUDA 轨迹距离比较并保存结果"
    )
    parser.add_argument(
        "--Ns",
        type=str,
        required=True,
        help="离子数列表，逗号分隔，例如: 10,20,50",
    )
    parser.add_argument(
        "--times",
        type=str,
        required=True,
        help="演化时间(μs)列表，逗号分隔，例如: 5,10,20",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/monolithic20241118.csv",
        help="电场 CSV 路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="电压配置 JSON（如 circle.json），不传则用默认",
    )
    parser.add_argument("--n_steps", type=int, default=None, help="积分步数，默认自动")
    parser.add_argument("--alpha", type=float, default=0.0, help="同位素参杂比例")
    parser.add_argument(
        "--isotope",
        type=str,
        default=None,
        choices=["Ba133", "Ba134", "Ba135", "Ba136", "Ba137", "Ba138"],
        help="单同位素模式：指定同位素种类",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tests/test_results/traj_dist_batch",
        help="输出目录（默认保存 png 图片，可选附加保存数据文件）",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="额外保存每组参数的 npz/json，并生成 summary.csv",
    )
    args = parser.parse_args()

    n_list = _parse_number_list(args.Ns, int, "--Ns")
    time_list = _parse_number_list(args.times, float, "--times")

    if not getattr(ionsim, "cuda_available", False):
        print("CUDA 不可用，无法比较 CPU-CUDA 轨迹距离")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    total_jobs = len(n_list) * len(time_list)
    done = 0

    for n_ions in n_list:
        for time_us_input in time_list:
            done += 1
            print(f"[{done}/{total_jobs}] N={n_ions}, time={time_us_input} μs")

            main_args = _build_main_args(
                n_ions=n_ions,
                time_us=time_us_input,
                csv_path=args.csv,
                config_path=args.config,
                alpha=args.alpha,
                isotope=args.isotope,
            )

            try:
                parsed = cli.parse_and_build(main_args, _ROOT)
            except FileNotFoundError as e:
                print(f"  跳过：配置或数据文件不存在: {e}")
                continue

            force = _build_force_from_parsed(parsed, _ROOT)
            result = run_comparison(
                parsed,
                force,
                duration_dt=parsed.params.duration,
                n_steps=args.n_steps,
            )

            dist_um = result["dist"] * parsed.config.dl * 1e6
            t_dt = result["t"]
            t_us = t_dt * parsed.config.dt * 1e6
            sim_time_us = result["duration"] * parsed.config.dt * 1e6
            max_dist = float(np.max(dist_um))
            mean_dist = float(np.mean(dist_um))

            use_isotope = args.alpha > 0 or args.isotope is not None
            suffix = "_isotope" if use_isotope else ""
            stem = f"N{result['n_ions']}_{sim_time_us:.1f}{suffix}"
            plot_path = out_dir / f"{stem}.png"
            import matplotlib
            matplotlib.use("Agg")
            plot_distance_vs_time(
                {
                    "t": t_dt,
                    "dist": dist_um,
                    "n_ions": result["n_ions"],
                    "duration": result["duration"],
                },
                out_path=str(plot_path),
                dt=parsed.config.dt,
            )

            npz_name = ""
            json_name = ""
            if args.save_data:
                npz_path = out_dir / f"{stem}.npz"
                json_path = out_dir / f"{stem}.json"
                np.savez_compressed(
                    npz_path,
                    t_dt=t_dt,
                    t_us=t_us,
                    dist_um=dist_um,
                    n_ions=result["n_ions"],
                    duration_dt=result["duration"],
                    duration_us=sim_time_us,
                )

                meta = {
                    "N": int(result["n_ions"]),
                    "input_time_us": float(time_us_input),
                    "sim_time_us": float(sim_time_us),
                    "duration_dt": float(result["duration"]),
                    "max_dist_um": max_dist,
                    "mean_dist_um": mean_dist,
                    "npz": npz_path.name,
                }
                json_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                npz_name = npz_path.name
                json_name = json_path.name

            summary_rows.append(
                {
                    "N": result["n_ions"],
                    "input_time_us": time_us_input,
                    "sim_time_us": sim_time_us,
                    "max_dist_um": max_dist,
                    "mean_dist_um": mean_dist,
                    "npz": npz_name,
                    "json": json_name,
                    "plot": plot_path.name,
                }
            )

    if args.save_data:
        summary_path = out_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "N",
                    "input_time_us",
                    "sim_time_us",
                    "max_dist_um",
                    "mean_dist_um",
                    "npz",
                    "json",
                    "plot",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"完成，已保存图片与数据；汇总文件: {summary_path}")
    else:
        print(f"完成，已保存 {len(summary_rows)} 张图片到: {out_dir}")


if __name__ == "__main__":
    main()
