"""
param_scan 共享 log CSV 读写。

列定义在扫描与热力图间共用，保证"绘图仅依赖 log"的解耦约定：

    run_idx, x2_coeff, x4_coeff, seed, reached_threshold, sim_time_us, sigma_y_final_um

  * run_idx            —— 运行序号（1 起，与 .npy 文件名一致）
  * x2_coeff, x4_coeff —— 当轮 x²/x⁴ 多项式系数 (V)
  * seed               —— 当轮随机种子（初态可复现）
  * reached_threshold  —— 1/0，50µs 内是否达到 σ_y < 阈值
  * sim_time_us        —— 达标→跨越时刻 (µs)；未达标→TIME_MAX_US (50.0)
  * sigma_y_final_um   —— 模拟结束时的 σ_y (µm)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

COLUMNS = [
    "run_idx",
    "x2_coeff",
    "x4_coeff",
    "seed",
    "reached_threshold",
    "sim_time_us",
    "sigma_y_final_um",
]


def write_header(path: str | Path) -> None:
    """log 不存在或为空时写表头（幂等）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size == 0:
        with p.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)


def append_row(
    path: str | Path,
    *,
    run_idx: int,
    x2_coeff: float,
    x4_coeff: float,
    seed: int,
    reached_threshold: bool,
    sim_time_us: float,
    sigma_y_final_um: float,
) -> None:
    """追加一行并立即 flush（流式落盘，抗崩溃 / 支持断点续跑）。"""
    row = {
        "run_idx": int(run_idx),
        "x2_coeff": f"{float(x2_coeff):.12g}",
        "x4_coeff": f"{float(x4_coeff):.12g}",
        "seed": int(seed),
        "reached_threshold": 1 if reached_threshold else 0,
        "sim_time_us": f"{float(sim_time_us):.6g}",
        "sigma_y_final_um": f"{float(sigma_y_final_um):.6g}",
    }
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
        f.flush()


def read_log(path: str | Path) -> list[dict[str, Any]]:
    """读 log；数值列转 float，reached_threshold 转 int。文件缺失返回 []。"""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "run_idx": int(r["run_idx"]),
                    "x2_coeff": float(r["x2_coeff"]),
                    "x4_coeff": float(r["x4_coeff"]),
                    "seed": int(r["seed"]),
                    "reached_threshold": int(r["reached_threshold"]),
                    "sim_time_us": float(r["sim_time_us"]),
                    "sigma_y_final_um": float(r["sigma_y_final_um"]),
                }
            )
    return rows
