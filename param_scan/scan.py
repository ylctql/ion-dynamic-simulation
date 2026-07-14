"""
poly-potential x²/x⁴ 系数扫描（动力学部分）。

对每组 (x², x⁴) 组合跑一次 N 离子动力学（VV + CUDA 等，见顶部常量）：
  * 固定 softmode.json 中其余系数（y²/z²/z⁴…），仅改 x²(coeffs[2,0,0]) 与 x⁴(coeffs[4,0,0])；
  * 初态用随机种子生成的均匀随机盒（seed 记入 log，可复现）；
  * 终止条件：σ_y < SIGMA_Y_THRESH_UM（提前 STOP，记录跨越时刻）或模拟达 TIME_MAX_US；
  * 存结束位置为 ``{idx:04d}.npy``（µm），并追加一行 scan.log。

架构（方案 B）：复用 main.py 的 backend 启停（``_create_backend_and_start``），
backend 在 ``time = 50µs`` 自动 STOP；σ_y 达标则由本模块的消费者提前 STOP。
``main.py`` / ``backend.py`` 不改动。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

# ============================================================================
# 顶部易改常量（与 N/device/calc-method 同级，按需修改）
# ============================================================================
N_IONS = 6000
DEVICE = "cuda"                         # "cpu" / "cuda"
CALC_METHOD = "VV"                      # "RK4" / "VV"
CONFIG_JSON = "1000.json"               # 提供 cfg(dt,dl,dV)；softmode.json 引用之
SOFTMODE_JSON = "softmode.json"         # 基底系数（固定项）来源（poly_potential/ 下）
TIME_MAX_US = 50.0                      # 模拟物理时间上限 (µs)
SIGMA_Y_THRESH_UM = 0.0005              # σ_y 早终止阈值 (µm)
INIT_RANGE_UM = (100.0, 100.0, 100.0)   # 种子化初态随机盒 ±范围(每轴 µm)；勿超 scale_um
GAMMA = None                            # None→FieldSettings.get_gamma() 默认(0.1)；可覆盖
STEP = 10                               # 每帧积分步数
INTERVAL = 1.0                          # 帧间隔(dt 单位)→决定 σ_y 检查粒度
BATCH = 50                              # 每批帧数
BASE_SEED = 0
FIXED_SEED = False                      # True→整网格共用一初态(受控扫描)；False→每轮 seed=BASE_SEED+idx
OUT_DIR = "param_scan/results"

ROOT = Path(__file__).resolve().parent.parent


def parse_axis_spec(spec: str) -> list[float]:
    """
    扫描轴解析（两种都支持）：
      * ``min,max,n``（第 3 个为**正整数** n）→ ``linspace(min, max, n)``
      * 其它 → 显式数值列表

    判别依据：x²/x⁴ 系数通常是小非整数，不会等于正整数；故"第 3 个值是否为 ≥1 的整数"
    可安全区分 ``linspace`` 与 3 元素显式列表。例如 ``0.005,0.025,5`` → linspace 5 点，
    ``0.01,0.02,0.03`` → 显式 3 值。若确实需要以正整数结尾的 3 元素显式列表，用 ≥4 个值
    或改写避免歧义。
    """
    vals = [float(x) for x in spec.split(",")]
    if len(vals) < 1:
        raise ValueError("扫描轴至少需 1 个值")
    if len(vals) == 3:
        lo, hi, third = vals
        # 第 3 个值为 ≥1 的整数 → 解释为 linspace 点数
        if third >= 1 and float(int(third)) == third:
            return list(np.linspace(lo, hi, int(third)))
    return vals


def sigma_y_um(r_norm: np.ndarray, dl: float) -> float:
    """
    σ_y = y 方向离子位置总体标准差 (µm)，与 ``Plotter/dataplot.py:513`` 同定义
    （``np.std`` 自动减 y 质心，ddof=0）。
    """
    r = np.asarray(r_norm)
    return float(np.std(r[:, 1] * dl * 1e6))


def _build_parsed():
    """构造一次 ParsedRun（cfg / params / field_settings）；force 由调用方每点重建。"""
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless 批跑，避免 matplotlib 找显示
    import setup_path

    setup_path.ensure_build_in_path(ROOT)
    from Interface.cli import create_parser, parse_and_build

    argv = [
        "--N", str(N_IONS),
        "--device", DEVICE,
        "--calc-method", CALC_METHOD,
        "--config", CONFIG_JSON,
        "--poly-potential", SOFTMODE_JSON,   # 占位使 field_settings 合法；force 自建
        "--time", str(TIME_MAX_US),
        "--step", str(STEP),
        "--interval", str(INTERVAL),
        "--batch", str(BATCH),
    ]
    args = create_parser().parse_args(argv)
    return parse_and_build(args, ROOT)


def run_until_sigma_or_time(parsed, force: Callable, cfg, log_every_us: float = 5.0):
    """
    跑到 σ_y < SIGMA_Y_THRESH_UM（提前 STOP，记录跨越帧）或 backend 自停于 TIME_MAX_US。

    Returns
    -------
    (r_um_final, reached, sim_time_us, sigma_y_final_um)
        r_um_final : (N, 3) µm，模拟结束位置
        reached    : 是否在 50µs 内达到 σ_y < 阈值
        sim_time_us: 达标→跨越时刻 (µs)；未达标→TIME_MAX_US
        sigma_y_final_um : 模拟结束时 σ_y (µm)
    """
    # 延迟导入：main 顶层会拉起 matplotlib / DataPlotter，仅 scan 路径需要
    from main import _consume_queue_until_done, _create_backend_and_start, _get_from_queue
    from utils import CommandType, Message

    proc, frame_init, q_ctrl, q_data = _create_backend_and_start(parsed, force)
    dl = cfg.dl
    dl_um = dl * 1e6
    dt_si = cfg.dt

    final = frame_init
    reached = False
    cross = None  # (r_um, t_us, sigma_y) 跨越帧
    last_log_us = -log_every_us
    try:
        while True:
            item = _get_from_queue(q_data, proc)
            if item is None:
                continue
            if item is False:
                break  # backend 自停于 50µs
            final = item
            sigma_y = sigma_y_um(final.r, dl)
            t_us = final.timestamp * dt_si * 1e6
            if t_us - last_log_us >= log_every_us:
                print(f"        t={t_us:7.3f}µs  σ_y={sigma_y:.6g}µm", flush=True)
                last_log_us = t_us
            if sigma_y < SIGMA_Y_THRESH_UM:
                reached = True
                cross = (final.r.copy() * dl_um, t_us, sigma_y)
                q_ctrl.put(Message(CommandType.STOP))
                _consume_queue_until_done(q_data, q_ctrl, proc)  # 排空尾帧
                break
    finally:
        proc.join(timeout=120)
        if proc.is_alive():
            proc.terminate()
            proc.join()

    if reached:
        r_um, t_us, sigma_y = cross
        return r_um, True, t_us, sigma_y
    return final.r * dl_um, False, TIME_MAX_US, sigma_y_um(final.r, dl)


def run_scan(
    x2_list: list[float],
    x4_list: list[float],
    out_dir: str | Path = OUT_DIR,
    *,
    force: bool = False,
    new_log: bool = False,
    base_seed: int = BASE_SEED,
    fixed_seed: bool = FIXED_SEED,
    init_range_um: tuple[float, float, float] = INIT_RANGE_UM,
) -> None:
    """
    遍历 (x², x⁴) 网格，每组跑一次动力学并存 .npy + 追加 log。

    输出布局::

        <out_dir>/scan.log              # log（所有运行序号汇总）
        <out_dir>/positions/{idx:04d}.npy  # 各轮结束位置 (N,3) µm

    log 与 npy 的写入策略：
      * **默认追加**：scan.log 已有内容则仅追加行（``write_header`` 幂等）。配合
        ``skip-if-npy-exists`` 可**续跑同一网格**——已完成的序号跳过且不重复记 log。
      * ``--force``：覆盖已存在的 .npy（重跑指定点），log 仍追加。
      * ``--new-log``：截断 scan.log 重开，并强制重跑所有点（覆盖旧 .npy），
        用于扫描**不同参数集**时另起一轮（同序号不与旧数据冲突）。
    """
    mp.set_start_method("fork", force=True)  # backend 依赖 fork；与 main.main() 一致
    parsed = _build_parsed()
    cfg = parsed.config
    p = parsed.params

    from equilibrium.potential_fit_3d import (
        fit_result_from_coeff_map,
        load_poly_potential_json,
        quartic_fit_coeff_map,
    )
    from FieldParser.force import build_poly_potential_force

    from .logio import append_row, write_header

    # softmode 基底系数（parse_and_build 已把 poly_potential 解析为绝对路径）
    softmode_path = parsed.field_settings.poly_potential
    base_fit = load_poly_potential_json(softmode_path)
    base_map = quartic_fit_coeff_map(base_fit)          # {label: V}，含固定项
    center_um, scale_um = base_fit.center_um, base_fit.scale_um
    offset = base_fit.potential_offset_V
    print(
        f"基底 softmode: {softmode_path}\n  scale_um={scale_um} center_um={center_um}\n"
        f"  固定项: { {k: v for k, v in base_map.items() if k not in ('x^2', 'x^4')} }"
    )

    charge = np.asarray(p.q, dtype=float)
    gamma = GAMMA if GAMMA is not None else parsed.field_settings.get_gamma()
    box = np.asarray(init_range_um, dtype=float)
    if np.any(box > scale_um):
        print(
            f"  ⚠ 警告：INIT_RANGE_UM {tuple(box)} 部分轴 > scale_um={scale_um}，"
            f"初态可能落在多项式有效区外（外推发散）。"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pos_dir = out / "positions"
    pos_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "scan.log"
    if new_log and log_path.exists():
        log_path.unlink()  # 截断重开
        print(f"已截断旧 log，重开：{log_path}")
    write_header(log_path)  # 幂等：追加模式下已有内容则不重写表头

    overwrite = force or new_log  # new_log 强制重跑覆盖旧 npy

    grid = list(product(x2_list, x4_list))
    total = len(grid)
    print(f"扫描网格：{len(x2_list)} × {len(x4_list)} = {total} 组；N={p.N} "
          f"{DEVICE}/{CALC_METHOD} γ={gamma} 阈值 σ_y<{SIGMA_Y_THRESH_UM}µm 上限 {TIME_MAX_US}µs")
    print(f"输出：log={log_path}  npy={pos_dir}/{'{idx:04d}.npy'}  "
          f"({'覆盖/重跑' if overwrite else '追加/续跑(跳过已存在)'} )")

    for idx, (x2, x4) in enumerate(grid, start=1):
        npy_path = pos_dir / f"{idx:04d}.npy"
        if (not overwrite) and npy_path.exists():
            print(f"[{idx}/{total}] skip (exists): {npy_path.name}", flush=True)
            continue
        seed = base_seed if fixed_seed else base_seed + idx
        print(f"[{idx}/{total}] x²={x2:g} x⁴={x4:g} seed={seed}", flush=True)
        wall_t0 = time.time()
        try:
            # 1) 种子化初态（均匀随机盒）→ 无量纲注入 params
            rng = np.random.default_rng(seed)
            r0_um = (rng.random((p.N, 3)) - 0.5) * box
            p.r0 = (r0_um * 1e-6 / cfg.dl).astype(float, order="C")
            p.v0 = np.zeros((p.N, 3), dtype=float)
            # 2) 改 x²/x⁴ 重建力（fork 前设模块级全局，子进程继承）
            coeff_map = dict(base_map)
            coeff_map["x^2"] = float(x2)
            coeff_map["x^4"] = float(x4)
            fit = fit_result_from_coeff_map(coeff_map, center_um, scale_um, offset)
            force = build_poly_potential_force(fit, cfg, charge, gamma)
            # 3) 跑到 σ_y 达标或 50µs
            r_um, reached, t_us, sigma_y = run_until_sigma_or_time(parsed, force, cfg)
            # 4) 存 npy + 追加 log（含 seed）
            np.save(npy_path, r_um)
            append_row(
                log_path,
                run_idx=idx,
                x2_coeff=x2,
                x4_coeff=x4,
                seed=seed,
                reached_threshold=reached,
                sim_time_us=t_us,
                sigma_y_final_um=sigma_y,
            )
            print(
                f"        → reached={int(reached)} t={t_us:.3f}µs σ_y={sigma_y:.6g}µm "
                f"({time.time() - wall_t0:.1f}s)",
                flush=True,
            )
        except Exception:
            print(f"        !! 失败，记 NaN 跳过：\n{traceback.format_exc()}", flush=True)
            append_row(
                log_path,
                run_idx=idx,
                x2_coeff=x2,
                x4_coeff=x4,
                seed=seed,
                reached_threshold=False,
                sim_time_us=0.0,
                sigma_y_final_um=float("nan"),
            )

    print(f"完成：{total} 组 → {out}（log: {log_path}）")
