"""radial_trap CLI：解析参数 → 晶格条件 → 平衡构型 → 解析 micromotion → 输出。

用法:
    python -m radial_trap [--A ... --B ... --D ... --E ... --F ...] [options]

示例:
    python -m radial_trap                                   # 默认演示参数
    python -m radial_trap --N 2 --phonon                    # N=2 + 面内声子
    python -m radial_trap --B 1e-7 --D 0.5 --F -5e-8        # DB+F=0 精确抵消演示
    python -m radial_trap --json params.json --report r.json

参数优先级: CLI 显式 > --json > 内置默认（默认值即有效演示参数：
q≈0.29，f_x≈4.31 MHz，f_y≈2.78 MHz，4 条件全过）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

from radial_trap.types import RadialTrapParams

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "radial_trap/results"

# --json 允许的键（严格键集，未知键报错）
_JSON_KEYS = (
    "A", "B", "D", "E", "F",
    "freq_rf_mhz", "species", "N",
    "x_range", "y_range", "seed", "softening_um",
)


def _parse_range(s: str) -> tuple[float, float]:
    """解析 'lo,hi' 范围（要求 lo < hi）。"""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"范围格式 'lo,hi'，收到 '{s}'")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"范围需为两个浮点数，收到 '{s}'") from exc
    if not lo < hi:
        raise argparse.ArgumentTypeError(f"范围需 lo < hi，收到 {lo}, {hi}")
    return (lo, hi)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radial_trap",
        description="2D 径向多项式势阱：平衡构型 + 解析 micromotion + 晶格条件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--A", type=float, default=None, metavar="V_UM2",
                        help="RF 四极系数 A (V/µm²)，默认 5e-3")
    parser.add_argument("--B", type=float, default=None, metavar="V_UM4",
                        help="RF 十六极系数 B (V/µm⁴)，默认 0")
    parser.add_argument("--D", type=float, default=None,
                        help="RF bias 无量纲系数 D，默认 0")
    parser.add_argument("--E", type=float, default=None, metavar="V_UM2",
                        help="DC 势 x²−y² 系数（势系数，非电场记号），默认 1.5e-4")
    parser.add_argument("--F", type=float, default=None, metavar="V_UM4",
                        help="DC 势 x⁴−6x²y²+y⁴ 系数（势系数），默认 0")
    parser.add_argument("--freq-rf-mhz", type=float, default=None,
                        help="RF 频率 (MHz)，默认 35.28")
    parser.add_argument("--species", type=str, default=None,
                        help="离子种类，默认 Ba135+")
    parser.add_argument("--N", type=int, default=None,
                        help="离子数，默认 10")
    parser.add_argument("--x-range", type=_parse_range, default=None, metavar="LO,HI",
                        help="x 初始化/囚禁范围 (µm)，默认 -10,10；负数需 = 语法: --x-range=-10,10")
    parser.add_argument("--y-range", type=_parse_range, default=None, metavar="LO,HI",
                        help="y 初始化/囚禁范围 (µm)，默认 -80,80；负数需 = 语法: --y-range=-80,80")
    parser.add_argument("--seed", type=int, default=None,
                        help="初始化随机种子，默认 0")
    parser.add_argument("--softening-um", type=float, default=None,
                        help="库伦软化长度 (µm)，默认 1e-3")
    parser.add_argument("--scale-um", type=float, default=None,
                        help="FitResult3D 归一化长度 (µm)，默认 100")
    parser.add_argument("--maxiter", type=int, default=5000,
                        help="L-BFGS-B 最大迭代数，默认 5000")
    parser.add_argument("--tol", type=float, default=1e-15,
                        help="L-BFGS-B ftol，默认 1e-15")
    parser.add_argument("--cond-tol", type=float, default=1e-12,
                        help="条件(1)(2) 的趋零判定容差，默认 1e-12")
    parser.add_argument("--phonon", action="store_true",
                        help="求解面内 (x,y) 声子模式")
    parser.add_argument("--phonon-print-modes", type=int, default=10, metavar="K",
                        help="控制台打印的声子模式数（按频率降序），默认 10")
    parser.add_argument("--json", type=str, default=None, metavar="PATH",
                        help="参数 JSON 存档读入（严格键集；被 CLI 显式旗标覆盖）")
    parser.add_argument("--out", type=str, default=None, metavar="PATH",
                        help=f"npz 输出路径，默认 {DEFAULT_OUT_DIR}/radial_N{{N}}.npz")
    parser.add_argument("--report", type=str, default=None, metavar="PATH",
                        help="JSON 报告输出路径（默认不写）")
    parser.add_argument("--plot", action="store_true",
                        help="输出晶格 + micromotion 双联图 PNG（无头 Agg 保存）")
    parser.add_argument("--plot-out", type=str, default=None, metavar="PATH",
                        help=f"图输出路径，默认 {DEFAULT_OUT_DIR}/radial_N{{N}}.png")
    parser.add_argument("--export-poly", type=str, default=None, metavar="PATH",
                        help="导出总势多项式系数 JSON（write_potential_fit_coeff_json "
                             "格式，可被 main.py --poly-potential / load_poly_potential_json "
                             "读回；含 x^6 项，需次数上限 6 支持）")
    return parser


def _load_json_params(path: str) -> dict:
    """读入参数 JSON；顶层必须为对象且键集严格合法。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象")
    unknown = sorted(set(data) - set(_JSON_KEYS))
    if unknown:
        raise ValueError(f"未知键 {unknown}，合法键: {list(_JSON_KEYS)}")
    return data


def _check_range(val, name: str) -> tuple[float, float]:
    try:
        lo, hi = (float(val[0]), float(val[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{name} 需为 [lo, hi] 两元素数组，收到 {val!r}") from exc
    if not lo < hi:
        raise ValueError(f"{name} 需 lo < hi，收到 {lo}, {hi}")
    return (lo, hi)


def _merge_params(args: argparse.Namespace, js: dict | None) -> RadialTrapParams:
    """合并参数：CLI 显式 > --json > 内置默认（默认值即演示参数）。"""
    src = dict(js or {})

    def pick(cli_val, key: str, default):
        if cli_val is not None:
            return cli_val
        return src.get(key, default)

    params = RadialTrapParams(
        A_V_per_um2=pick(args.A, "A", 5e-3),
        B_V_per_um4=pick(args.B, "B", 0.0),
        D_dimless=pick(args.D, "D", 0.0),
        E_dc_V_per_um2=pick(args.E, "E", 1.5e-4),
        F_dc_V_per_um4=pick(args.F, "F", 0.0),
        freq_rf_mhz=pick(args.freq_rf_mhz, "freq_rf_mhz", 35.28),
        species_name=str(pick(args.species, "species", "Ba135+")),
        n_ions=int(pick(args.N, "N", 10)),
        x_range_um=_check_range(pick(args.x_range, "x_range", (-10.0, 10.0)), "x_range"),
        y_range_um=_check_range(pick(args.y_range, "y_range", (-80.0, 80.0)), "y_range"),
        seed=int(pick(args.seed, "seed", 0)),
        softening_um=float(pick(args.softening_um, "softening_um", 1e-3)),
        scale_um=float(pick(args.scale_um, "scale_um", 100.0)),
    )
    if params.n_ions < 1:
        raise ValueError(f"N 需 >= 1，收到 {params.n_ions}")
    if params.softening_um < 0.0:
        raise ValueError(f"softening_um 需 >= 0，收到 {params.softening_um}")
    if params.scale_um <= 0.0:
        raise ValueError(f"scale_um 需 > 0，收到 {params.scale_um}")
    return params


def _format_pass(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _print_report(result, phonon_print_modes: int = 10) -> None:
    p, c, eq, mm = result.params, result.conditions, result.equilibrium, result.micromotion
    print("=== 径向势阱参数 ===")
    print(f"  φ_rf: A={p.A_V_per_um2:g} V/µm², B={p.B_V_per_um4:g} V/µm⁴   "
          f"RF bias: D={p.D_dimless:g}")
    print(f"  φ_DC: E={p.E_dc_V_per_um2:g} V/µm², F={p.F_dc_V_per_um4:g} V/µm⁴")
    print(f"  f_RF={p.freq_rf_mhz:g} MHz, 物种={p.species_name}, N={p.n_ions}, seed={p.seed}")
    print("=== 晶格条件（设计文档 4 条） ===")
    print(f"  (1) 压制交叉项  DB+F={c.db_plus_f:.3e}, |B|={c.abs_b:.3e}"
          f"  →  {_format_pass(c.pass_cross)}")
    print(f"  (2) 压制 y 高次  −16αAB+DB+F={c.y4_comb:.3e}"
          f"  →  {_format_pass(c.pass_y_high)}")
    print(f"  (3) y 束缚  c_y2={c.c_y2:.6e} V/µm²  →  {_format_pass(c.pass_y_confine)}"
          f"  (f_y={c.f_trap_y_mhz:.4g} MHz)")
    print(f"  (4) x 束缚  c_x2={c.c_x2:.6e} V/µm²  →  {_format_pass(c.pass_x_confine)}"
          f"  (f_x={c.f_trap_x_mhz:.4g} MHz)")
    print(f"  α_eff={c.alpha_eff_um2_per_V:.6g} µm²/V, "
          f"q_x={c.q_mathieu_x:.4f}, q_y={c.q_mathieu_y:.4f}")
    print(f"  四次项相对重要性 eps4_x={c.eps4_x:.3g}, eps4_y={c.eps4_y:.3g} (L_ref=10 µm)")
    if c.xy_degenerate:
        print("  ⚠ x/y 二次系数近简并：链取向由初始条件决定（xy_degenerate=True）")
    print("=== 平衡构型 ===")
    print(f"  converged={eq.converged}, nit={eq.n_iter}, "
          f"|grad|={eq.grad_norm_eV_per_um:.2e} eV/µm, E_total={eq.total_eV:.6e} eV")
    r = eq.r_eq_um
    print(f"  x ∈ [{r[:, 0].min():.3f}, {r[:, 0].max():.3f}] µm, "
          f"y ∈ [{r[:, 1].min():.3f}, {r[:, 1].max():.3f}] µm")
    if eq.hit_bounds:
        print("  ⚠ 有离子贴近 x/y 边界（疑似非囚禁或范围不足）")
    print("=== 解析 micromotion（一阶 RF 谐波峰值） ===")
    print(f"  |a_mm|: max={mm.mag_um.max():.4f} µm, mean={mm.mag_um.mean():.4f} µm")
    finite = np.isfinite(mm.excess_ratio)
    if finite.any():
        print(f"  excess 比 ρ = a_mm/((q/2)·r): "
              f"[{np.nanmin(mm.excess_ratio):.4f}, {np.nanmax(mm.excess_ratio):.4f}]"
              "  (四极极限=1)")
    if result.phonon is not None:
        f_mhz = result.phonon.freq_hz_signed / 1e6
        print(f"=== 面内声子（{f_mhz.size} 模，频率降序，负值=不稳定） ===")
        for i, f in enumerate(f_mhz[: max(phonon_print_modes, 0)]):
            print(f"  [{i:2d}] {f:10.4f} MHz")
        if f_mhz.size > max(phonon_print_modes, 0):
            print(f"  ... 其余 {f_mhz.size - phonon_print_modes} 模略"
                  "（--phonon-print-modes 调整）")


def _npz_payload(result) -> dict:
    """npz 输出键值（params_* / cond_* 全量 + 平衡/micromotion/声子）。"""
    p, c, eq, mm = result.params, result.conditions, result.equilibrium, result.micromotion
    payload: dict = {}
    for f in fields(p):
        payload[f"params_{f.name}"] = getattr(p, f.name)
    for f in fields(c):
        payload[f"cond_{f.name}"] = getattr(c, f.name)
    payload.update(
        {
            "r_eq_um": eq.r_eq_um,
            "r_init_um": eq.r_init_um,
            "energy_total_eV": eq.total_eV,
            "grad_norm_eV_per_um": eq.grad_norm_eV_per_um,
            "converged": eq.converged,
            "hit_bounds": eq.hit_bounds,
            "n_iter": eq.n_iter,
            "message": eq.message,
            "a_mm_um": mm.a_mm_um,
            "a_mm_mag_um": mm.mag_um,
            "a_mm_excess_ratio": mm.excess_ratio,
        }
    )
    if result.phonon is not None:
        payload["phonon_freqs_hz"] = result.phonon.freq_hz_signed
    return payload


def _json_safe(x):
    """递归转换为 JSON 安全类型（NaN/Inf → None）。"""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def _report_dict(result) -> dict:
    p, c, eq, mm = result.params, result.conditions, result.equilibrium, result.micromotion
    report = {
        "params": {f.name: _json_safe(getattr(p, f.name)) for f in fields(p)},
        "conditions": {f.name: _json_safe(getattr(c, f.name)) for f in fields(c)},
        "equilibrium": {
            "total_eV": eq.total_eV,
            "grad_norm_eV_per_um": eq.grad_norm_eV_per_um,
            "converged": eq.converged,
            "n_iter": eq.n_iter,
            "hit_bounds": eq.hit_bounds,
            "message": eq.message,
            "r_eq_um": eq.r_eq_um.tolist(),
        },
        "micromotion": {
            "a_mm_um": mm.a_mm_um.tolist(),
            "mag_um": mm.mag_um.tolist(),
            "excess_ratio": _json_safe(mm.excess_ratio),
        },
    }
    if result.phonon is not None:
        report["phonon"] = {
            "freq_hz_signed": result.phonon.freq_hz_signed.tolist(),
            "freq_mhz": (result.phonon.freq_hz_signed / 1e6).tolist(),
        }
    return _json_safe(report)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = create_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, os.environ.get("ISM_LOG_LEVEL", "INFO")),
        format="%(levelname)s: %(message)s",
    )

    js = None
    if args.json:
        try:
            js = _load_json_params(args.json)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"--json 加载失败: {exc}")

    from radial_trap.lattice import (
        find_radial_equilibrium,
        micromotion_amplitude,
        solve_inplane_phonons,
    )
    from radial_trap.potential import (
        build_total_potential_fit,
        evaluate_conditions,
        get_species,
    )
    from radial_trap.types import RadialTrapResult

    try:
        params = _merge_params(args, js)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        get_species(params.species_name)
    except KeyError as exc:
        parser.error(str(exc))

    try:
        fit = build_total_potential_fit(params)
    except ValueError as exc:
        parser.error(str(exc))

    conditions = evaluate_conditions(params, cond_tol=args.cond_tol)
    if not conditions.pass_cross:
        logger.warning("条件(1) 未满足：总势含 x²y² 交叉项，晶格将变形")
    if not conditions.pass_y_high:
        logger.warning("条件(2) 未满足：y 方向高次项显著，链可能变粗/成双阱")
    if not conditions.pass_y_confine:
        logger.warning("条件(3) 未满足：c_y2 ≤ 0，y 方向无二次囚禁（仍继续求解）")
    if not conditions.pass_x_confine:
        logger.warning("条件(4) 未满足：c_x2 ≤ 0，x 方向无二次囚禁（仍继续求解）")

    equilibrium = find_radial_equilibrium(fit, params, maxiter=args.maxiter, tol=args.tol)
    micromotion = micromotion_amplitude(equilibrium.r_eq_um, params)
    phonon = (
        solve_inplane_phonons(fit, equilibrium.r_eq_um, params) if args.phonon else None
    )
    result = RadialTrapResult(
        params=params,
        fit=fit,
        conditions=conditions,
        equilibrium=equilibrium,
        micromotion=micromotion,
        phonon=phonon,
    )

    _print_report(result, phonon_print_modes=args.phonon_print_modes)

    out_path = Path(args.out) if args.out else root / DEFAULT_OUT_DIR / f"radial_N{params.n_ions}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **_npz_payload(result))
    print(f"npz 输出: {out_path}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(_report_dict(result), f, ensure_ascii=False, indent=2)
        print(f"JSON 报告: {report_path}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        from radial_trap.plots import plot_lattice

        plot_path = (
            Path(args.plot_out)
            if args.plot_out
            else root / DEFAULT_OUT_DIR / f"radial_N{params.n_ions}.png"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_lattice(result, out_path=plot_path)

    if args.export_poly:
        from equilibrium.potential_fit_3d import write_potential_fit_coeff_json

        export_path = Path(args.export_poly)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        write_potential_fit_coeff_json(fit, export_path)
        print(f"多项式势导出: {export_path}（main.py --poly-potential 可读回）")

    return 0
