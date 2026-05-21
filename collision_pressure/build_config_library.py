"""
构建构型库：对 N 离子运行多起点能量最小化

势场来源二选一:
  (A) --csv + --config     从电场 CSV + 电压配置 JSON 拟合
  (B) --trap-freq FX FY FZ  理想谐振势 (MHz)

用法:
    python -m collision_pressure.build_config_library --n-scans 20
    python -m collision_pressure --trap-freq 1.0 5.0 0.2 --n-ions 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="构建离子晶格构型库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-ions", type=int, default=54)
    parser.add_argument("--n-scans", type=int, default=200,
                        help="随机初猜次数 (默认 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument(
        "--fit-mode",
        type=str,
        default="quartic",
        choices=["none", "even", "quartic", "quartic_even", "quadratic"],
        help="3D 势拟合模式，仅在使用 CSV 数据时生效（默认 quartic）",
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认 collision_pressure/configs/N{N})")

    # 势场来源：互斥组
    field_group = parser.add_mutually_exclusive_group()
    field_group.add_argument(
        "--csv", type=str, default=None,
        help="电场 CSV 路径 (默认 data/monolithic20241118.csv)",
    )
    field_group.add_argument(
        "--trap-freq", nargs=3, type=float, metavar=("FX", "FY", "FZ"),
        help="理想谐振势阱频 FX FY FZ (MHz)，与 --csv 互斥",
    )

    parser.add_argument(
        "--config", type=str,
        default="FieldConfiguration/configs/collision.json",
        help="电压配置 JSON (仅配合 --csv 使用)",
    )
    parser.add_argument("--mass-amu", type=float, default=135.0)
    parser.add_argument("--softening-um", type=float, default=0.001)
    parser.add_argument("--edge-filter", type=float, default=1.5,
                        help="Delaunay 边长过滤因子 (0=不过滤)")
    parser.add_argument("--plane", type=str, default="xoz",
                        help="投影平面: xoz/yoz/xoy/auto")
    args = parser.parse_args()

    use_harmonic = args.trap_freq is not None
    csv_path = args.csv or "data/monolithic20241118.csv"

    output_dir = args.output_dir or str(
        _ROOT / "collision_pressure" / "configs" / f"N{args.n_ions}"
    )

    print("=== 构型库构建 ===")
    print(f"  离子数:     {args.n_ions}")
    print(f"  物种:       Ba{args.mass_amu:.0f}+")
    if use_harmonic:
        fx, fy, fz = args.trap_freq
        print(f"  势场模式:   理想谐振势")
        print(f"  阱频:       fx={fx} MHz, fy={fy} MHz, fz={fz} MHz")
    else:
        print(f"  势场模式:   CSV + Config 拟合")
        print(f"  电场数据:   {csv_path}")
        print(f"  电压配置:   {args.config}")
        print(f"  fit-mode:   {args.fit_mode}")
    print(f"  扫描次数:   {args.n_scans}")
    print(f"  输出目录:   {output_dir}")
    print()

    sys.path.insert(0, str(_ROOT))

    from collision_pressure.config_scan import scan_configurations

    print("[1/2] 构造势场...")
    t0 = time.time()

    if use_harmonic:
        from collision_pressure.config_scan import setup_fit_harmonic

        fx_Hz = args.trap_freq[0] * 1e6
        fy_Hz = args.trap_freq[1] * 1e6
        fz_Hz = args.trap_freq[2] * 1e6
        fit, v_min = setup_fit_harmonic(
            fx_Hz, fy_Hz, fz_Hz,
            mass_amu=args.mass_amu,
        )
        print(f"  完成 ({time.time() - t0:.1f}s), 理想谐振势")
        print(f"  V(x) = 0.5 * m*(wx^2*x^2 + wy^2*y^2 + wz^2*z^2) / q")

        # 根据阱频推算 L-BFGS-B 搜索盒 (单位 μm；l0 先以 m 计算再 ×1e6)
        omega = [2.0 * 3.141592653589793 * f for f in (fx_Hz, fy_Hz, fz_Hz)]
        AMU = 1.66053906660e-27
        EC = 1.602176634e-19
        m_kg = args.mass_amu * AMU
        eps0 = 8.854187817e-12
        l0_um = (
            EC**2 / (4 * 3.141592653589793 * eps0 * m_kg * omega[2]**2)
        ) ** (1 / 3) * 1e6
        L_approx_um = l0_um * args.n_ions**0.56 * 3
        x_range = (-L_approx_um * 2, L_approx_um * 2)
        y_range = (-L_approx_um * 2, L_approx_um * 2)
        z_range = (-L_approx_um * 4, L_approx_um * 4)
        print(
            f"  搜索范围 (μm): x,y∈[{x_range[0]:.1f}, {x_range[1]:.1f}], "
            f"z∈[{z_range[0]:.1f}, {z_range[1]:.1f}]"
        )
    else:
        from collision_pressure.config_scan import setup_fit

        fit, v_min = setup_fit(
            csv_path=csv_path,
            config_path=args.config,
            fit_mode=args.fit_mode,
        )
        n_terms = len(fit.basis_exps)
        fm = fit.fit_mode if fit.fit_mode else "none"
        print(
            f"  完成 ({time.time() - t0:.1f}s), R-squared={fit.r_squared:.6f}, "
            f"fit_mode={fm} ({n_terms} terms)"
        )
        x_range = (-50.0, 50.0)
        y_range = (-20.0, 20.0)
        z_range = (-150.0, 150.0)

    print(f"\n[2/2] 运行 {args.n_scans} 次随机初猜优化...")
    t0 = time.time()
    lib = scan_configurations(
        fit,
        n_ions=args.n_ions,
        n_scans=args.n_scans,
        seed=args.seed,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        softening_um=args.softening_um,
        mass_amu=args.mass_amu,
        maxiter=args.maxiter,
        plane=args.plane,
        edge_filter_factor=args.edge_filter,
    )
    print(f"\n  耗时 {time.time() - t0:.1f}s")

    lib.save(Path(output_dir))
    print(f"\n构型库已保存至 {output_dir}/")
    print(lib.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
