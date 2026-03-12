"""
离子晶格平衡构型求解入口

流程：
1) 从电场数据拟合 3D 四次外势 V(x,y,z)
2) 构建总势能 U_total = U_trap + U_coulomb
3) 通过最小化 U_total 求平衡位置
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_range_2(s: str, arg_name: str, parser: argparse.ArgumentParser) -> tuple[float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        parser.error(f"--{arg_name} 须为两个数: min,max")
    return (parts[0], parts[1])


def _parse_center(s: str, parser: argparse.ArgumentParser) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        parser.error("--center 须为 x,y,z 三个数")
    return (parts[0], parts[1], parts[2])


def _build_initial_positions(
    n_ions: int,
    x_range_um: tuple[float, float],
    y_range_um: tuple[float, float],
    z_range_um: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """默认初值：在给定 x/y/z 范围内均匀随机排布"""
    r = np.zeros((n_ions, 3), dtype=float)
    r[:, 0] = rng.uniform(x_range_um[0], x_range_um[1], size=n_ions)
    r[:, 1] = rng.uniform(y_range_um[0], y_range_um[1], size=n_ions)
    r[:, 2] = rng.uniform(z_range_um[0], z_range_um[1], size=n_ions)
    return r


def _plot_equilibrium_layout(
    r_um: np.ndarray,
    x_range_um: tuple[float, float],
    y_range_um: tuple[float, float],
    z_range_um: tuple[float, float],
    color_mode: str | None = None,
    charge_ec: np.ndarray | None = None,
    plot_out: str | None = None,
) -> None:
    """
    绘制平衡位置离子的空间分布：
    - 上子图：zoy（x 轴 z，y 轴 y）
    - 下子图：zox（x 轴 z，y 轴 x）
    """
    import matplotlib.pyplot as plt

    z = r_um[:, 2]
    y = r_um[:, 1]
    x = r_um[:, 0]

    mode = (color_mode or "none").lower()
    scatter_kwargs: dict = {"s": 22, "alpha": 0.9}
    # 尽量对齐 Plotter 约定：none / y_pos / v2 / isotope
    if mode in ("none", ""):
        scatter_kwargs["c"] = "tab:red"
    elif mode == "y_pos":
        scatter_kwargs["c"] = y
        scatter_kwargs["cmap"] = "RdBu"
    elif mode == "v2":
        # 平衡位置图仅有坐标无速度，v2 模式不可用，降级为单色
        print("[plot] color_mode=v2 需速度信息，当前降级为 none")
        scatter_kwargs["c"] = "tab:red"
    elif mode == "isotope":
        # 当前流程无同位素质量数组，若提供 charge 则按电荷离散上色
        if charge_ec is not None and len(np.unique(charge_ec)) > 1:
            uq = np.unique(charge_ec)
            idx = np.searchsorted(uq, charge_ec)
            scatter_kwargs["c"] = idx
            scatter_kwargs["cmap"] = "tab10"
        else:
            print("[plot] color_mode=isotope 需同位素/多电荷信息，当前降级为 none")
            scatter_kwargs["c"] = "tab:red"
    else:
        print(f"[plot] 未识别 color_mode={color_mode}，当前降级为 none")
        scatter_kwargs["c"] = "tab:red"

    # 参考 Plotter 的布局风格：两行子图使用更高画布，避免元素拥挤
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # 上图：zoy
    sc_top = ax_top.scatter(z, y, **scatter_kwargs)
    ax_top.set_xlim(z_range_um[0], z_range_um[1])
    ax_top.set_ylim(y_range_um[0], y_range_um[1])
    ax_top.set_xlabel("z (μm)")
    ax_top.set_ylabel("y (μm)")
    ax_top.set_title("Equilibrium Layout: zoy")
    ax_top.set_aspect("equal")
    ax_top.grid(True, alpha=0.3)

    # 下图：zox
    ax_bottom.scatter(z, x, **scatter_kwargs)
    ax_bottom.set_xlim(z_range_um[0], z_range_um[1])
    ax_bottom.set_ylim(x_range_um[0], x_range_um[1])
    ax_bottom.set_xlabel("z (μm)")
    ax_bottom.set_ylabel("x (μm)")
    ax_bottom.set_title("Equilibrium Layout: zox")
    ax_bottom.set_aspect("equal")
    ax_bottom.grid(True, alpha=0.3)

    # 不使用 tight_layout，改为手动控制间距（与 Plotter 思路一致）
    fig.subplots_adjust(hspace=0.28, bottom=0.08, top=0.95)

    if plot_out:
        out_path = Path(plot_out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200)
        plt.close(fig)
        print(f"平衡位置图已保存: {out_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="最小化总势能（外势+库伦）以求离子晶格平衡位置"
    )
    parser.add_argument("--csv", type=str, default="", help="电场 CSV 路径")
    parser.add_argument("--config", type=str, default="", help="电压配置 JSON 路径")
    parser.add_argument("--N", type=int, default=20, help="离子数，默认 20")
    parser.add_argument("--charge", type=float, default=1.0, help="每个离子的电荷（单位 e），默认 +1")
    parser.add_argument(
        "--center",
        type=str,
        default="0,0,0",
        help="参考中心 x,y,z (μm)",
    )
    parser.add_argument("--x_range", type=str, default="-50,50", help="x 轴拟合/优化范围 (μm)")
    parser.add_argument("--y_range", type=str, default="-20,20", help="y 轴拟合/优化范围 (μm)")
    parser.add_argument("--z_range", type=str, default="-100,100", help="z 轴拟合/优化范围 (μm)")
    parser.add_argument("--fit-n-pts", type=int, default=8, help="3D 拟合每轴采样点数，默认 8")
    parser.add_argument("--softening-um", type=float, default=0.001, help="库伦软化长度 (μm)，默认 0.001")
    parser.add_argument("--maxiter", type=int, default=500, help="优化最大迭代步数，默认 500")
    parser.add_argument("--tol", type=float, default=1e-10, help="优化收敛阈值，默认 1e-10")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--init_file", type=str, default="", help="初值 npz 路径（含 r，shape=(N,3)，单位 μm）")
    parser.add_argument("--plot", action="store_true", help="绘制平衡位置空间分布（上 zoy，下 zox）")
    parser.add_argument(
        "--color_mode",
        type=str,
        default="none",
        choices=["none", "y_pos", "v2", "isotope"],
        help="绘图颜色模式：none / y_pos / v2 / isotope，默认 none",
    )
    parser.add_argument("--plot-out", type=str, default="", help="--plot 时输出图片路径；不传则弹窗显示")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 npz 路径（保存平衡位置与能量）；不传则保存到 equilibrium/equi_pos/{N}.npz",
    )
    parser.add_argument("--smooth-axes", type=str, default="z", help="势场平滑方向，默认 z；none 关闭")
    parser.add_argument("--smooth-sg", type=str, default="11,3", help="Savitzky-Golay 参数 window,poly，默认 11,3")
    args = parser.parse_args()

    from Interface.cli import (
        DEFAULT_CONFIG_DIR,
        DEFAULT_CONFIG_PATH,
        DEFAULT_CSV_DIR,
        DEFAULT_CSV_PATH,
    )
    from FieldConfiguration.constants import init_from_config
    from FieldConfiguration.loader import build_voltage_list, field_settings_from_config
    from FieldParser.calc_field import calc_field, calc_potential
    from FieldParser.csv_reader import read as read_csv
    from field_visualize.core import apply_savgol_smooth, compute_potentials, um_to_norm

    from equilibrium.energy import total_energy_and_grad
    from equilibrium.potential_fit_3d import fit_potential_3d_quartic

    def _resolve_path(arg: str, default_full: str, default_dir: str) -> str:
        if not arg:
            return str(_ROOT / default_full)
        p = Path(arg)
        if not p.is_absolute() and "/" not in arg and "\\" not in arg:
            return str(_ROOT / default_dir / arg)
        return str(_ROOT / arg) if not p.is_absolute() else arg

    config_path = _resolve_path(args.config, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_DIR)
    csv_path = _resolve_path(args.csv, DEFAULT_CSV_PATH, DEFAULT_CSV_DIR)

    cfg, config = init_from_config(config_path)
    grid_coord, grid_voltage = read_csv(csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV)
    n_voltage = grid_voltage.shape[1]
    if config:
        field_settings = field_settings_from_config(csv_path, config_path, n_voltage, cfg)
    else:
        from FieldConfiguration.field_settings import FieldSettings

        field_settings = FieldSettings(csv_filename=csv_path, voltage_list=[])
        field_settings.voltage_list = build_voltage_list({"voltage_list": []}, n_voltage, cfg)

    if args.smooth_axes.strip().lower() != "none":
        axes = tuple(a.strip().lower() for a in args.smooth_axes.split(",") if a.strip() in "xyz")
        if axes:
            parts = [p.strip() for p in args.smooth_sg.split(",")]
            wl = int(parts[0]) if parts else 11
            poly = int(parts[1]) if len(parts) >= 2 else 3
            grid_voltage = apply_savgol_smooth(grid_coord, grid_voltage, axes, window_length=wl, polyorder=poly)

    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)
    voltage_list = field_settings.voltage_list

    def compute_V_total(r_norm: np.ndarray) -> np.ndarray:
        _, _, _, v_total = compute_potentials(potential_interps, field_interps, voltage_list, cfg, r_norm)
        return v_total

    center_um = _parse_center(args.center, parser)
    x_range = _parse_range_2(args.x_range, "x_range", parser)
    y_range = _parse_range_2(args.y_range, "y_range", parser)
    z_range = _parse_range_2(args.z_range, "z_range", parser)
    range_um = (x_range, y_range, z_range)

    fit = fit_potential_3d_quartic(
        compute_V_total=compute_V_total,
        um_to_norm=lambda v: um_to_norm(v, cfg.dl),
        center_um=center_um,
        range_um=range_um,
        n_pts_per_axis=args.fit_n_pts,
    )

    n_ions = int(args.N)
    if n_ions <= 0:
        parser.error("--N 必须为正整数")
    charge_ec = np.full(n_ions, float(args.charge), dtype=float)

    if args.init_file:
        init_path = Path(args.init_file)
        if not init_path.is_absolute():
            init_path = _ROOT / init_path
        arr = np.load(str(init_path))
        if "r" not in arr:
            parser.error("--init_file 文件需包含键 'r'，shape=(N,3)，单位 μm")
        r0 = np.asarray(arr["r"], dtype=float)
        if r0.shape != (n_ions, 3):
            parser.error(f"init r 形状应为 ({n_ions},3)，当前 {r0.shape}")
    else:
        rng = np.random.default_rng(args.seed)
        r0 = _build_initial_positions(
            n_ions=n_ions,
            x_range_um=x_range,
            y_range_um=y_range,
            z_range_um=z_range,
            rng=rng,
        )

    bounds = [x_range, y_range, z_range] * n_ions

    def _reshape(x_flat: np.ndarray) -> np.ndarray:
        return x_flat.reshape(n_ions, 3)

    def objective(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
        r = _reshape(x_flat)
        breakdown, grad = total_energy_and_grad(
            fit=fit,
            r_um=r,
            charge_ec=charge_ec,
            softening_um=float(args.softening_um),
        )
        return breakdown.total_eV, grad.ravel()

    e0, g0 = objective(r0.ravel())
    print("=" * 68)
    print("离子晶格平衡构型求解")
    print("=" * 68)
    print(f"N = {n_ions}, q = {args.charge:+.3f} e")
    print(f"拟合 R² = {fit.r_squared:.6f}, scale L = {fit.scale_um:.1f} μm")
    print(f"势能零点平移: V_shifted = V_true - ({fit.potential_offset_V:.6e} V)")
    print(f"初始总能量: {e0:.6e} eV")

    res = minimize(
        fun=lambda x: objective(x)[0],
        x0=r0.ravel(),
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(args.maxiter), "ftol": float(args.tol)},
    )

    r_eq = _reshape(res.x)
    e_eq, g_eq = objective(res.x)
    breakdown_eq, _ = total_energy_and_grad(
        fit=fit,
        r_um=r_eq,
        charge_ec=charge_ec,
        softening_um=float(args.softening_um),
    )
    grad_norm = float(np.linalg.norm(g_eq))
    y_min = float(np.min(r_eq[:, 1]))
    y_max = float(np.max(r_eq[:, 1]))
    y_span = y_max - y_min

    print(f"优化状态: success={res.success}, status={res.status}")
    print(f"message: {res.message}")
    print(f"迭代: nit={res.nit}, nfev={res.nfev}, njev={getattr(res, 'njev', -1)}")
    print(f"终态总能量: {e_eq:.6e} eV")
    print(f"终态梯度范数: {grad_norm:.6e} eV/μm")
    print(f"  其中 U_trap={breakdown_eq.trap_eV:.6e} eV")
    print(f"      U_coul={breakdown_eq.coulomb_eV:.6e} eV")
    print(f"y 范围: [{y_min:.3f}, {y_max:.3f}] μm (span={y_span:.3f} μm)")
    print("=" * 68)

    if args.out.strip():
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
    else:
        out_path = _ROOT / "equilibrium" / "equi_pos" / f"{n_ions}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        r=r_eq,
        r0=r0,
        total_energy_eV=e_eq,
        trap_energy_eV=float(breakdown_eq.trap_eV),
        coulomb_energy_eV=float(breakdown_eq.coulomb_eV),
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        nit=int(res.nit),
        nfev=int(res.nfev),
        fit_r2=float(fit.r_squared),
        center_um=np.array(center_um, dtype=float),
        x_range_um=np.array(x_range, dtype=float),
        y_range_um=np.array(y_range, dtype=float),
        z_range_um=np.array(z_range, dtype=float),
    )
    print(f"结果已保存: {out_path}")

    if args.plot:
        plot_out = args.plot_out.strip() if args.plot_out else ""
        _plot_equilibrium_layout(
            r_um=r_eq,
            x_range_um=x_range,
            y_range_um=y_range,
            z_range_um=z_range,
            color_mode=args.color_mode,
            charge_ec=charge_ec,
            plot_out=plot_out or None,
        )


if __name__ == "__main__":
    main()

