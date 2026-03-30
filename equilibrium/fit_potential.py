"""
三轴四次拟合入口脚本

加载场配置与势场，沿 x,y,z 分别做四次多项式拟合。
用法: python -m equilibrium.fit_potential [options]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="沿 x,y,z 三轴对总势场做四次多项式拟合"
    )
    parser.add_argument("--csv", type=str, default="", help="电场 CSV 路径")
    parser.add_argument("--config", type=str, default="", help="电压配置 JSON 路径")
    parser.add_argument(
        "--center",
        type=str,
        default="0,0,0",
        help="参考中心 x,y,z (μm)，逗号分隔",
    )
    parser.add_argument(
        "--x_range",
        type=str,
        default="-50,50",
        help="x 轴拟合范围 x_min,x_max (μm)，默认 -50,50",
    )
    parser.add_argument(
        "--y_range",
        type=str,
        default="-20,20",
        help="y 轴拟合范围 y_min,y_max (μm)，默认 -20,20",
    )
    parser.add_argument(
        "--z_range",
        type=str,
        default="-100,100",
        help="z 轴拟合范围 z_min,z_max (μm)，默认 -100,100",
    )
    parser.add_argument(
        "--n-pts",
        type=int,
        default=100,
        help="沿每轴采样点数，默认 100",
    )
    parser.add_argument(
        "--fit-mode",
        type=str,
        default="none",
        choices=["none", "even", "quartic", "quartic_even", "quadratic"],
        help=(
            "none=125 项张量；even=27 项全偶；quartic=35 项总次数≤4；"
            "quartic_even=10 项；quadratic=4 项（均为缩放坐标 u,v,w）"
        ),
    )
    parser.add_argument(
        "--plot-fit-report",
        action="store_true",
        help="生成拟合可视化报告（1D/2D/残差/梯度误差）并保存图片",
    )
    parser.add_argument(
        "--plot-fit-show",
        action="store_true",
        help="显示拟合报告图窗（可与 --plot-fit-report 同时使用）",
    )
    parser.add_argument(
        "--plot-fit-out-dir",
        type=str,
        default="",
        help="拟合报告输出目录；默认 equilibrium/results/potential_fit/{config_stem}",
    )
    parser.add_argument(
        "--plot-fit-line-pts",
        type=int,
        default=241,
        help="1D 线切片采样点数，默认 241",
    )
    parser.add_argument(
        "--plot-fit-slice-pts",
        type=int,
        default=181,
        help="2D 截面采样点数（每轴），默认 181",
    )
    parser.add_argument(
        "--smooth-axes",
        type=str,
        default="z",
        help="势场平滑方向，默认 z；用 none 关闭",
    )
    parser.add_argument(
        "--smooth-sg",
        type=str,
        default="11,3",
        help="Savitzky-Golay 参数 窗口,阶数，默认 11,3",
    )
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
    grid_coord, grid_voltage = read_csv(
        csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV
    )
    n_voltage = grid_voltage.shape[1]
    if config:
        field_settings = field_settings_from_config(
            csv_path, config_path, n_voltage, cfg
        )
    else:
        from FieldConfiguration.field_settings import FieldSettings

        field_settings = FieldSettings(csv_filename=csv_path, voltage_list=[])
        field_settings.voltage_list = build_voltage_list(
            {"voltage_list": []}, n_voltage, cfg
        )

    if args.smooth_axes.strip().lower() != "none":
        axes = tuple(a.strip().lower() for a in args.smooth_axes.split(",") if a.strip() in "xyz")
        if axes:
            parts = [p.strip() for p in args.smooth_sg.split(",")]
            wl = int(parts[0]) if parts else 11
            poly = int(parts[1]) if len(parts) >= 2 else 3
            grid_voltage = apply_savgol_smooth(
                grid_coord, grid_voltage, axes, window_length=wl, polyorder=poly
            )

    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)
    voltage_list = field_settings.voltage_list

    def compute_V_total(r):
        _, _, _, V_total = compute_potentials(
            potential_interps, field_interps, voltage_list, cfg, r
        )
        return V_total

    v_grid_all = np.asarray(compute_V_total(grid_coord), dtype=float).ravel()
    v_grid_valid = v_grid_all[np.isfinite(v_grid_all)]
    if v_grid_valid.size == 0:
        parser.error("格点总势场全为非有限值，无法确定统一势能零点")
    v_min_grid = float(np.min(v_grid_valid))

    def um_to_norm_fn(val_um: float) -> float:
        return um_to_norm(val_um, cfg.dl)

    center_parts = [float(x.strip()) for x in args.center.split(",")]
    if len(center_parts) != 3:
        parser.error("--center 须为 x,y,z 三个数")
    center_um: tuple[float, float, float] = (
        center_parts[0],
        center_parts[1],
        center_parts[2],
    )

    def parse_range_2(s: str, arg_name: str) -> tuple[float, float]:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 2:
            parser.error(f"--{arg_name} 须为两个数: min,max")
        return (parts[0], parts[1])

    range_um = (
        parse_range_2(args.x_range, "x_range"),
        parse_range_2(args.y_range, "y_range"),
        parse_range_2(args.z_range, "z_range"),
    )

    from equilibrium.potential_fit_3d import (
        eval_fit_3d,
        fit_potential_3d_quartic,
        grad_fit_3d,
        write_potential_fit_coeff_json,
    )

    fit = fit_potential_3d_quartic(
        compute_V_total=compute_V_total,
        um_to_norm=um_to_norm_fn,
        center_um=center_um,
        range_um=range_um,
        n_pts_per_axis=args.n_pts,
        potential_offset_V=v_min_grid,
        fit_mode=args.fit_mode,
    )
    write_potential_fit_coeff_json(
        fit,
        _ROOT / "equilibrium" / "results" / "potential_fit_coeff.json",
        csv=csv_path,
        config=config_path,
    )

    print("=" * 60)
    print("总势场 3D 四次多项式拟合")
    print("=" * 60)
    print(f"参考中心: ({center_um[0]:.1f}, {center_um[1]:.1f}, {center_um[2]:.1f}) μm")
    print(f"拟合范围: x∈[{range_um[0][0]}, {range_um[0][1]}], "
          f"y∈[{range_um[1][0]}, {range_um[1][1]}], "
          f"z∈[{range_um[2][0]}, {range_um[2][1]}] μm")
    print()
    print(f"拟合优度 R²: {fit.r_squared:.6f}")
    print()
    mode_disp = fit.fit_mode if fit.fit_mode else "none"
    sym_note = f"，fit_mode={mode_disp}（{len(fit.basis_exps)} 项）"
    print(
        "模型: V_shifted = Σ c_ijk u^i v^j w^k"
        f"{sym_note}, u=(x-x0)/L, v=(y-y0)/L, w=(z-z0)/L"
    )
    print(f"  缩放半跨度 L = {fit.scale_um:.1f} μm, 系数存 shape (5,5,5)，未参与拟合的幂次为 0")
    print(f"  势能零点平移: V_shifted = V_true - V_min_grid = V_true - ({fit.potential_offset_V:.6e} V)")
    print()
    # 验证：在中心点求值
    r_center = np.array([list(center_um)])
    V_fit_center = eval_fit_3d(fit, r_center)[0]
    grad_center = grad_fit_3d(fit, r_center)[0]
    print("中心点验证:")
    print(f"  V_fit(中心) = {V_fit_center:.6e} V")
    print(f"  grad V(中心) = ({grad_center[0]:.4e}, {grad_center[1]:.4e}, {grad_center[2]:.4e}) V/μm")
    print()
    print("系数数组 (部分，低阶项):")
    print("  c[0,0,0] (常数项):", fit.coeffs[0, 0, 0])
    print("  c[1,0,0], c[0,1,0], c[0,0,1] (一次项):", fit.coeffs[1, 0, 0], fit.coeffs[0, 1, 0], fit.coeffs[0, 0, 1])
    print("  c[2,0,0], c[0,2,0], c[0,0,2] (二次项):", fit.coeffs[2, 0, 0], fit.coeffs[0, 2, 0], fit.coeffs[0, 0, 2])
    print("=" * 60)

    def _to_norm_xyz(x_um: np.ndarray, y_um: np.ndarray, z_um: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                um_to_norm_fn(np.asarray(x_um, dtype=float)),
                um_to_norm_fn(np.asarray(y_um, dtype=float)),
                um_to_norm_fn(np.asarray(z_um, dtype=float)),
            ]
        )

    def _potential_true_from_um(r_um: np.ndarray) -> np.ndarray:
        r_um = np.asarray(r_um, dtype=float)
        r_norm = _to_norm_xyz(r_um[:, 0], r_um[:, 1], r_um[:, 2])
        return compute_V_total(r_norm)

    def _potential_fit_true_residual(r_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # eval_fit_3d 返回的是已做零点平移后的 V_shifted，因此真值也需先减去同一 offset
        v_true_raw = _potential_true_from_um(r_um)
        v_true_shifted = v_true_raw - fit.potential_offset_V
        v_fit = eval_fit_3d(fit, r_um)
        return v_fit, v_true_shifted, (v_fit - v_true_shifted)

    def _residual_stats(v_res: np.ndarray) -> dict[str, float]:
        abs_res = np.abs(v_res)
        rmse = float(np.sqrt(np.mean(v_res * v_res)))
        return {
            "rmse": rmse,
            "mae": float(np.mean(abs_res)),
            "p95": float(np.percentile(abs_res, 95)),
            "p99": float(np.percentile(abs_res, 99)),
            "max_abs": float(np.max(abs_res)),
        }

    def _run_fit_report() -> None:
        if args.plot_fit_line_pts < 5:
            parser.error("--plot-fit-line-pts 至少为 5")
        if args.plot_fit_slice_pts < 8:
            parser.error("--plot-fit-slice-pts 至少为 8")
        if args.n_pts < 3:
            parser.error("--n-pts 至少为 3 才能进行边界误差检查")

        import matplotlib.pyplot as plt

        (xr0, xr1), (yr0, yr1), (zr0, zr1) = range_um
        x0, y0, z0 = center_um

        config_stem = Path(config_path).stem if config_path else "default"
        if args.plot_fit_out_dir.strip():
            out_dir = Path(args.plot_fit_out_dir.strip())
            if not out_dir.is_absolute():
                out_dir = _ROOT / out_dir
        else:
            out_dir = _ROOT / "equilibrium" / "results" / "potential_fit" / config_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1D：三轴中心切线，对比 V_shifted(true) / V_fit 与残差
        line_pts = int(args.plot_fit_line_pts)
        x_line = np.linspace(xr0, xr1, line_pts)
        y_line = np.linspace(yr0, yr1, line_pts)
        z_line = np.linspace(zr0, zr1, line_pts)

        r_x = np.column_stack([x_line, np.full_like(x_line, y0), np.full_like(x_line, z0)])
        r_y = np.column_stack([np.full_like(y_line, x0), y_line, np.full_like(y_line, z0)])
        r_z = np.column_stack([np.full_like(z_line, x0), np.full_like(z_line, y0), z_line])

        vx_fit, vx_true, rx = _potential_fit_true_residual(r_x)
        vy_fit, vy_true, ry = _potential_fit_true_residual(r_y)
        vz_fit, vz_true, rz = _potential_fit_true_residual(r_z)

        fig1, axes1 = plt.subplots(3, 2, figsize=(12.5, 10.0), sharex="col")
        axis_data = (
            ("x", x_line, vx_true, vx_fit, rx),
            ("y", y_line, vy_true, vy_fit, ry),
            ("z", z_line, vz_true, vz_fit, rz),
        )
        for i, (label, coord, v_true, v_fit, v_res) in enumerate(axis_data):
            ax_l = axes1[i, 0]
            ax_r = axes1[i, 1]
            ax_l.plot(coord, v_true, label="V_true_shifted", lw=2.0, alpha=0.85)
            ax_l.plot(coord, v_fit, "--", label="V_fit", lw=1.8, alpha=0.9)
            ax_l.set_ylabel("V (V)")
            ax_l.set_title(f"{label}-axis center line")
            ax_l.grid(alpha=0.25)
            ax_l.legend(loc="best")

            ax_r.plot(coord, v_res, color="tab:red", lw=1.7)
            ax_r.axhline(0.0, color="k", lw=1.0, alpha=0.5)
            ax_r.set_ylabel("Delta V (V)")
            ax_r.set_title(f"{label}-axis residual")
            ax_r.grid(alpha=0.25)
        axes1[2, 0].set_xlabel("coordinate (um)")
        axes1[2, 1].set_xlabel("coordinate (um)")
        fig1.suptitle("Quartic fit 1D center-line comparison", fontsize=13)
        fig1.tight_layout(rect=[0, 0, 1, 0.97])
        out_1d = out_dir / "fit_1d_center_lines.png"
        fig1.savefig(out_1d, dpi=160)

        # 梯度误差（用 1D 方向差分近似真值偏导）
        gx_true = np.gradient(vx_true, x_line)
        gy_true = np.gradient(vy_true, y_line)
        gz_true = np.gradient(vz_true, z_line)
        gx_fit = grad_fit_3d(fit, r_x)[:, 0]
        gy_fit = grad_fit_3d(fit, r_y)[:, 1]
        gz_fit = grad_fit_3d(fit, r_z)[:, 2]
        gerr_x = gx_fit - gx_true
        gerr_y = gy_fit - gy_true
        gerr_z = gz_fit - gz_true

        fig_g, axes_g = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=False)
        grad_data = (
            ("x", x_line, gerr_x),
            ("y", y_line, gerr_y),
            ("z", z_line, gerr_z),
        )
        for i, (label, coord, gerr) in enumerate(grad_data):
            ax = axes_g[i]
            ax.plot(coord, gerr, lw=1.7, color="tab:purple")
            ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)
            ax.set_ylabel("Delta(dV/d%s) (V/um)" % label)
            ax.set_title(f"{label}-axis gradient error")
            ax.grid(alpha=0.25)
            ax.set_xlabel("coordinate (um)")
        fig_g.suptitle("Quartic fit gradient error on center lines", fontsize=13)
        fig_g.tight_layout(rect=[0, 0, 1, 0.97])
        out_grad = out_dir / "fit_1d_gradient_error.png"
        fig_g.savefig(out_grad, dpi=160)

        # 2D：x-z 和 y-z 截面（true / fit / residual）
        def _plot_slice(
            a_vals: np.ndarray,
            b_vals: np.ndarray,
            fixed_val: float,
            plane: str,
        ) -> Path:
            aa, bb = np.meshgrid(a_vals, b_vals, indexing="ij")
            if plane == "xz":
                r = np.column_stack([aa.ravel(), np.full(aa.size, fixed_val), bb.ravel()])
                x_label, y_label = "x (um)", "z (um)"
            elif plane == "yz":
                r = np.column_stack([np.full(aa.size, fixed_val), aa.ravel(), bb.ravel()])
                x_label, y_label = "y (um)", "z (um)"
            else:
                raise ValueError(f"unsupported plane: {plane}")

            v_fit, v_true, v_res = _potential_fit_true_residual(r)
            v_fit_2d = v_fit.reshape(aa.shape)
            v_true_2d = v_true.reshape(aa.shape)
            v_res_2d = v_res.reshape(aa.shape)

            v_min = float(min(np.min(v_true_2d), np.min(v_fit_2d)))
            v_max = float(max(np.max(v_true_2d), np.max(v_fit_2d)))
            vmax_abs_res = float(np.max(np.abs(v_res_2d)))

            fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4), sharex=True, sharey=True)
            extent = [float(a_vals[0]), float(a_vals[-1]), float(b_vals[0]), float(b_vals[-1])]
            im0 = axes[0].imshow(
                v_true_2d.T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                vmin=v_min,
                vmax=v_max,
            )
            axes[0].set_title("V_true_shifted (V)")
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel(y_label)

            im1 = axes[1].imshow(
                v_fit_2d.T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                vmin=v_min,
                vmax=v_max,
            )
            axes[1].set_title("V_fit (V)")
            axes[1].set_xlabel(x_label)

            im2 = axes[2].imshow(
                v_res_2d.T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="coolwarm",
                vmin=-vmax_abs_res,
                vmax=vmax_abs_res,
            )
            axes[2].set_title("Delta V = V_fit - V_true_shifted (V)")
            axes[2].set_xlabel(x_label)

            fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            fig.suptitle(f"Quartic fit 2D slice: {plane} plane", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            out_path = out_dir / f"fit_2d_slice_{plane}.png"
            fig.savefig(out_path, dpi=170)
            return out_path

        slice_pts = int(args.plot_fit_slice_pts)
        x_slice = np.linspace(xr0, xr1, slice_pts)
        y_slice = np.linspace(yr0, yr1, slice_pts)
        z_slice = np.linspace(zr0, zr1, slice_pts)
        out_xz = _plot_slice(x_slice, z_slice, y0, "xz")
        out_yz = _plot_slice(y_slice, z_slice, x0, "yz")

        # 3D 采样网格残差统计 + 边界误差检查
        nx = ny = nz = int(args.n_pts)
        xg = np.linspace(xr0, xr1, nx)
        yg = np.linspace(yr0, yr1, ny)
        zg = np.linspace(zr0, zr1, nz)
        xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
        r_grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        _, _, res_grid = _potential_fit_true_residual(r_grid)
        stats = _residual_stats(res_grid)
        res_grid_3d = res_grid.reshape(nx, ny, nz)
        on_boundary = np.zeros((nx, ny, nz), dtype=bool)
        on_boundary[0, :, :] = True
        on_boundary[-1, :, :] = True
        on_boundary[:, 0, :] = True
        on_boundary[:, -1, :] = True
        on_boundary[:, :, 0] = True
        on_boundary[:, :, -1] = True
        b_abs = np.abs(res_grid_3d[on_boundary])
        i_abs = np.abs(res_grid_3d[~on_boundary])
        b_max = float(np.max(b_abs))
        i_max = float(np.max(i_abs)) if i_abs.size else float("nan")
        b_p95 = float(np.percentile(b_abs, 95))
        i_p95 = float(np.percentile(i_abs, 95)) if i_abs.size else float("nan")

        print()
        print("[拟合可视化报告] 残差统计（采样网格）")
        print(f"  RMSE      = {stats['rmse']:.6e} V")
        print(f"  MAE       = {stats['mae']:.6e} V")
        print(f"  P95|ΔV|   = {stats['p95']:.6e} V")
        print(f"  P99|ΔV|   = {stats['p99']:.6e} V")
        print(f"  max|ΔV|   = {stats['max_abs']:.6e} V")
        print("  边界误差检查（boundary vs interior）:")
        print(f"    boundary: max|ΔV|={b_max:.6e} V, P95|ΔV|={b_p95:.6e} V")
        if i_abs.size:
            print(f"    interior: max|ΔV|={i_max:.6e} V, P95|ΔV|={i_p95:.6e} V")
        else:
            print("    interior: (无内部网格点，已跳过)")
        print("  图像输出:")
        print(f"    {out_1d}")
        print(f"    {out_grad}")
        print(f"    {out_xz}")
        print(f"    {out_yz}")
        print()

        if args.plot_fit_show:
            plt.show()
        else:
            plt.close("all")

    if args.plot_fit_report or args.plot_fit_show:
        _run_fit_report()


if __name__ == "__main__":
    main()
