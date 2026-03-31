"""
用法（在仓库根目录）::

    python -m bilayer --layer-spacing-um 5.0 --out bilayer/out/hessian.npz
    python -m bilayer --layer-spacing-um 5.0 --plot --plot-stride 3

可选：--r1 --r2 --charge --softening-um --plot --plot-out --plot-show --plot-stride
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bilayer.lattice import (
    bilayer_hessian_block_stats,
    coulomb_hessian_bilayer,
    load_positions_from_npz,
)
from bilayer.visualize import plot_bilayer_hessian


def _default_r0() -> Path:
    return Path(__file__).resolve().parent / "r0_single_layer"


def main() -> None:
    here = _default_r0()
    p = argparse.ArgumentParser(description="双层离子晶格库仑 Hessian（eV/μm²）")
    p.add_argument("--r1", type=Path, default=here / "r1.npz", help="第一层 npz")
    p.add_argument("--r2", type=Path, default=here / "r2.npz", help="第二层 npz")
    p.add_argument(
        "--layer-spacing-um",
        type=float,
        required=True,
        help="第二层沿 +y 相对其文件内坐标的平移距离（μm）",
    )
    p.add_argument("--charge", type=float, default=1.0, help="每离子电荷（e），默认 +1")
    p.add_argument(
        "--softening-um",
        type=float,
        default=0.0,
        help="库仑软化距离（μm），与 equilibrium 一致",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="保存 r, hessian, layer_spacing_um 等的 npz；默认 bilayer/out/hessian_dy{spacing}.npz",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="保存 Hessian 热力图（默认路径紧挨 npz：同 stem + _hessian.png）",
    )
    p.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="热力图输出路径；若不指定且启用 --plot，则根据 --out 推断",
    )
    p.add_argument(
        "--plot-show",
        action="store_true",
        help="交互显示 matplotlib 窗口（需在图形界面环境下）",
    )
    p.add_argument(
        "--plot-stride",
        type=int,
        default=1,
        help="热力图子采样步长（>1 时先 H[::s,::s] 再绘制，适合 1800 阶全矩阵预览）",
    )
    args = p.parse_args()

    r1 = load_positions_from_npz(args.r1)
    r2 = load_positions_from_npz(args.r2)
    n = r1.shape[0]
    charge = np.full(2 * n, args.charge, dtype=float)

    r_bi, h = coulomb_hessian_bilayer(
        r1,
        r2,
        args.layer_spacing_um,
        charge_ec=charge,
        softening_um=args.softening_um,
    )

    st = bilayer_hessian_block_stats(h, n)
    print(
        "Hessian 分块 |元素| 统计（eV/μm²）：\n"
        f"  同层 对角元（{st.n_intra_diag} 个）："
        f" 平均={st.intra_diag_abs_mean:.6g}，最大={st.intra_diag_abs_max:.6g}\n"
        f"  同层 非对角元（{st.n_intra_offdiag} 个）："
        f" 平均={st.intra_offdiag_abs_mean:.6g}，最大={st.intra_offdiag_abs_max:.6g}\n"
        f"  层间 矩形 H[:3N,3N:]（{st.n_inter} 个）："
        f" 平均={st.inter_abs_mean:.6g}，最大={st.inter_abs_max:.6g}"
    )

    out = args.out
    if out is None:
        out_dir = Path(__file__).resolve().parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"hessian_dy{args.layer_spacing_um:g}.npz"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out,
        r_um=r_bi,
        hessian_coulomb_eV_per_um2=h,
        layer_spacing_um=np.float64(args.layer_spacing_um),
        charge_ec=charge,
        softening_um=np.float64(args.softening_um),
        source_r1=str(args.r1),
        source_r2=str(args.r2),
        stats_intra_diag_abs_mean=np.float64(st.intra_diag_abs_mean),
        stats_intra_diag_abs_max=np.float64(st.intra_diag_abs_max),
        stats_intra_offdiag_abs_mean=np.float64(st.intra_offdiag_abs_mean),
        stats_intra_offdiag_abs_max=np.float64(st.intra_offdiag_abs_max),
        stats_inter_abs_mean=np.float64(st.inter_abs_mean),
        stats_inter_abs_max=np.float64(st.inter_abs_max),
        stats_n_intra_diag=np.int64(st.n_intra_diag),
        stats_n_intra_offdiag=np.int64(st.n_intra_offdiag),
        stats_n_inter=np.int64(st.n_inter),
    )
    print(f"写入 {out}，r shape {r_bi.shape}，H shape {h.shape}")

    if args.plot or args.plot_show or args.plot_out is not None:
        plot_path: Path | None = args.plot_out
        if plot_path is None and args.plot:
            plot_path = out.with_name(out.stem + "_hessian.png")
        save_fig = bool(args.plot or args.plot_out is not None)
        plot_bilayer_hessian(
            h,
            n_ions_per_layer=n,
            layer_spacing_um=args.layer_spacing_um,
            out_path=plot_path if save_fig and plot_path is not None else None,
            show=args.plot_show,
            stride=max(1, args.plot_stride),
        )


if __name__ == "__main__":
    main()
