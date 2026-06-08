"""可视化能量最高/最低的 5 种晶格构型"""
import argparse
import sys
from pathlib import Path

import json
import numpy as np

from collision_pressure._mpl_backend import pyplot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ROOT = Path(__file__).resolve().parent.parent


def load_library(lib_dir: Path):
    with open(lib_dir / "summary.json", encoding="utf-8") as f:
        meta = json.load(f)
    configs = meta["configurations"]
    configs.sort(key=lambda c: c["energy_total_eV"])
    return configs


def load_config(lib_dir: Path, npz_file):
    data = np.load(lib_dir / npz_file)
    return {
        "r": data["r"],
        "coord_numbers": data["coord_numbers"],
        "adj_matrix": data["adj_matrix"],
        "boundary": set(data["boundary"].tolist()),
        "energy": float(data["energy_total_eV"]),
    }


def plot_config(ax, cfg, title):
    plt = pyplot()
    r = cfg["r"]
    cn = cfg["coord_numbers"]
    adj = cfg["adj_matrix"]
    bnd = cfg["boundary"]

    x, z = r[:, 0], r[:, 2]

    # 绘制邻接边
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            if adj[i, j]:
                ax.plot([x[i], x[j]], [z[i], z[j]],
                        color="#c0c0c0", linewidth=0.4, zorder=1)

    # 按配位数分配颜色
    coord_vals = sorted(set(cn.tolist()))
    cmap = plt.colormaps.get_cmap("Set1").resampled(max(len(coord_vals), 3))
    color_map = {c: cmap(k) for k, c in enumerate(coord_vals)}

    for c in coord_vals:
        mask = cn == c
        is_bnd = np.array([i in bnd for i in range(len(r))])
        # 内部离子：实心圆
        interior = mask & ~is_bnd
        boundary = mask & is_bnd
        if interior.any():
            ax.scatter(x[interior], z[interior], s=60, c=[color_map[c]],
                       edgecolors="black", linewidths=0.6, zorder=3,
                       label=f"CN={c}" if not interior.sum() == mask.sum() else None)
        # 边界离子：空心圆
        if boundary.any():
            ax.scatter(x[boundary], z[boundary], s=60, c=[color_map[c]],
                       edgecolors="red", linewidths=1.5, zorder=4,
                       marker="o", facecolors="none" if True else color_map[c])

    # 实际绘制：边界离子用红色边框+半透明填充
    for c in coord_vals:
        mask = cn == c
        is_bnd = np.array([i in bnd for i in range(len(r))])
        boundary = mask & is_bnd
        if boundary.any():
            ax.scatter(x[boundary], z[boundary], s=60,
                       facecolors=color_map[c], alpha=0.5,
                       edgecolors="red", linewidths=1.8, zorder=4)

    ax.set_title(title, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("x (μm)", fontsize=8)
    ax.set_ylabel("z (μm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


def main():
    plt = pyplot()
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser(
        description="Plot the 5 lowest- and 5 highest-energy ion crystal configs "
        "from a collision_pressure config library.",
    )
    parser.add_argument(
        "--n-ions",
        type=int,
        default=54,
        help="Ion count N; reads collision_pressure/configs/N{N}/ (default: 54)",
    )
    args = parser.parse_args()
    n_ions = args.n_ions
    lib_dir = _ROOT / "collision_pressure" / "configs" / f"N{n_ions}"

    configs = load_library(lib_dir)
    n = len(configs)

    picks = configs[:5] + configs[-5:]
    labels = ([f"#{i} (E={c['energy_total_eV']:.6f} eV) — lowest {i+1}"
               for i, c in enumerate(configs[:5])] +
              [f"#{n-5+i} (E={c['energy_total_eV']:.6f} eV) — highest {i+1}"
               for i, c in enumerate(configs[-5:])])

    fig, axes = plt.subplots(2, 5, figsize=(24, 8), constrained_layout=True)
    fig.suptitle(
        f"N={n_ions} Ba-135+ Ion Crystal Configurations "
        f"(Top 5 Lowest / Bottom 5 Highest Energy)",
        fontsize=13,
        fontweight="bold",
    )

    # 统一颜色映射
    all_cn = set()
    for p in picks:
        d = np.load(lib_dir / p["npz_file"])
        all_cn.update(d["coord_numbers"].tolist())
    coord_vals = sorted(all_cn)
    cmap = plt.colormaps.get_cmap("Set1").resampled(max(len(coord_vals), 3))
    color_map = {c: cmap(k) for k, c in enumerate(coord_vals)}

    for idx, (p, label) in enumerate(zip(picks, labels)):
        ax = axes[idx // 5][idx % 5]
        cfg = load_config(lib_dir, p["npz_file"])
        r = cfg["r"]
        cn = cfg["coord_numbers"]
        adj = cfg["adj_matrix"]
        bnd = cfg["boundary"]
        x, z = r[:, 0], r[:, 2]

        # 边
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                if adj[i, j]:
                    ax.plot([x[i], x[j]], [z[i], z[j]],
                            color="#d0d0d0", linewidth=0.3, zorder=1)

        is_bnd = np.array([i in bnd for i in range(len(r))])

        # 内部离子
        interior = ~is_bnd
        for c in coord_vals:
            m = interior & (cn == c)
            if m.any():
                ax.scatter(x[m], z[m], s=55, c=[color_map[c]],
                           edgecolors="black", linewidths=0.5, zorder=3)

        # 边界离子
        for c in coord_vals:
            m = is_bnd & (cn == c)
            if m.any():
                ax.scatter(x[m], z[m], s=55,
                           facecolors=color_map[c], alpha=0.45,
                           edgecolors="red", linewidths=1.6, zorder=4)

        ax.set_title(label, fontsize=7.5, fontfamily="monospace")
        ax.set_xlabel("x (μm)", fontsize=7)
        ax.set_ylabel("z (μm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal")

    # 图例
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c],
               markeredgecolor="black", markersize=8,
               label=f"CN={c} ({'inner' if True else ''})")
        for c in coord_vals
    ]
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="silver",
               markeredgecolor="red", markeredgewidth=1.5, markersize=8,
               alpha=0.5, label="boundary (red edge)")
    )
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    out_path = f"collision_pressure/configs/N{n_ions}/top_bottom_configurations.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
