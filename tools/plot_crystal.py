"""
从 saves/rv/traj 读取离子晶格坐标（.npz：r 单位为 μm，列为 x,y,z），
统计 y 方向极差与标准差，并在 zox 平面上按 y_pos 模式绘图（与 Plotter 约定一致）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Plotter.color import get_colors

# 默认与 main/DataPlotter 保存结构一致：{root}/{device}/{N}/t{time 两位小数}us.npz
DEFAULT_TRAJ_DIR = _ROOT / "saves/rv/traj/cuda/10000"

_NPZ_NAME_RE = re.compile(r"^t([\d.]+)us\.npz$", re.IGNORECASE)


def list_traj_npz(directory: Path | str) -> list[Path]:
    """
    列出目录下轨迹 npz，按文件名中的时刻（μs）升序排列。
    """
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"目录不存在: {d}")
    files: list[tuple[float, Path]] = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        m = _NPZ_NAME_RE.match(p.name)
        if m:
            files.append((float(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def load_rv_npz(path: Path | str) -> tuple[np.ndarray, np.ndarray, float]:
    """
    读取单帧 r/v。

    Returns
    -------
    r_um : (N, 3), μm，列顺序 x, y, z
    v_m_s : (N, 3), m/s（若无 v 键则返回零数组）
    t_us : float，时刻 μs（若无则 nan）
    """
    path = Path(path)
    data = np.load(path)
    r = np.asarray(data["r"], dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"r 形状应为 (N,3)，当前 {r.shape} ({path})")
    if "v" in data:
        v = np.asarray(data["v"], dtype=np.float64)
    else:
        v = np.zeros_like(r)
    t_us = float(data["t_us"]) if "t_us" in data else float("nan")
    return r, v, t_us


def y_extent_and_std(r_um: np.ndarray) -> tuple[float, float]:
    """
    y 方向极差（max - min）与总体标准差，单位与 r 相同（μm）。
    """
    y = np.asarray(r_um[:, 1], dtype=np.float64)
    extent = float(np.max(y) - np.min(y))
    std = float(np.std(y, ddof=0))
    return extent, std


def get_xy_zox(r_um: np.ndarray) -> np.ndarray:
    """zox 投影：(横轴 z, 纵轴 x)，单位同 r（μm）。与 Plotter/dataplot._get_xy 一致。"""
    return np.column_stack((r_um[:, 2], r_um[:, 0]))


def plot_zox_y_pos(
    r_um: np.ndarray,
    v_m_s: np.ndarray | None = None,
    *,
    ax=None,
    ion_size: float = 5.0,
    title: str | None = None,
    z_range_um: tuple[float, float] | None = None,
    x_range_um: tuple[float, float] | None = None,
):
    """
    在 zox 平面上散点图，颜色为 y_pos（RdBu，与 Plotter/color.get_colors 一致）。

    Parameters
    ----------
    r_um, v_m_s
        位置 μm、速度 m/s；y_pos 仅用到 r 的 y 列，v 可省略。
    z_range_um, x_range_um
        若给定则 set_xlim/set_ylim，否则按数据各留 5% 边距。
    """
    import matplotlib.pyplot as plt

    r_um = np.asarray(r_um, dtype=np.float64)
    if v_m_s is None:
        v_m_s = np.zeros_like(r_um)
    else:
        v_m_s = np.asarray(v_m_s, dtype=np.float64)

    colors = get_colors(r_um, v_m_s, "y_pos", mass=None, cmap_name="RdBu")
    xy = get_xy_zox(r_um)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=ion_size, c=colors)
    ax.set_xlabel("z (μm)", fontsize=14)
    ax.set_ylabel("x (μm)", fontsize=14)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if z_range_um is not None:
        ax.set_xlim(z_range_um[0], z_range_um[1])
    if x_range_um is not None:
        ax.set_ylim(x_range_um[0], x_range_um[1])
    if z_range_um is None and x_range_um is None:
        zc, xc = xy[:, 0], xy[:, 1]
        if zc.size and xc.size:
            pad_z = 0.05 * max(np.ptp(zc), 1e-9)
            pad_x = 0.05 * max(np.ptp(xc), 1e-9)
            ax.set_xlim(float(zc.min() - pad_z), float(zc.max() + pad_z))
            ax.set_ylim(float(xc.min() - pad_x), float(xc.max() + pad_x))
    return ax, sc


def summarize_directory(directory: Path | str) -> list[dict[str, float | str]]:
    """对目录内每个 npz 计算 t_us、y 极差、y 标准差。"""
    rows: list[dict[str, float | str]] = []
    for p in list_traj_npz(directory):
        r, _, t_us = load_rv_npz(p)
        ext, std = y_extent_and_std(r)
        rows.append(
            {
                "file": str(p.name),
                "t_us": t_us,
                "y_extent_um": ext,
                "y_std_um": std,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="读取 rv 轨迹 npz，统计 y 极差/标准差并绘制 zox+y_pos")
    p.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_TRAJ_DIR,
        help=f"含 t*us.npz 的目录（默认 {DEFAULT_TRAJ_DIR}）",
    )
    p.add_argument(
        "--file",
        type=Path,
        default=None,
        help="仅处理单个 npz（指定时忽略 --dir 批量逻辑，仍可用 --out 保存该帧图）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="保存图片路径；不指定则尝试 plt.show()",
    )
    p.add_argument("--ion_size", type=float, default=5.0, help="散点大小")
    return p.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt

    args = _parse_args()
    if args.file is not None:
        r, v, t_us = load_rv_npz(args.file)
        ext, std = y_extent_and_std(r)
        print(f"{args.file.name}: t_us={t_us:g}  y极差={ext:.6g} μm  y_std={std:.6g} μm")
        _, _ = plot_zox_y_pos(
            r,
            v,
            ion_size=args.ion_size,
            title=f"zox (y_pos), t={t_us:g} μs",
        )
    else:
        rows = summarize_directory(args.dir)
        if not rows:
            print(f"目录中未找到 t*us.npz: {args.dir}")
            return
        for row in rows:
            print(
                f"{row['file']}: t_us={row['t_us']:g}  "
                f"y极差={row['y_extent_um']:.6g} μm  y_std={row['y_std_um']:.6g} μm"
            )
        last = rows[-1]
        r, v, t_us = load_rv_npz(Path(args.dir) / str(last["file"]))
        _, _ = plot_zox_y_pos(
            r,
            v,
            ion_size=args.ion_size,
            title=f"zox (y_pos), t={t_us:g} μs ({last['file']})",
        )

    plt.tight_layout()
    if args.out is not None:
        args.out = Path(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=200)
        print(f"已保存: {args.out}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
