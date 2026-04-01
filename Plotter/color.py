"""
根据 color_mode 实现不同的离子上色方式
参考 outline.md 与 ism-hybrid/dataplot.py
返回 np.ndarray，供 matplotlib scatter 使用
"""
from typing import Literal

import numpy as np

ColorMode = Literal["y_pos", "v2", "isotope"] | None

# 同位素颜色映射（Ba133/134/135/136/137/138）
ISOTOPE_MASSES = np.array([133 / 135, 134 / 135, 1.0, 136 / 135, 137 / 135, 138 / 135])
ISOTOPE_COLORS = np.array(["yellow", "lime", "red", "blue", "purple", "black"])
ISOTOPE_LABELS = ["Ba133", "Ba134", "Ba135", "Ba136", "Ba137", "Ba138"]


def get_colors(
    r: np.ndarray,
    v: np.ndarray,
    color_mode: ColorMode = None,
    mass: np.ndarray | None = None,
    cmap_name: str = "RdBu",
) -> np.ndarray:
    """
    根据 color_mode 计算离子颜色

    Parameters
    ----------
    r : np.ndarray, shape (N, 3)
        位置
    v : np.ndarray, shape (N, 3)
        速度
    color_mode : None | "y_pos" | "v2" | "isotope"
        None: 全部红色
        "y_pos": 按 y 坐标大小上色（RdBu：小蓝大红）
        "v2": 按速度模平方上色
        "isotope": 按同位素种类上色（需提供 mass）
    mass : np.ndarray, shape (N,), optional
        质量，仅 isotope 模式需要
    cmap_name : str
        连续色图名称，用于 y_pos 和 v2

    Returns
    -------
    colors : np.ndarray
        - color_mode 为 None 或 isotope 时：shape (N,) 颜色名数组
        - color_mode 为 y_pos 或 v2 时：shape (N, 4) RGBA 数组
    """
    N = r.shape[0]

    if color_mode is None:
        return np.full(N, "red", dtype=object)

    if color_mode == "y_pos":
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        values = r[:, 1]
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=values.min(), vmax=values.max())
        return cmap(norm(values))

    if color_mode == "v2":
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        values = np.sum(v**2, axis=1)
        vmin, vmax = values.min(), values.max()
        if vmax <= vmin:
            vmax = vmin + 1e-9
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=vmin, vmax=vmax)
        return cmap(norm(values))

    if color_mode == "isotope":
        if mass is None:
            return np.full(N, "red", dtype=object)
        mass = np.asarray(mass).ravel()
        if mass.size != N:
            return np.full(N, "red", dtype=object)
        indices = np.array(
            [np.argmin(np.abs(ISOTOPE_MASSES - m)) for m in mass],
            dtype=int,
        )
        return ISOTOPE_COLORS[indices]

    return np.full(N, "red", dtype=object)


def get_colors_layerwise_y_pos(
    r: np.ndarray,
    split: int,
    cmap_name: str = "RdBu",
) -> np.ndarray:
    """
    双层绘图：各层独立按 y 做 Min-Max 归一化后着色（与 bilayer_prompt 一致）。
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n = int(r.shape[0])
    split = int(split)
    if split < 0 or split > n:
        raise ValueError(f"split={split} 与 N={n} 不兼容")
    cmap = plt.get_cmap(cmap_name)
    colors = np.zeros((n, 4), dtype=float)
    for sl in (slice(0, split), slice(split, n)):
        y = np.asarray(r[sl, 1], dtype=float)
        if y.size == 0:
            continue
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi <= lo:
            hi = lo + 1e-30
        norm = Normalize(vmin=lo, vmax=hi)
        colors[sl] = cmap(norm(y))
    return colors


def get_mass_indices(mass: np.ndarray) -> np.ndarray:
    """
    将质量数组映射为同位素索引

    Parameters
    ----------
    mass : np.ndarray, shape (N,)
        各离子质量（相对 Ba135）

    Returns
    -------
    indices : np.ndarray, shape (N,), dtype=int
        0~5 对应 Ba133/134/135/136/137/138
    """
    mass = np.asarray(mass).ravel()
    return np.array(
        [np.argmin(np.abs(ISOTOPE_MASSES - m)) for m in mass],
        dtype=int,
    )
