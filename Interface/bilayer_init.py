"""
双层离子晶格初态：在已有 r0（无量纲）上按索引将两层沿 ±y 分开。
"""
from __future__ import annotations

import numpy as np


def apply_bilayer_y_split(
    r0: np.ndarray,
    n: int,
    y0_um: float,
    dl: float,
) -> np.ndarray:
    """
    索引 ``0:n//2`` 的离子 y 分量增加 ``+y0_um``（μm），``n//2:n`` 减少 ``y0_um``。

    Parameters
    ----------
    r0
        (N, 3) 无量纲坐标（与主程序一致：SI 位移 m / dl）。
    n
        离子数，须为偶数且与 r0 行数一致。
    y0_um
        单层平移量（微米）。
    dl
        长度单位（米）。
    """
    r = np.asarray(r0, dtype=float, order="C").copy()
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"r0 须为 (N,3)，当前 {r.shape}")
    if r.shape[0] != n:
        raise ValueError(f"r0 行数 {r.shape[0]} 与 N={n} 不一致")
    if n % 2 != 0:
        raise ValueError("bilayer 初态要求 N 为偶数")
    dy = float(y0_um) * 1e-6 / float(dl)
    h = n // 2
    r[:h, 1] += dy
    r[h:, 1] -= dy
    return r
