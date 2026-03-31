"""
双层晶格：将两份 (N,3) 位置在 y 方向错位拼接，并计算纯库仑相互作用的 Hessian。

复用 equilibrium.phonon.coulomb_hessian（单位 eV/μm²，形状 (6N, 6N)）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from equilibrium.phonon import coulomb_hessian


@dataclass(frozen=True)
class BilayerHessianBlockStats:
    """按层块划分的库仑 Hessian 绝对值统计（单位与 H 一致，一般为 eV/μm²）。"""

    intra_diag_abs_mean: float
    intra_diag_abs_max: float
    intra_offdiag_abs_mean: float
    intra_offdiag_abs_max: float
    inter_abs_mean: float
    inter_abs_max: float
    n_intra_diag: int
    n_intra_offdiag: int
    n_inter: int


def bilayer_hessian_block_stats(
    hessian_eV_per_um2: np.ndarray,
    n_ions_per_layer: int,
) -> BilayerHessianBlockStats:
    """
    将总 Hessian 按两层 DOF 分块统计 |H_ij|。

    - **同层**：下标均属于第一层 ``0:3N`` 或均属于第二层 ``3N:6N``。
    - **同层对角元**：``H[k,k]``（等价于块 ``H[:3N,:3N]`` 与 ``H[3N:,3N:]`` 的对角并集）。
    - **同层非对角元**：上述两块内部的 ``i!=j`` 元素（上下三角均计入）。
    - **层间**：仅取矩形 ``H[:3N, 3N:]``（形状 ``(3N,3N)``），每个层间离子对的耦合块只计一次。
    """
    h = np.asarray(hessian_eV_per_um2, dtype=float)
    nl = int(n_ions_per_layer)
    m = 3 * nl
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] != 2 * m:
        raise ValueError(
            f"期望 Hessian 形状 (6N,6N) 且 N={nl}，当前 {h.shape}"
        )

    diag = np.diag(h)
    intra_diag_abs = np.abs(diag)
    h11 = h[:m, :m]
    h22 = h[m:, m:]
    eye = np.eye(m, dtype=bool)
    off11 = h11[~eye]
    off22 = h22[~eye]
    intra_off_abs = np.abs(np.concatenate([off11.ravel(), off22.ravel()]))

    h12 = h[:m, m:]
    inter_abs = np.abs(h12.ravel())

    def _mean_max(a: np.ndarray) -> tuple[float, float]:
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(a)), float(np.max(a))

    dm, dM = _mean_max(intra_diag_abs)
    om, oM = _mean_max(intra_off_abs)
    im, iM = _mean_max(inter_abs)

    return BilayerHessianBlockStats(
        intra_diag_abs_mean=dm,
        intra_diag_abs_max=dM,
        intra_offdiag_abs_mean=om,
        intra_offdiag_abs_max=oM,
        inter_abs_mean=im,
        inter_abs_max=iM,
        n_intra_diag=int(intra_diag_abs.size),
        n_intra_offdiag=int(intra_off_abs.size),
        n_inter=int(inter_abs.size),
    )


def load_positions_from_npz(path: str | Path, key: str = "r") -> np.ndarray:
    """从 npz 读取离子位置，形状 (N, 3)，单位 μm。"""
    path = Path(path)
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"{path} 中无数组 {key!r}，仅有: {data.files}")
        r = np.asarray(data[key], dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"位置应为 (N,3)，当前 {r.shape}")
    return r


def stack_bilayer_along_y(
    r_layer1_um: np.ndarray,
    r_layer2_um: np.ndarray,
    layer_spacing_um: float,
) -> np.ndarray:
    """
    在 +y 方向拼接两层：第二层在自身坐标基础上整体平移 ``layer_spacing_um``（μm）。

    Parameters
    ----------
    r_layer1_um, r_layer2_um
        各 (N, 3)，单位 μm。
    layer_spacing_um
        沿 +y 施加在第二层上的平移量（μm），即用户指定的层间距变量。

    Returns
    -------
    np.ndarray
        (2N, 3)，[第一层; 第二层平移后]。
    """
    r1 = np.asarray(r_layer1_um, dtype=float)
    r2 = np.asarray(r_layer2_um, dtype=float)
    if r1.shape != r2.shape:
        raise ValueError(f"两层离子数/形状须一致: {r1.shape} vs {r2.shape}")
    offset = np.array([0.0, float(layer_spacing_um), 0.0], dtype=float)
    r2_shifted = r2 + offset
    return np.vstack([r1, r2_shifted])


def coulomb_hessian_bilayer(
    r_layer1_um: np.ndarray,
    r_layer2_um: np.ndarray,
    layer_spacing_um: float,
    charge_ec: np.ndarray | None = None,
    softening_um: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    拼接双层并返回 (位置, 库仑 Hessian)。

    Returns
    -------
    r_bilayer_um : (2N, 3)
    hessian_eV_per_um2 : (6N, 6N)
    """
    r_bi = stack_bilayer_along_y(r_layer1_um, r_layer2_um, layer_spacing_um)
    n = r_bi.shape[0]
    if charge_ec is None:
        q = np.ones(n, dtype=float)
    else:
        q = np.asarray(charge_ec, dtype=float).ravel()
        if q.shape[0] != n:
            raise ValueError(f"charge 长度 {q.shape[0]} 与双层离子数 {n} 不一致")
    h = coulomb_hessian(r_bi, q, softening_um=softening_um)
    return r_bi, h
