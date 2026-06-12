"""
拉普拉斯方程调和多项式基分解：2D 势场在 xoy 平面内的多极展开

物理原理
--------
离子阱无电荷区域中，静电势满足拉普拉斯方程 ∇²Φ = 0。
在横向 (xoy) 平面固定 z 处，2D 调和多项式基为 Re[(x+iy)^n] 和 Im[(x+iy)^n]。

对于具有 x, y 轴镜像对称性的线性 Paul 阱（V(x,y) = V(-x,y) = V(x,-y)），
仅有偶数阶实部项存活：

  n=0:  1                              （常数）
  n=2:  x² − y²                        （四极项 — 主导束缚势）
  n=4:  x⁴ − 6x²y² + y⁴               （十六极项 — 首阶非谐修正）
  n=6:  x⁶ − 15x⁴y² + 15x²y⁴ − y⁶    （六十四极项）
  ...

严格适用于 DC 静电势和 RF 电势（均满足 Laplace 方程）。
赝势 V_pseudo ∝ |∇Φ_RF|² 不满足 Laplace 方程，仅作经验近似。

参考文献
--------
- D. J. Wineland et al., J. Res. Natl. Inst. Stand. Technol. 103, 259 (1998)
- M. G. House, Phys. Rev. A 78, 033402 (2008)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 调和多项式基函数
# ---------------------------------------------------------------------------

def harmonic_poly_2d(n: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算具有 x,y 镜像对称性的 n 阶 2D 调和多项式。

    即 Re[(x+iy)^n] 对于偶数 n，用二项式系数展开：
        Re[(x+iy)^n] = Σ_{k even} C(n,k) (-1)^{k/2} x^{n-k} y^k

    Parameters
    ----------
    n : int
        阶数（必须为偶数非负）。
    x, y : ndarray
        坐标数组（形状相同）。

    Returns
    -------
    ndarray
        基函数在 (x, y) 处的值。
    """
    if n % 2 != 0:
        raise ValueError(f"仅支持偶数阶基函数；当前 n={n}")
    result = np.zeros_like(x, dtype=float)
    for k in range(0, n + 1, 2):
        coeff = math.comb(n, k) * ((-1) ** (k // 2))
        result = result + coeff * (x ** (n - k)) * (y ** k)
    return result


def _basis_label(n: int) -> str:
    """n 阶调和多项式的人类可读标签。"""
    labels = {
        0: "1",
        2: "x² − y²",
        4: "x⁴ − 6x²y² + y⁴",
        6: "x⁶ − 15x⁴y² + 15x²y⁴ − y⁶",
        8: "x⁸ − 28x⁶y² + 70x⁴y⁴ − 28x²y⁶ + y⁸",
        10: "x¹⁰ − 45x⁸y² + 210x⁶y⁴ − 210x⁴y⁶ + 45x²y⁸ − y¹⁰",
    }
    return labels.get(n, f"Re[(x+iy)^{n}]")


def _basis_unit(n: int) -> str:
    """n 阶系数的物理单位字符串。"""
    units = {0: "V", 2: "V/µm²", 4: "V/µm⁴", 6: "V/µm⁶", 8: "V/µm⁸", 10: "V/µm¹⁰"}
    return units.get(n, f"V/µm^{n}")


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaplaceTerm:
    """拉普拉斯分解中的单项。"""
    degree: int                          # 阶数
    coefficient: float                   # 物理单位系数（V/µm^n）
    coefficient_dimless: float | None    # 无量纲系数 c̃_n = c_n · dl_µm^n / dV
    coefficient_norm: float              # 归一化系数（拟合中间量）
    label: str                           # 人类可读标签


@dataclass(frozen=True)
class LaplaceDecompositionResult:
    """
    2D 拉普拉斯调和多项式分解结果。

    Attributes
    ----------
    terms : tuple[LaplaceTerm, ...]
        按阶数排列的拟合基函数项。
    r_squared : float
        决定系数 R²。
    adjusted_r_squared : float
        调整后 R²（考虑参数数量）。
    rmse : float
        均方根误差（V）。
    relative_rmse : float
        RMSE / RMS(势场)，无量纲相对误差。
    residual_max : float
        最大绝对残差（V）。
    center_um : tuple[float, float]
        展开中心 (cx, cy)，单位 µm。
    scale_um : float
        坐标归一化尺度 L，单位 µm。
    n_points : int
        有效数据点数。
    max_degree : int
        包含的最高调和阶数。
    potential_min : float
        输入势场最小值（V）。
    potential_max : float
        输入势场最大值（V）。
    """
    terms: Tuple[LaplaceTerm, ...]
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    relative_rmse: float
    residual_max: float
    center_um: Tuple[float, float]
    scale_um: float
    n_points: int
    max_degree: int
    potential_min: float
    potential_max: float


@dataclass(frozen=True)
class LaplaceConvergenceResult:
    """拉普拉斯分解收敛性分析结果。"""
    fits: Tuple[LaplaceDecompositionResult, ...]
    degrees: Tuple[int, ...]
    r_squared_list: Tuple[float, ...]
    rmse_list: Tuple[float, ...]


# ---------------------------------------------------------------------------
# 核心拟合
# ---------------------------------------------------------------------------

def fit_laplace_2d(
    x_um: np.ndarray,
    y_um: np.ndarray,
    V: np.ndarray,
    max_degree: int = 4,
    center: Optional[Tuple[float, float]] = None,
    dl_um: Optional[float] = None,
    dV: Optional[float] = None,
) -> LaplaceDecompositionResult:
    """
    使用拉普拉斯调和多项式基拟合 2D 势场。

    Parameters
    ----------
    x_um, y_um : ndarray
        1D 坐标数组（µm）或 2D meshgrid 数组。
    V : ndarray
        2D 势场值（V），形状与 meshgrid 一致。
    max_degree : int
        最高调和多项式阶数（偶数，≥ 2）。
    center : (float, float), optional
        展开中心 (cx, cy) µm。默认使用网格几何中心。
    dl_um : float, optional
        无量纲化长度单位（µm），即 cfg.dl * 1e6。提供后计算无量纲系数。
    dV : float, optional
        无量纲化电压单位（V），即 cfg.dV。提供后计算无量纲系数。

    Returns
    -------
    LaplaceDecompositionResult
    """
    if max_degree < 2 or max_degree % 2 != 0:
        raise ValueError(f"max_degree 须为 ≥ 2 的偶数；当前 {max_degree}")

    # 构建 meshgrid
    if x_um.ndim == 1 and y_um.ndim == 1:
        X, Y = np.meshgrid(x_um, y_um, indexing="ij")
    else:
        X, Y = np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float)

    # 展平
    x_flat = X.ravel()
    y_flat = Y.ravel()
    V_flat = V.ravel()

    # 剔除 NaN / Inf
    valid = np.isfinite(V_flat) & np.isfinite(x_flat) & np.isfinite(y_flat)
    if valid.sum() < 5:
        raise ValueError(f"有效数据点过少（{valid.sum()}），无法拟合")
    xv, yv, Vv = x_flat[valid], y_flat[valid], V_flat[valid]

    # 展开中心
    if center is not None:
        cx, cy = center
    else:
        cx = float((X.max() + X.min()) / 2)
        cy = float((Y.max() + Y.min()) / 2)

    # 坐标归一化到 [-1, 1]，保证数值稳定性
    L = max(np.ptp(X), np.ptp(Y)) / 2
    if L < 1e-12:
        raise ValueError("空间范围过小，无法归一化")

    u = (xv - cx) / L
    v = (yv - cy) / L

    # 构建设计矩阵：列 = [1, x²-y², x⁴-6x²y²+y⁴, ...]
    degrees = list(range(0, max_degree + 1, 2))
    A = np.column_stack([harmonic_poly_2d(d, u, v) for d in degrees])

    # 最小二乘求解
    coeffs_norm, _, _, _ = np.linalg.lstsq(A, Vv, rcond=None)

    # 拟合优度
    V_fit = A @ coeffs_norm
    residuals = Vv - V_fit
    SS_res = float(np.sum(residuals ** 2))
    SS_tot = float(np.sum((Vv - np.mean(Vv)) ** 2))
    n_pts = len(Vv)
    p = len(degrees)

    r_squared = 1 - SS_res / SS_tot if SS_tot > 0 else 0.0
    adj_r2 = (1 - (1 - r_squared) * (n_pts - 1) / (n_pts - p - 1)
              if n_pts > p + 1 else r_squared)
    rmse = float(np.sqrt(SS_res / n_pts))
    rms_V = float(np.sqrt(np.mean(Vv ** 2)))
    relative_rmse = rmse / rms_V if rms_V > 0 else 0.0

    # 物理系数：c_phys = c_norm / L^n
    # 因为 h_n(u,v) = h_n((x-cx)/L, (y-cy)/L) = h_n(x-cx, y-cy) / L^n
    # 无量纲系数：c̃_n = c_phys · dl_µm^n / dV（与 trap_stability 的 anh4/anh6 同方案）
    have_nondim = dl_um is not None and dV is not None and dl_um > 0 and dV != 0

    terms = []
    for d, c_norm in zip(degrees, coeffs_norm):
        c_phys = float(c_norm / (L ** d))
        c_dimless = float(c_phys * dl_um ** d / dV) if have_nondim else None
        terms.append(LaplaceTerm(
            degree=d,
            coefficient=c_phys,
            coefficient_dimless=c_dimless,
            coefficient_norm=float(c_norm),
            label=_basis_label(d),
        ))

    return LaplaceDecompositionResult(
        terms=tuple(terms),
        r_squared=float(r_squared),
        adjusted_r_squared=float(adj_r2),
        rmse=rmse,
        relative_rmse=float(relative_rmse),
        residual_max=float(np.max(np.abs(residuals))),
        center_um=(float(cx), float(cy)),
        scale_um=float(L),
        n_points=n_pts,
        max_degree=max_degree,
        potential_min=float(np.min(Vv)),
        potential_max=float(np.max(Vv)),
    )


def eval_laplace_fit(
    result: LaplaceDecompositionResult,
    x_um: np.ndarray,
    y_um: np.ndarray,
) -> np.ndarray:
    """
    在给定物理坐标处求值拉普拉斯拟合。

    Parameters
    ----------
    result : LaplaceDecompositionResult
        拟合结果。
    x_um, y_um : ndarray
        坐标（µm）。若均为 1D，自动构建 meshgrid 返回 2D 结果。

    Returns
    -------
    ndarray
        拟合势场值。
    """
    # 1D 输入自动 meshgrid
    if x_um.ndim == 1 and y_um.ndim == 1:
        X, Y = np.meshgrid(x_um, y_um, indexing="ij")
    else:
        X, Y = np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float)

    out_shape = X.shape
    x = X.ravel()
    y = Y.ravel()

    cx, cy = result.center_um
    L = result.scale_um
    u = (x - cx) / L
    v = (y - cy) / L

    V = np.zeros_like(x, dtype=float)
    for term in result.terms:
        V = V + term.coefficient_norm * harmonic_poly_2d(term.degree, u, v)

    return V.reshape(out_shape)


# ---------------------------------------------------------------------------
# 收敛性分析
# ---------------------------------------------------------------------------

def laplace_convergence(
    x_um: np.ndarray,
    y_um: np.ndarray,
    V: np.ndarray,
    max_degree: int = 8,
    center: Optional[Tuple[float, float]] = None,
    dl_um: Optional[float] = None,
    dV: Optional[float] = None,
) -> LaplaceConvergenceResult:
    """
    分析拉普拉斯分解随阶数递增的收敛行为。

    依次以 max_degree = 2, 4, 6, ..., max_degree 拟合。

    Returns
    -------
    LaplaceConvergenceResult
    """
    degrees = list(range(2, max_degree + 1, 2))
    results = []
    for d in degrees:
        results.append(fit_laplace_2d(
            x_um, y_um, V, max_degree=d, center=center,
            dl_um=dl_um, dV=dV,
        ))

    return LaplaceConvergenceResult(
        fits=tuple(results),
        degrees=tuple(degrees),
        r_squared_list=tuple(r.r_squared for r in results),
        rmse_list=tuple(r.rmse for r in results),
    )


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def print_laplace_report(
    result: LaplaceDecompositionResult,
    title: str = "Laplace Harmonic Decomposition",
) -> None:
    """格式化打印拉普拉斯分解报告。"""
    has_dimless = any(t.coefficient_dimless is not None for t in result.terms)

    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"  Center:     ({result.center_um[0]:.2f}, {result.center_um[1]:.2f}) µm")
    print(f"  Scale:      {result.scale_um:.2f} µm")
    print(f"  Points:     {result.n_points}")
    print(f"  Max degree: {result.max_degree}")
    v_span = result.potential_max - result.potential_min
    print(f"  V range:    [{result.potential_min:.4g}, {result.potential_max:.4g}] V  "
          f"(span = {v_span:.4g} V)")
    print()
    if has_dimless:
        print(f"  {'Deg':<5} {'Label':<28} {'Coeff (V/µm^n)':<18} {'Coeff (dimless)':<18}")
        print(f"  {'-' * 69}")
        for t in result.terms:
            cd = f"{t.coefficient_dimless:>+14.6e}" if t.coefficient_dimless is not None else "  —"
            print(f"  {t.degree:<5} {t.label:<28} {t.coefficient:>+14.6e}    {cd}")
    else:
        print(f"  {'Deg':<5} {'Label':<28} {'Coeff (V/µm^n)':<18}")
        print(f"  {'-' * 51}")
        for t in result.terms:
            print(f"  {t.degree:<5} {t.label:<28} {t.coefficient:>+14.6e}")
    print()
    print(f"  R²            = {result.r_squared:.10f}")
    print(f"  Adjusted R²   = {result.adjusted_r_squared:.10f}")
    print(f"  RMSE          = {result.rmse:.6e} V")
    print(f"  Relative RMSE = {result.relative_rmse:.6e}")
    print(f"  Max |residual| = {result.residual_max:.6e} V")

    if result.r_squared > 0.9999:
        quality = "Excellent (≥ 99.99%)"
    elif result.r_squared > 0.999:
        quality = "Very good (≥ 99.9%)"
    elif result.r_squared > 0.99:
        quality = "Good (≥ 99%)"
    elif result.r_squared > 0.95:
        quality = "Fair (≥ 95%)"
    else:
        quality = "Poor (< 95%)"
    print(f"  Fit quality:  {quality}")
    print(f"{'=' * 72}\n")


def print_laplace_convergence(conv: LaplaceConvergenceResult) -> None:
    """格式化打印收敛性分析。"""
    print(f"\n{'=' * 50}")
    print("  Laplace Decomposition — Convergence")
    print(f"{'=' * 50}")
    print(f"  {'Max deg':<10} {'R²':<16} {'RMSE (V)':<16} {'ΔR²':<14}")
    print(f"  {'-' * 56}")
    prev_r2 = 0.0
    for d, r2, rmse in zip(conv.degrees, conv.r_squared_list, conv.rmse_list):
        delta = r2 - prev_r2
        print(f"  {d:<10} {r2:<16.10f} {rmse:<16.6e} "
              f"{delta:+.2e}" if d > 2 else
              f"  {d:<10} {r2:<16.10f} {rmse:<16.6e}  —")
        prev_r2 = r2
    print(f"{'=' * 50}\n")
