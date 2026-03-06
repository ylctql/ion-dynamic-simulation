"""
势场一维多项式拟合：二次、四次等
采用完整多项式形式 V = a + bx + cx² [+ dx³ + ex⁴]，线性最小二乘，数值稳定
中心由极值点 dV/dx=0 求得，支持阱形与势垒形
"""
from __future__ import annotations

import numpy as np


def _extrema_from_poly(coefs: np.ndarray, x_range: tuple[float, float]) -> float:
    """
    从多项式系数求极值点（dV/dx=0 在 x_range 内的实根）
    coefs = [a, b, c] 或 [a, b, c, d, e]，对应 V = a + bx + cx² + ...
    导数系数：dV/dx = b + 2cx + 3dx² + 4ex³
    """
    if len(coefs) == 3:
        a, b, c = coefs
        if abs(c) < 1e-30:
            return float(np.mean(x_range))
        return float(-b / (2 * c))

    # degree 4: 导数 b + 2cx + 3dx² + 4ex³，系数 [4e, 3d, 2c, b]
    a, b, c, d, e = coefs
    deriv_coefs = np.array([4 * e, 3 * d, 2 * c, b])
    roots = np.roots(deriv_coefs)
    real_roots = roots[np.isreal(roots)].real
    x_min, x_max = min(x_range), max(x_range)
    in_range = real_roots[(real_roots >= x_min - 1e-6) & (real_roots <= x_max + 1e-6)]
    if len(in_range) == 0:
        return float(np.median(x_range))
    # 若有多个极值点，取使 V 最小的（阱）或最大的（势垒）
    V_at_roots = np.polyval(coefs[::-1], in_range)
    return float(in_range[np.argmin(V_at_roots)])


def _k2_at_center(coefs: np.ndarray, center: float) -> float:
    """极值点处的曲率 k2 = V''(center)/2"""
    if len(coefs) == 3:
        a, b, c = coefs
        return float(c)
    a, b, c, d, e = coefs
    # V'' = 2c + 6dx + 12ex²
    return float(c + 3 * d * center + 6 * e * center**2)


def fit_potential_1d(
    coord_1d: np.ndarray,
    V_1d: np.ndarray,
    degree: int = 2,
    center: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    对沿单轴的电势数据做多项式拟合（线性最小二乘）

    模型：V(x) = a + bx + cx² [+ dx³ + ex⁴]
    - 线性拟合，无迭代，数值稳定
    - 中心由极值点 dV/dx=0 求得
    - 可拟合非对称势（奇次项 b,d）

    Parameters
    ----------
    coord_1d : np.ndarray
        沿某轴的坐标 (μm 或归一化单位)
    V_1d : np.ndarray
        该轴上的电势值
    degree : int
        拟合阶数，2=二次，4=四次
    center : float | None
        已弃用，保留仅为接口兼容

    Returns
    -------
    result : np.ndarray
        [center, a, b, c] 或 [center, a, b, c, d, e]
    r_squared : float
        拟合优度 R²
    """
    coord_1d = np.asarray(coord_1d, dtype=float).ravel()
    V_1d = np.asarray(V_1d, dtype=float).ravel()
    if coord_1d.shape != V_1d.shape:
        raise ValueError(f"coord_1d 与 V_1d 长度须一致: {coord_1d.shape} vs {V_1d.shape}")

    valid = np.isfinite(coord_1d) & np.isfinite(V_1d)
    n_coef = degree + 1
    if np.sum(valid) < n_coef:
        raise ValueError(
            f"有效点数 {np.sum(valid)} 不足，至少需 {n_coef} 点做 {degree} 次拟合"
        )
    x = coord_1d[valid]
    y = V_1d[valid]

    # Vandermonde 矩阵：每行为 [1, x, x², x³, x⁴]
    X = np.vander(x, N=n_coef, increasing=True)
    coefs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    coefs = np.atleast_1d(coefs)

    x_range = (float(x.min()), float(x.max()))
    center_fit = _extrema_from_poly(coefs, x_range)

    # result = [center, a, b, c, ...]
    result = np.concatenate([[center_fit], coefs])

    y_pred = eval_fit(x, result, degree)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return result, float(r_squared)


def eval_fit(coord_1d: np.ndarray, fit_result: np.ndarray, degree: int) -> np.ndarray:
    """
    用拟合结果在给定坐标上求值

    Parameters
    ----------
    coord_1d : np.ndarray
        坐标数组
    fit_result : np.ndarray
        [center, a, b, c, ...]，fit_potential_1d 的返回值
    degree : int
        拟合阶数 2 或 4

    Returns
    -------
    V_fit : np.ndarray
        拟合电势值
    """
    x = np.asarray(coord_1d, dtype=float)
    coefs = fit_result[1:]  # [a, b, c] 或 [a, b, c, d, e]
    # np.polyval 需要从高次到低次：[e,d,c,b,a]
    return np.polyval(coefs[::-1], x)


def get_center_and_k2(fit_result: np.ndarray, degree: int) -> tuple[float, float]:
    """从 fit_result 提取 center 与极值点处的 k2（曲率系数）"""
    center = float(fit_result[0])
    coefs = fit_result[1:]
    k2 = _k2_at_center(coefs, center)
    return center, k2
