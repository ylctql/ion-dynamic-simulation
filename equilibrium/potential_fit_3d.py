"""
总势场 3D 多项式拟合（缩放坐标 u,v,w）

fit_potential_3d_quartic: V_shifted = Σ c_{ijk} u^i v^j w^k，多项式形式由两个**正交**维度决定：

- ``fit_mode``（非负整数 N）：总次数 i+j+k≤N 的完整三元多项式基，C(N+3,3) 项（如 2→10、4→35、6→84）。
  仅指定"用哪些次数的项"。
- ``symmetry_axes``（{'x','y','z'} 的子集，可省略）：在上述基底上**拟合前**剔除对称轴指数为奇数的单项式，
  强制势场关于 center_um 处相应坐标平面镜面对称（V(x₀+δ)=V(x₀−δ) ⟺ 该轴仅保留偶次项）。
  例如 ``symmetry_axes=("x","z")`` 删去所有 i 或 k 为奇的项；``("x","y","z")`` 仅留全偶项。

二者组合即可表达任意想要的对称多项式形式：例如 N=2 + 全对称 ≡ 旧 quadratic（常数 + u²,v²,w²）。

系数存 (D+1,D+1,D+1)，D 为基底每变量最高次且 D≥4，未用位置为 0，与 numpy polynomial.polyval3d 兼容；
求值/梯度/Hessian 均按 coeffs 实际维度动态计算。
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.polynomial import polynomial as P


DEGREE_QUARTIC = 4

# 对称轴名 → 指数位（与 (i,j,k) 顺序一致）
_AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


def quartic_3d_exponents_total_degree(max_total: int = DEGREE_QUARTIC) -> tuple[tuple[int, int, int], ...]:
    """
    三元总次数基底：所有 (i,j,k) 满足 i+j+k ≤ max_total，按总次数 s 递增，再按 i、j 字典序枚举。
    max_total=4 时共 C(4+3,3)=35 项。
    """
    out: list[tuple[int, int, int]] = []
    for s in range(max_total + 1):
        for i in range(s + 1):
            for j in range(s + 1 - i):
                k = s - i - j
                out.append((i, j, k))
    return tuple(out)


QUARTIC_3D_EXPS: tuple[tuple[int, int, int], ...] = quartic_3d_exponents_total_degree(DEGREE_QUARTIC)
N_QUARTIC_3D_TERMS = len(QUARTIC_3D_EXPS)

# 理想谐振阱专用：常数项 + u², v², w²（= N=2 + 全对称 的子集，JSON 标签为 1, x^2, y^2, z^2）
QUADRATIC_FIT_EXPS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
)


def normalize_fit_mode(fit_mode: int | str) -> int:
    """
    将 fit_mode 规范化为非负整数 N（总次数）。

    接受非负整数，或可解析为非负整数的字符串（如 ``"6"``，便于 CLI 透传）。
    不接受 None / 布尔 / 负数 / 任意字符串——多项式形式完全由整数次数 N 表达。
    """
    # bool 是 int 子类，先排除
    if isinstance(fit_mode, bool):
        raise ValueError(f"fit_mode 不接受布尔值: {fit_mode!r}")
    if isinstance(fit_mode, int):
        if fit_mode < 0:
            raise ValueError(f"fit_mode 须为非负整数（总次数 N），得到 {fit_mode}")
        return fit_mode
    s = str(fit_mode).strip()
    if s.isdigit():  # 数字字符串 → 整数（便于 CLI 透传）
        return int(s)
    raise ValueError(
        f"fit_mode={fit_mode!r} 不受支持，请使用非负整数（总次数 N，如 4→35 项）"
    )


def normalize_symmetry_axes(
    symmetry_axes: str | tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """
    将对称轴输入规范化为按 (x,y,z) 顺序排列的唯一轴元组。

    接受：
    - None / 空串 / 空容器 → () （无对称约束）；
    - 字符串："x,z" 或 "xz" 或 "x"（逗号分隔或直接拼接均可）；
    - 序列：("x","z")、["y"] 等。

    每个轴须 ∈ {'x','y','z'}（大小写不敏感），重复轴自动去重。
    """
    if symmetry_axes is None:
        return ()
    if isinstance(symmetry_axes, str):
        tokens = symmetry_axes.replace(",", " ").split()
    else:
        tokens = [str(a) for a in symmetry_axes]
    # 允许拼接形式 "xz" → 拆成单字符
    chars: list[str] = []
    for tok in tokens:
        tok = tok.strip().lower()
        if len(tok) == 1:
            chars.append(tok)
        elif len(tok) > 1 and all(c in _AXIS_INDEX for c in tok):
            chars.extend(list(tok))
        elif tok:
            raise ValueError(
                f"symmetry_axes 含非法轴 {tok!r}，仅支持 x/y/z 的子集（如 'x,z' 或 'xz'）"
            )
    bad = [c for c in chars if c not in _AXIS_INDEX]
    if bad:
        raise ValueError(
            f"symmetry_axes 含非法轴 {bad}，仅支持 x/y/z 的子集（如 'x,z' 或 'xz'）"
        )
    # 去重并按 (x,y,z) 固定顺序排列
    return tuple(ax for ax in ("x", "y", "z") if ax in set(chars))


def filter_basis_by_symmetry(
    basis_exps: tuple[tuple[int, int, int], ...],
    symmetry_axes: str | tuple[str, ...] | list[str] | None,
) -> tuple[tuple[int, int, int], ...]:
    """
    从基底中剔除违反镜面对称的单项式：对称轴上指数为奇数的项一律删除。

    这是**拟合前**的基底选择——被剔除的单项式根本不进入设计矩阵，最小二乘仅在
    保留项张成的对称子空间上求解（而非"全拟合后置零"），故保留项系数会被重新优化，
    把被剔除方向上的（噪声/非对称）投影吸收掉。
    """
    axes = normalize_symmetry_axes(symmetry_axes)
    if not axes:
        return basis_exps
    odd_positions = {_AXIS_INDEX[a] for a in axes}
    return tuple(
        (i, j, k)
        for (i, j, k) in basis_exps
        if all(exp % 2 == 0 for pos, exp in enumerate((i, j, k)) if pos in odd_positions)
    )


def fit_mode_basis_exponents(
    fit_mode: int | str,
    symmetry_axes: str | tuple[str, ...] | list[str] | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """
    返回拟合用单项式指数 (i,j,k)，对应 u^i v^j w^k。

    先按 ``fit_mode``（非负整数 N）取总次数 ≤N 的完整基，再按 ``symmetry_axes``
    剔除对称轴上奇次项。返回的基底即实际参与最小二乘的列集合。
    """
    n = normalize_fit_mode(fit_mode)
    basis = quartic_3d_exponents_total_degree(n)
    return filter_basis_by_symmetry(basis, symmetry_axes)


def _monomial_factor(var: str, exp: int) -> str:
    if exp <= 0:
        return ""
    if exp == 1:
        return var
    return f"{var}^{exp}"


def quartic_3d_term_label(i: int, j: int, k: int) -> str:
    """
    项名字符串：monomial x^i y^j z^k，
    其中 x,y,z 表示缩放位移 u,v,w（即 (x_phys-x0)/L 等）。
    """
    parts: list[str] = []
    for var, exp in (("x", i), ("y", j), ("z", k)):
        fac = _monomial_factor(var, exp)
        if fac:
            parts.append(fac)
    return "*".join(parts) if parts else "1"


def quartic_fit_coeff_map(fit: FitResult3D) -> dict[str, float]:
    """拟合系数表：项名 -> 数值（与本次拟合所用 basis_exps 顺序一致）。"""
    out: dict[str, float] = {}
    c = fit.coeffs
    for i, j, k in fit.basis_exps:
        key = quartic_3d_term_label(i, j, k)
        out[key] = float(c[i, j, k])
    return out


def write_potential_fit_coeff_json(
    fit: FitResult3D,
    path: Path | str,
    *,
    csv: str | Path | None = None,
    config: str | Path | None = None,
) -> None:
    """
    将多项式拟合各阶项系数写入 JSON。

    文件结构：csv、config、fit_mode、symmetry_axes，随后 coefficients（项名 -> float）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    csv_name = Path(csv).name if csv else ""
    config_name = Path(config).name if config else ""
    payload: dict[str, object] = {
        "csv": csv_name,
        "config": config_name,
        "fit_mode": fit.fit_mode if fit.fit_mode is not None else "none",
        "symmetry_axes": list(fit.symmetry_axes),
        "center_um": [float(c) for c in fit.center_um],
        "scale_um": float(fit.scale_um),
        "potential_offset_V": float(fit.potential_offset_V),
        "coefficients": quartic_fit_coeff_map(fit),
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@dataclass
class FitResult3D:
    """
    3D 多项式拟合结果

    势场模型: V_shifted = Σ c_ijk u^i v^j w^k，u=(x-x0)/L, v=(y-y0)/L, w=(z-z0)/L，
    仅 (fit_mode, symmetry_axes) 选定基底中的项非零（见 fit_mode_basis_exponents）；
    存于 (D+1)³ 张量，D≥4，故低次/对称削项基底仍占 (5,5,5)，未参与拟合的位置恒为 0。
    为数值稳定，拟合在缩放坐标上进行。L 为半跨度。
    V_shifted = V_true - V_min_ref（将零点平移到参考最小势）。坐标单位: μm
    """

    coeffs: np.ndarray  # shape (D+1,D+1,D+1)，D≥4；c[i,j,k] 对应 u^i v^j w^k，未使用的项恒为 0
    center_um: tuple[float, float, float]
    scale_um: float  # L，坐标缩放半跨度
    potential_offset_V: float  # 参考最小势 V_min_ref（被减去）
    r_squared: float
    fit_mode: int | None = None  # 本次拟合的总次数 N；None 表示非拟合来源（显式系数/理想阱）
    symmetry_axes: tuple[str, ...] = ()  # 拟合时施加的对称轴子集（规范化的 (x,y,z) 子序）
    basis_exps: tuple[tuple[int, int, int], ...] = field(default_factory=lambda: QUARTIC_3D_EXPS)

    def _build_grad_coeffs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-compute derivative coefficient arrays for grad_fit_3d（按 coeffs 实际维度动态）。"""
        c = self.coeffs
        n = c.shape[0]          # = D+1
        D = n - 1               # 每变量最高次
        if D < 1:               # 常数势：梯度恒为 0
            return np.zeros((1, n, n)), np.zeros((n, 1, n)), np.zeros((n, n, 1))
        c_du = np.zeros((D, n, n))
        for i in range(D):
            c_du[i, :, :] = c[i + 1, :, :] * (i + 1)
        c_dv = np.zeros((n, D, n))
        for j in range(D):
            c_dv[:, j, :] = c[:, j + 1, :] * (j + 1)
        c_dw = np.zeros((n, n, D))
        for k in range(D):
            c_dw[:, :, k] = c[:, :, k + 1] * (k + 1)
        return c_du, c_dv, c_dw

    def grad_coeffs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached (c_du, c_dv, c_dw) for grad_fit_3d, computed once."""
        if not hasattr(self, '_grad_coeffs_cache'):
            self._grad_coeffs_cache = self._build_grad_coeffs()
        return self._grad_coeffs_cache


def fit_potential_3d_quartic(
    compute_V_total: Callable[[np.ndarray], np.ndarray],
    um_to_norm: Callable[[float], float],
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    n_pts_per_axis: int | tuple[int, int, int] = 8,
    potential_offset_V: float | None = None,
    fit_mode: int | str = 4,
    symmetry_axes: str | tuple[str, ...] | list[str] | None = None,
) -> FitResult3D:
    """
    对总势场在缩放坐标下做多项式最小二乘拟合，并将势能零点平移到参考最小势。

    多项式形式由 ``fit_mode``（次数）与 ``symmetry_axes``（对称轴）共同决定（见模块文档）。
    对称约束在**拟合前**从基底中剔除违反镜面对称的单项式，最小二乘仅在保留的对称子空间上求解。

    Parameters
    ----------
    compute_V_total : callable
        接收 r (N, 3) 归一化坐标，返回 V_total (N,) 单位 V
    um_to_norm : callable
        μm → 归一化坐标的转换函数
    center_um : tuple
        参考中心 (x0, y0, z0) μm；对称约束关于此点（镜像面 x=x0 等），须设为真实对称中心。
    range_um : tuple of tuples, optional
        各轴拟合范围 ((x_min, x_max), (y_min, y_max), (z_min, z_max)) μm
        默认 (-50, 50) 每轴
    n_pts_per_axis : int | tuple[int, int, int]
        采样点数。可传单个整数（x/y/z 相同），或传 (nx, ny, nz) 分别指定三轴采样点数。
        基函数多时须保证有效网格点数 ≥ 未知数个数。
    potential_offset_V : float | None
        势能零点平移参考值 V_min_ref（单位 V）。若为 None，则退化为当前拟合采样点的最小势。
    fit_mode : int | str
        非负整数 N：总次数 i+j+k≤N 的完整基（C(N+3,3) 项，如 4→35、6→84）。
        默认 4（= quartic，35 项）。
    symmetry_axes : str | tuple[str,...] | None
        对称轴子集 {'x','y','z'}，如 "x,z" 或 ("x","z")；None/空表示不施加对称约束。
        每个列出的轴强制镜面对称（剔除该轴奇次单项式）。

    Returns
    -------
    FitResult3D
        拟合结果
    """
    if range_um is None:
        range_um = ((-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0))

    (xr0, xr1), (yr0, yr1), (zr0, zr1) = range_um
    x0, y0, z0 = center_um
    # 缩放：u = (x-x0)/L，使 u,v,w ∈ [-1,1]，提高数值稳定性
    Lx = max(xr1 - x0, x0 - xr0, 1e-6)
    Ly = max(yr1 - y0, y0 - yr0, 1e-6)
    Lz = max(zr1 - z0, z0 - zr0, 1e-6)
    scale_um = max(Lx, Ly, Lz)

    xr_n = (um_to_norm(xr0), um_to_norm(xr1))
    yr_n = (um_to_norm(yr0), um_to_norm(yr1))
    zr_n = (um_to_norm(zr0), um_to_norm(zr1))

    # 3D 网格采样（支持三轴不同采样数）
    if isinstance(n_pts_per_axis, int):
        nx = ny = nz = int(n_pts_per_axis)
    else:
        nx, ny, nz = [int(v) for v in n_pts_per_axis]
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError(f"n_pts_per_axis 每轴至少为 2，当前为 ({nx}, {ny}, {nz})")

    x_um = np.linspace(xr0, xr1, nx)
    y_um = np.linspace(yr0, yr1, ny)
    z_um = np.linspace(zr0, zr1, nz)
    xx, yy, zz = np.meshgrid(x_um, y_um, z_um, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = zz.ravel()

    # 缩放坐标 u,v,w ∈ [-1,1]
    u_flat = (x_flat - x0) / scale_um
    v_flat = (y_flat - y0) / scale_um
    w_flat = (z_flat - z0) / scale_um

    # 归一化坐标供 compute_V_total 使用
    r_norm = np.column_stack([
        um_to_norm(x_flat),
        um_to_norm(y_flat),
        um_to_norm(z_flat),
    ])
    V_true = compute_V_total(r_norm)

    mode_key = normalize_fit_mode(fit_mode)
    sym_axes = normalize_symmetry_axes(symmetry_axes)
    # 基底 = 总次数≤N 的完整基，再按对称轴剔除奇次项（拟合前的形式约束）
    basis_exps = fit_mode_basis_exponents(mode_key, sym_axes)
    n_basis = len(basis_exps)

    # 设计矩阵：仅保留基底中的项，缩放坐标 u,v,w
    V_mat = np.empty((u_flat.size, n_basis), dtype=float)
    for col, (i, j, k) in enumerate(basis_exps):
        V_mat[:, col] = (u_flat**i) * (v_flat**j) * (w_flat**k)
    valid = np.isfinite(V_true)
    if np.sum(valid) < n_basis:
        raise ValueError(
            f"有效采样点 {np.sum(valid)} 不足，至少需 {n_basis} 点拟合"
            f"（fit_mode={mode_key}，symmetry_axes={sym_axes or '()'}，"
            f"当前基底项数 {n_basis}）。"
            f"请增大 n_pts_per_axis（当前网格 {nx}x{ny}x{nz}）或检查势场范围。"
        )
    V_mat_valid = V_mat[valid]
    V_true_valid = V_true[valid]

    # 势能零点平移：默认减去拟合采样最小值；若给定参考值则统一使用该参考值
    if potential_offset_V is None:
        v_min_ref = float(np.min(V_true_valid))
    else:
        v_min_ref = float(potential_offset_V)
    V_shifted_valid = V_true_valid - v_min_ref

    coefs_flat, residuals, rank, s = np.linalg.lstsq(V_mat_valid, V_shifted_valid, rcond=None)
    # 系数数组维度 = 每变量最高次 +1；至少 (5,5,5) 以兼容低次/对称削项基底
    deg = max(max(e) for e in basis_exps)
    deg = max(deg, DEGREE_QUARTIC)
    coefs = np.zeros((deg + 1, deg + 1, deg + 1))
    for idx, (i, j, k) in enumerate(basis_exps):
        coefs[i, j, k] = coefs_flat[idx]

    # R²
    V_pred = V_mat_valid @ coefs_flat
    ss_res = np.sum((V_shifted_valid - V_pred) ** 2)
    ss_tot = np.sum((V_shifted_valid - np.mean(V_shifted_valid)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return FitResult3D(
        coeffs=coefs,
        center_um=center_um,
        scale_um=scale_um,
        potential_offset_V=v_min_ref,
        r_squared=r_squared,
        fit_mode=mode_key,
        symmetry_axes=sym_axes,
        basis_exps=basis_exps,
    )


def eval_fit_3d(fit: FitResult3D, r_um: np.ndarray) -> np.ndarray:
    """
    用 3D 拟合结果在给定坐标上求值。

    Parameters
    ----------
    fit : FitResult3D
        fit_potential_3d_quartic 的返回值
    r_um : np.ndarray, shape (N, 3)
        坐标 (x, y, z) 单位 μm

    Returns
    -------
    V : np.ndarray, shape (N,)
        拟合电势值，单位 V
    """
    r_um = np.atleast_2d(r_um)
    x0, y0, z0 = fit.center_um
    L = fit.scale_um
    u = (r_um[:, 0] - x0) / L
    v = (r_um[:, 1] - y0) / L
    w = (r_um[:, 2] - z0) / L
    return P.polyval3d(u, v, w, fit.coeffs)


def grad_fit_3d(fit: FitResult3D, r_um: np.ndarray) -> np.ndarray:
    """
    用 3D 拟合结果计算电势梯度 dV/dr，单位 V/μm。

    拟合在 (u,v,w) 上，u=(x-x0)/L，故 dV/dx = (dV/du) / L
    """
    r_um = np.atleast_2d(r_um)
    x0, y0, z0 = fit.center_um
    L = fit.scale_um
    u = (r_um[:, 0] - x0) / L
    v = (r_um[:, 1] - y0) / L
    w = (r_um[:, 2] - z0) / L

    c_du, c_dv, c_dw = fit.grad_coeffs()
    dV_du = P.polyval3d(u, v, w, c_du)
    dV_dv = P.polyval3d(u, v, w, c_dv)
    dV_dw = P.polyval3d(u, v, w, c_dw)

    return np.column_stack([dV_du / L, dV_dv / L, dV_dw / L])


def hessian_fit_3d(fit: FitResult3D, r_um: np.ndarray) -> np.ndarray:
    """
    用 3D 拟合结果计算电势 Hessian，单位 V/μm^2。

    Returns
    -------
    hess : np.ndarray, shape (N, 3, 3)
        每个点对应一个 3x3 对称 Hessian 矩阵，变量顺序为 (x, y, z)。
    """
    r_um = np.atleast_2d(r_um)
    x0, y0, z0 = fit.center_um
    L = fit.scale_um
    u = (r_um[:, 0] - x0) / L
    v = (r_um[:, 1] - y0) / L
    w = (r_um[:, 2] - z0) / L
    c = fit.coeffs
    n = c.shape[0]          # = D+1
    D = n - 1               # 每变量最高次

    # 二阶纯偏导（需 D≥2；D<2 时该方向二阶导恒 0，避免访问越界索引）
    if D >= 2:
        c_duu = np.zeros((D - 1, n, n))
        for i in range(D - 1):
            c_duu[i, :, :] = c[i + 2, :, :] * (i + 2) * (i + 1)
        d2V_duu = P.polyval3d(u, v, w, c_duu)

        c_dvv = np.zeros((n, D - 1, n))
        for j in range(D - 1):
            c_dvv[:, j, :] = c[:, j + 2, :] * (j + 2) * (j + 1)
        d2V_dvv = P.polyval3d(u, v, w, c_dvv)

        c_dww = np.zeros((n, n, D - 1))
        for k in range(D - 1):
            c_dww[:, :, k] = c[:, :, k + 2] * (k + 2) * (k + 1)
        d2V_dww = P.polyval3d(u, v, w, c_dww)
    else:
        d2V_duu = np.zeros_like(u)
        d2V_dvv = np.zeros_like(u)
        d2V_dww = np.zeros_like(u)

    # 二阶混合偏导（需 D≥1）
    h1 = max(D, 1)
    c_duv = np.zeros((h1, h1, n))
    for i in range(h1):
        for j in range(h1):
            c_duv[i, j, :] = c[i + 1, j + 1, :] * (i + 1) * (j + 1)
    d2V_duv = P.polyval3d(u, v, w, c_duv)

    c_duw = np.zeros((h1, n, h1))
    for i in range(h1):
        for k in range(h1):
            c_duw[i, :, k] = c[i + 1, :, k + 1] * (i + 1) * (k + 1)
    d2V_duw = P.polyval3d(u, v, w, c_duw)

    c_dvw = np.zeros((n, h1, h1))
    for j in range(h1):
        for k in range(h1):
            c_dvw[:, j, k] = c[:, j + 1, k + 1] * (j + 1) * (k + 1)
    d2V_dvw = P.polyval3d(u, v, w, c_dvw)

    # 链式法则：每个一阶导都会引入 1/L，因此二阶导统一为 1/L^2
    scale2 = L * L
    hxx = d2V_duu / scale2
    hyy = d2V_dvv / scale2
    hzz = d2V_dww / scale2
    hxy = d2V_duv / scale2
    hxz = d2V_duw / scale2
    hyz = d2V_dvw / scale2

    n = r_um.shape[0]
    hess = np.zeros((n, 3, 3), dtype=float)
    hess[:, 0, 0] = hxx
    hess[:, 1, 1] = hyy
    hess[:, 2, 2] = hzz
    hess[:, 0, 1] = hxy
    hess[:, 1, 0] = hxy
    hess[:, 0, 2] = hxz
    hess[:, 2, 0] = hxz
    hess[:, 1, 2] = hyz
    hess[:, 2, 1] = hyz
    return hess


# ---------------------------------------------------------------------------
# 理想二次势（由阱频构造 FitResult3D）
# ---------------------------------------------------------------------------

# "quadratic" 基底：常数 + u²,v²,w²
_QUADRATIC_BASIS_EXPS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2),
)


def make_ideal_trap_fit(
    freq_MHz: tuple[float, float, float],
    mass_amu: float,
    charge_ec: float = 1.0,
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale_um: float = 100.0,
) -> FitResult3D:
    """
    从阱频构造理想二次势 FitResult3D，无需 CSV 数据。

    势场模型：V = ax*x² + ay*y² + az*z²  (V, µm)
    其中 ax = m*ωx²/(2*q) * 1e-12  (V/µm²)

    Parameters
    ----------
    freq_MHz : (fx, fy, fz)
        三轴阱频 (MHz)
    mass_amu : float
        离子质量 (amu)
    charge_ec : float
        离子电荷 (单位 e)，默认 1.0
    center_um : tuple
        势场中心 (µm)
    scale_um : float
        缩放半跨度 L (µm)，用于 FitResult3D 坐标变换

    Returns
    -------
    FitResult3D
        代表理想二次势的拟合结果，可直接用于 energy/phonon 模块
    """
    from scipy.constants import e as ECHARGE, physical_constants

    amu_kg, _, _ = physical_constants["atomic mass constant"]

    m_kg = mass_amu * amu_kg
    q_C = charge_ec * ECHARGE

    coeffs = np.zeros((5, 5, 5), dtype=float)
    for axis, freq in enumerate(freq_MHz):
        omega = 2.0 * np.pi * freq * 1e6  # rad/s
        # a = m*ω² / (2*q)  (V/m²) → ×1e-12 → V/µm²
        a_um = m_kg * omega ** 2 / (2.0 * q_C) * 1e-12
        # 缩放坐标系数: c = a * L²
        c = a_um * scale_um ** 2
        exp = [0, 0, 0]
        exp[axis] = 2
        coeffs[exp[0], exp[1], exp[2]] = c

    return FitResult3D(
        coeffs=coeffs,
        center_um=center_um,
        scale_um=scale_um,
        potential_offset_V=0.0,
        r_squared=1.0,
        fit_mode=2,
        symmetry_axes=("x", "y", "z"),
        basis_exps=_QUADRATIC_BASIS_EXPS,
    )


# ---------------------------------------------------------------------------
# 显式系数 → FitResult3D（无需 CSV/拟合，供动力学直接指定多项式势）
# ---------------------------------------------------------------------------

# 单项式因子：x / y / z 或 x^n 形式
_TERM_FACTOR_RE = re.compile(r"^(x|y|z)(?:\^(\d+))?$")


def parse_term_label(label: str) -> tuple[int, int, int]:
    """
    quartic_3d_term_label 的反函数：单项式标签字符串 -> 指数 (i,j,k)。

    支持 "1"（常数）、"x"、"x^2"、"x*y"、"x^2*y^3*z" 等；
    变量出现顺序无关（"y*x" 与 "x*y" 等价），但同一变量重复出现视为非法。
    每变量次数须在 0..4（与 (5,5,5) 系数张量一致，对应高次 quartic 拟合）。

    Raises
    ------
    ValueError
        标签无法解析、变量重复、或某变量次数越界时。
    """
    s = str(label).strip()
    if s in ("", "1"):
        return (0, 0, 0)
    idx = {"x": 0, "y": 1, "z": 2}
    exps = [0, 0, 0]
    seen: set[str] = set()
    for tok in s.split("*"):
        tok = tok.strip()
        if not tok:
            raise ValueError(f"单项式标签 {label!r} 含空因子")
        m = _TERM_FACTOR_RE.match(tok)
        if not m:
            raise ValueError(
                f"无法解析单项式因子 {tok!r}（来自标签 {label!r}）；"
                f"应为 x/y/z 或 x^n 形式（如 x^2、y*z）"
            )
        var, exp_s = m.group(1), m.group(2)
        if var in seen:
            raise ValueError(f"单项式标签 {label!r} 中变量 {var} 重复出现")
        seen.add(var)
        exp = int(exp_s) if exp_s else 1
        if exp < 0 or exp > DEGREE_QUARTIC:
            raise ValueError(
                f"单项式因子 {tok!r} 次数 {exp} 越界，每变量次数须在 0..{DEGREE_QUARTIC}"
            )
        exps[idx[var]] = exp
    return (exps[0], exps[1], exps[2])


def fit_result_from_coeff_map(
    coeff_map: dict[str, float],
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale_um: float = 100.0,
    potential_offset_V: float = 0.0,
) -> FitResult3D:
    """
    由 {项标签: 系数} 直接构造 FitResult3D，无需 CSV 数据或拟合。

    系数定义在缩放坐标 u=(x-x0)/L, v, w 上，量纲为伏特 (V)，与
    fit_potential_3d_quartic 的拟合系数完全同量纲——故可用于把"高次拟合"
    得到的系数直接喂回，或手写任意多项式势。

    Parameters
    ----------
    coeff_map : dict[str, float]
        项标签（如 "x^2"、"x^2*y^3*z"、"1"）-> 系数 (V)。
    center_um : tuple
        势场中心 (x0, y0, z0) (µm)。
    scale_um : float
        缩放半跨度 L (µm)。
    potential_offset_V : float
        势能零点平移 (V)，默认 0。

    Returns
    -------
    FitResult3D
        可直接用于 eval_fit_3d / grad_fit_3d / hessian_fit_3d，或经
        FieldParser.poly_force._make_field_callable 转为动力学力场。
    """
    coeffs = np.zeros((5, 5, 5), dtype=float)
    seen: set[tuple[int, int, int]] = set()
    exps: list[tuple[int, int, int]] = []
    for label, val in coeff_map.items():
        e = parse_term_label(label)
        if e in seen:
            raise ValueError(
                f"项 {label!r} 解析为 (i,j,k)={e}，与已有项重复（同一单项式被多次指定）"
            )
        seen.add(e)
        coeffs[e] = float(val)
        exps.append(e)
    # 规范化基底顺序：按总次数、再字典序
    exps.sort(key=lambda t: (sum(t), t))
    return FitResult3D(
        coeffs=coeffs,
        center_um=tuple(float(c) for c in center_um),
        scale_um=float(scale_um),
        potential_offset_V=float(potential_offset_V),
        r_squared=1.0,
        fit_mode=None,
        basis_exps=tuple(exps),
    )


def load_poly_potential_json(path: Path | str) -> FitResult3D:
    """
    从 JSON 加载显式多项式系数势场。

    文件格式（与 write_potential_fit_coeff_json 导出兼容，可往返）::

        {
          "coefficients": {"x^2": <V>, "x^2*y^3*z": <V>, "1": <V>, ...},
          "center_um": [0.0, 0.0, 0.0],   // 缺省 (0,0,0)
          "scale_um": 100.0,               // 缺省 100
          "potential_offset_V": 0.0        // 缺省 0
        }

    导出文件中的 csv/config/fit_mode 字段（若存在）被忽略。

    若省略 center_um / scale_um，则分别回退到 (0,0,0) / 100.0 并发出 UserWarning：
    对手写的原点中心势这是有意为之，但对从 config 拟合导出的文件而言，
    缺失 scale_um 会使系数无定义（force 幅度错误），故提醒用户显式给出。

    Raises
    ------
    ValueError
        缺少 'coefficients' 字段时。
    FileNotFoundError
        文件不存在时。
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if "coefficients" not in data:
        raise ValueError(
            f"多项式势 JSON {p} 缺少 'coefficients' 字段（项标签 -> 系数 V）"
        )
    if "center_um" not in data or "scale_um" not in data:
        warnings.warn(
            f"多项式势 JSON {p} 未显式给出 center_um/scale_um，"
            f"回退到默认 center_um=(0,0,0)、scale_um=100.0。"
            f"若此文件来自 config 拟合导出（potential_fit_coeff.json），"
            f"缺失 scale_um 会导致力幅度错误——请用最新版 find_equilibrium 重新导出。",
            UserWarning,
            stacklevel=2,
        )
    center_um = tuple(data.get("center_um", (0.0, 0.0, 0.0)))
    scale_um = float(data.get("scale_um", 100.0))
    offset = float(data.get("potential_offset_V", 0.0))
    return fit_result_from_coeff_map(data["coefficients"], center_um, scale_um, offset)


