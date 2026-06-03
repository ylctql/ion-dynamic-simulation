"""势场对称性定量表征：镜面对称、旋转对称、多项式系数奇偶性、Hessian 非对角项

支持分别对 DC 静电势、RF 赝势和总势场进行对称性分析。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import compute_potentials, um_to_norm


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
PotentialType = str  # "dc" | "pseudo" | "total"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MirrorSymmetryResult:
    """镜面对称系数（关于坐标平面通过中心点）"""
    plane: str                    # "yz" | "xz" | "xy"
    potential_type: PotentialType
    coefficient: float            # S ∈ [0, 1], 1 = 完美镜面对称
    max_relative_deviation: float
    mean_relative_deviation: float
    n_sample_points: int


@dataclass(frozen=True)
class RotationalSymmetryResult:
    """旋转对称系数（关于坐标轴通过中心点）"""
    axis: str                     # "x" | "y" | "z"
    potential_type: PotentialType
    coefficient: float            # S ∈ [0, 1]
    plane_pair: tuple[str, str]   # 比较的两个正交平面
    max_relative_deviation: float
    mean_relative_deviation: float


@dataclass(frozen=True)
class ParityTermDiag:
    """单个奇次项的诊断信息"""
    exponents: tuple[int, int, int]  # (i, j, k)
    scaled_coeff: float              # c̃[i,j,k]
    label: str                       # e.g. "u^1 v^0 w^2"


@dataclass(frozen=True)
class PolynomialSymmetryResult:
    """多项式系数奇偶性分析（缩放系数法）"""
    potential_type: PotentialType
    r_squared: float                     # 拟合 R²
    s_parity_yz: float                   # 关于 yz 平面
    s_parity_xz: float                   # 关于 xz 平面
    s_parity_xy: float                   # 关于 xy 平面
    top_odd_terms_yz: tuple[ParityTermDiag, ...]  # yz: i 为奇的最大贡献项
    top_odd_terms_xz: tuple[ParityTermDiag, ...]
    top_odd_terms_xy: tuple[ParityTermDiag, ...]


@dataclass(frozen=True)
class HessianSymmetryResult:
    """Hessian 非对角项分析（中心点）"""
    potential_type: PotentialType
    kappa_xx: float   # V/μm²
    kappa_yy: float
    kappa_zz: float
    kappa_xy: float
    kappa_xz: float
    kappa_yz: float
    offdiag_ratio: float  # max(|off-diag|) / mean(|diag|)


@dataclass(frozen=True)
class PotentialSymmetry:
    """单一势场类型的完整对称性分析"""
    potential_type: PotentialType
    mirror: dict[str, MirrorSymmetryResult]       # key: "yz", "xz", "xy"
    rotational: dict[str, RotationalSymmetryResult]  # key: "x", "y", "z"
    polynomial: PolynomialSymmetryResult | None    # None if fit fails
    hessian: HessianSymmetryResult | None          # None if fit fails


@dataclass(frozen=True)
class SymmetryReport:
    """DC / RF 赝势 / 总势场的完整对称性报告"""
    center_um: tuple[float, float, float]
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    dc: PotentialSymmetry
    pseudo: PotentialSymmetry
    total: PotentialSymmetry


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _safe_coeff(rmse: float, V_range: float, eps: float = 1e-30) -> float:
    """归一化对称系数 S = 1 - rmse / V_range，clamp 到 [0, 1]"""
    return max(0.0, min(1.0, 1.0 - rmse / (abs(V_range) + eps)))


def _filter_valid(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """去除含 NaN/Inf 的点（所有数组按行同步过滤）"""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    if mask.all():
        return arrays
    return tuple(a[mask] for a in arrays)


# ---------------------------------------------------------------------------
# compute_single_potential
# ---------------------------------------------------------------------------
def compute_single_potential(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    r: np.ndarray,
    potential_type: PotentialType,
) -> np.ndarray:
    """评估单一势场分量 (dc/pseudo/total)，返回 (N,) Volts"""
    V_dc, _, V_pseudo, V_total = compute_potentials(
        potential_interps, field_interps, voltage_list, cfg, r
    )
    if potential_type == "dc":
        return V_dc
    if potential_type == "pseudo":
        return V_pseudo
    return V_total


# ---------------------------------------------------------------------------
# 1. 镜面对称
# ---------------------------------------------------------------------------
def compute_mirror_symmetry(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    center_um: tuple[float, float, float],
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    potential_type: PotentialType,
    n_pts_per_axis: int = 10,
) -> dict[str, MirrorSymmetryResult]:
    """计算关于 yz/xz/xy 三个坐标平面的镜面对称系数"""
    dl = cfg.dl
    cx, cy, cz = center_um
    (x0, x1), (y0, y1), (z0, z1) = range_um

    # 构建采样网格（μm）
    xs = np.linspace(x0, x1, n_pts_per_axis)
    ys = np.linspace(y0, y1, n_pts_per_axis)
    zs = np.linspace(z0, z1, n_pts_per_axis)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    r_um = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # 归一化坐标
    r_norm = np.column_stack([
        um_to_norm(r_um[:, 0], dl),
        um_to_norm(r_um[:, 1], dl),
        um_to_norm(r_um[:, 2], dl),
    ])

    V = compute_single_potential(
        potential_interps, field_interps, voltage_list, cfg, r_norm, potential_type
    )

    results: dict[str, MirrorSymmetryResult] = {}
    # mirror_axis: 被翻转的坐标索引，plane_name: 对应的镜面
    for mirror_axis, plane_name in [(0, "yz"), (1, "xz"), (2, "xy")]:
        r_mirror = r_norm.copy()
        # 关于 center 做镜面反射: r_i → 2*center_i - r_i
        center_norm = um_to_norm([cx, cy, cz][mirror_axis], dl)
        r_mirror[:, mirror_axis] = 2.0 * center_norm - r_norm[:, mirror_axis]

        V_m = compute_single_potential(
            potential_interps, field_interps, voltage_list, cfg, r_mirror, potential_type
        )

        V_f, Vm_f = _filter_valid(V, V_m)
        if len(V_f) == 0:
            results[plane_name] = MirrorSymmetryResult(
                plane=plane_name, potential_type=potential_type,
                coefficient=float("nan"), max_relative_deviation=float("nan"),
                mean_relative_deviation=float("nan"), n_sample_points=0,
            )
            continue

        diff = np.abs(V_f - Vm_f)
        V_range = np.ptp(V_f)  # max - min
        rmse = np.sqrt(np.mean((V_f - Vm_f) ** 2))
        S = _safe_coeff(rmse, V_range)

        results[plane_name] = MirrorSymmetryResult(
            plane=plane_name,
            potential_type=potential_type,
            coefficient=S,
            max_relative_deviation=float(np.max(diff) / (abs(V_range) + 1e-30)),
            mean_relative_deviation=float(np.mean(diff) / (abs(V_range) + 1e-30)),
            n_sample_points=len(V_f),
        )

    return results


# ---------------------------------------------------------------------------
# 2. 旋转对称
# ---------------------------------------------------------------------------
def compute_rotational_symmetry(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    center_um: tuple[float, float, float],
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    potential_type: PotentialType,
    n_pts_per_axis: int = 50,
) -> dict[str, RotationalSymmetryResult]:
    """计算关于 x/y/z 轴的旋转对称系数

    对每个轴，比较包含该轴的两个正交平面中的势场分布。
    例如 z 轴：比较 V(s, 0, t)（xz 平面）与 V(0, s, t)（yz 平面）。
    """
    dl = cfg.dl
    cx, cy, cz = center_um
    (x0, x1), (y0, y1), (z0, z1) = range_um

    results: dict[str, RotationalSymmetryResult] = {}

    # 对每个旋转轴，定义两个正交平面
    # axis "x": 平面 xy (vary y, const z=cz) vs xz (vary z, const y=cy)
    #   → r1 = (cx+s, cy+t, cz), r2 = (cx+s, cy, cz+t) — 不对，应该是比较径向对称
    # 正确做法：对于 z 轴旋转对称，检查 V(s, 0, t) == V(0, s, t)
    # 即 xz 平面和 yz 平面内的势场关于径向坐标 s 是否相同

    axis_config = {
        # axis: (free_axis_1_idx, free_axis_2_idx, plane_1_name, plane_2_name)
        # free axes are the two axes perpendicular to the rotation axis
        "z": (0, 1, "xz", "yz"),  # free: x, y; const: z
        "y": (0, 2, "xy", "yz"),  # free: x, z; const: y
        "x": (1, 2, "xy", "xz"),  # free: y, z; const: x
    }

    for axis, (a1_idx, a2_idx, p1_name, p2_name) in axis_config.items():
        # 两个自由轴的范围
        range_map = {0: (x0, x1), 1: (y0, y1), 2: (z0, z1)}
        center_map = {0: cx, 1: cy, 2: cz}

        s = np.linspace(range_map[a1_idx][0], range_map[a1_idx][1], n_pts_per_axis)
        t = np.linspace(range_map[a2_idx][0], range_map[a2_idx][1], n_pts_per_axis)
        ss, tt = np.meshgrid(s, t, indexing="ij")

        # 平面 1: 在 (a1_idx, a2_idx) 平面内，a2_idx = center
        r1_um = np.full((ss.size, 3), [cx, cy, cz])
        r1_um[:, a1_idx] = ss.ravel()
        r1_um[:, a2_idx] = tt.ravel()  # 第二个自由轴正常变化
        # 固定轴（旋转轴）= center
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        r1_um[:, axis_idx] = center_map[axis_idx]

        # 平面 2: 将 a1_idx 的坐标映射到 a2_idx，检查旋转对称
        # 例如 z 轴: r1 = (s, center_y, t), r2 = (center_x, s, t)
        r2_um = np.full((ss.size, 3), [cx, cy, cz])
        r2_um[:, a2_idx] = ss.ravel()  # a1 的值映射到 a2 位置
        r2_um[:, a1_idx] = tt.ravel()  # a2 的值保持（但交换了自由度）

        # 不对，让我重新想一下。
        # 对于 z 轴旋转对称：比较 V(x, y_c, z) 与 V(x_c, y, z)
        # 其中 x 在 range 内扫描，y = y_c（平面1: xz）
        #   和 y 在 range 内扫描，x = x_c（平面2: yz）
        # 但这两个 profile 不一定有相同的 range...
        #
        # 更好的做法：比较 V(s, c_y, t) vs V(c_x, s, t) 其中 s 和 t 范围相同
        # 即用相同的 s 值替换 a1 和 a2 坐标

        r1_um = np.full((ss.size, 3), [cx, cy, cz])
        r1_um[:, a1_idx] = ss.ravel()
        # 第二个自由维度用 t，保持 axis 为 center
        r1_um[:, a2_idx] = tt.ravel()

        r2_um = np.full((ss.size, 3), [cx, cy, cz])
        # 交换 a1 和 a2：s → a2 位置，t → a1 位置
        r2_um[:, a2_idx] = ss.ravel()
        r2_um[:, a1_idx] = tt.ravel()

        r1_norm = np.column_stack([
            um_to_norm(r1_um[:, 0], dl),
            um_to_norm(r1_um[:, 1], dl),
            um_to_norm(r1_um[:, 2], dl),
        ])
        r2_norm = np.column_stack([
            um_to_norm(r2_um[:, 0], dl),
            um_to_norm(r2_um[:, 1], dl),
            um_to_norm(r2_um[:, 2], dl),
        ])

        V1 = compute_single_potential(
            potential_interps, field_interps, voltage_list, cfg, r1_norm, potential_type
        )
        V2 = compute_single_potential(
            potential_interps, field_interps, voltage_list, cfg, r2_norm, potential_type
        )

        V1f, V2f = _filter_valid(V1, V2)
        if len(V1f) == 0:
            results[axis] = RotationalSymmetryResult(
                axis=axis, potential_type=potential_type,
                coefficient=float("nan"), plane_pair=(p1_name, p2_name),
                max_relative_deviation=float("nan"),
                mean_relative_deviation=float("nan"),
            )
            continue

        diff = np.abs(V1f - V2f)
        V_range = np.ptp(np.concatenate([V1f, V2f]))
        rmse = np.sqrt(np.mean((V1f - V2f) ** 2))
        S = _safe_coeff(rmse, V_range)

        results[axis] = RotationalSymmetryResult(
            axis=axis,
            potential_type=potential_type,
            coefficient=S,
            plane_pair=(p1_name, p2_name),
            max_relative_deviation=float(np.max(diff) / (abs(V_range) + 1e-30)),
            mean_relative_deviation=float(np.mean(diff) / (abs(V_range) + 1e-30)),
        )

    return results


# ---------------------------------------------------------------------------
# 3. 多项式系数奇偶性分析（缩放系数法）
# ---------------------------------------------------------------------------
def _monomial_rms_scale(i: int, j: int, k: int) -> float:
    """单项式 u^i v^j w^k 在 [-1,1]³ 上的 RMS 值

    ∫_{-1}^{1} u^{2i} du = 2/(2i+1)，故 RMS = √(8 / ((2i+1)(2j+1)(2k+1)))
    """
    return np.sqrt(8.0 / ((2 * i + 1) * (2 * j + 1) * (2 * k + 1)))


def _parity_coefficient(coeffs: np.ndarray, axis: int) -> float:
    """计算关于指定坐标平面的奇偶性对称系数

    axis=0 → yz 平面（检查 i 为奇的项）
    axis=1 → xz 平面（检查 j 为奇的项）
    axis=2 → xy 平面（检查 k 为奇的项）

    使用缩放系数 c̃ = c * RMS(u^i v^j w^k)
    S_parity = 1 - ||c̃_odd||₂ / ||c̃||₂
    """
    # 构建缩放系数数组
    scaled = np.zeros_like(coeffs)
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            for k in range(coeffs.shape[2]):
                if coeffs[i, j, k] != 0.0:
                    scaled[i, j, k] = coeffs[i, j, k] * _monomial_rms_scale(i, j, k)

    total_norm = np.linalg.norm(scaled)
    if total_norm < 1e-30:
        return 1.0  # 全零 → 任意定义为一阶对称

    # 提取 axis 方向奇次项
    odd_mask = np.zeros_like(coeffs, dtype=bool)
    if axis == 0:
        for i in range(coeffs.shape[0]):
            if i % 2 == 1:
                odd_mask[i, :, :] = True
    elif axis == 1:
        for j in range(coeffs.shape[1]):
            if j % 2 == 1:
                odd_mask[:, j, :] = True
    else:
        for k in range(coeffs.shape[2]):
            if k % 2 == 1:
                odd_mask[:, :, k] = True

    odd_norm = np.linalg.norm(scaled[odd_mask])
    return float(max(0.0, 1.0 - odd_norm / total_norm))


def _top_odd_terms(
    coeffs: np.ndarray, axis: int, n_top: int = 5
) -> tuple[ParityTermDiag, ...]:
    """获取指定 axis 方向奇次项中缩放系数最大的 n_top 项（诊断用）"""
    terms: list[tuple[float, tuple[int, int, int]]] = []
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            for k in range(coeffs.shape[2]):
                # 只取对应 axis 为奇的项
                if axis == 0 and i % 2 == 0:
                    continue
                if axis == 1 and j % 2 == 0:
                    continue
                if axis == 2 and k % 2 == 0:
                    continue
                if coeffs[i, j, k] == 0.0:
                    continue
                sc = abs(coeffs[i, j, k] * _monomial_rms_scale(i, j, k))
                terms.append((sc, (i, j, k)))

    terms.sort(key=lambda x: x[0], reverse=True)

    # 构建标签
    var_names = ["u", "v", "w"]
    diags = []
    for sc, (i, j, k) in terms[:n_top]:
        parts = []
        for var, exp in zip(var_names, [i, j, k]):
            if exp == 1:
                parts.append(var)
            elif exp > 1:
                parts.append(f"{var}^{exp}")
        label = " ".join(parts) if parts else "1"
        diags.append(ParityTermDiag(
            exponents=(i, j, k),
            scaled_coeff=float(coeffs[i, j, k] * _monomial_rms_scale(i, j, k)),
            label=label,
        ))
    return tuple(diags)


def compute_polynomial_symmetry(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    center_um: tuple[float, float, float],
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    potential_type: PotentialType,
    n_pts_per_axis: int = 8,
    fit_mode: str = "quartic",
) -> PolynomialSymmetryResult | None:
    """多项式系数奇偶性分析"""
    from equilibrium.potential_fit_3d import fit_potential_3d_quartic

    dl = cfg.dl

    def _compute_V(r_norm: np.ndarray) -> np.ndarray:
        return compute_single_potential(
            potential_interps, field_interps, voltage_list, cfg, r_norm, potential_type
        )

    try:
        fit = fit_potential_3d_quartic(
            compute_V_total=_compute_V,
            um_to_norm=lambda v: um_to_norm(v, dl),
            center_um=center_um,
            range_um=range_um,
            n_pts_per_axis=n_pts_per_axis,
            fit_mode=fit_mode,
        )
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None

    coeffs = fit.coeffs  # shape (5, 5, 5)

    return PolynomialSymmetryResult(
        potential_type=potential_type,
        r_squared=fit.r_squared,
        s_parity_yz=_parity_coefficient(coeffs, 0),
        s_parity_xz=_parity_coefficient(coeffs, 1),
        s_parity_xy=_parity_coefficient(coeffs, 2),
        top_odd_terms_yz=_top_odd_terms(coeffs, 0),
        top_odd_terms_xz=_top_odd_terms(coeffs, 1),
        top_odd_terms_xy=_top_odd_terms(coeffs, 2),
    )


# ---------------------------------------------------------------------------
# 4. Hessian 非对角项分析
# ---------------------------------------------------------------------------
def compute_hessian_symmetry(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    center_um: tuple[float, float, float],
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    potential_type: PotentialType,
    n_pts_per_axis: int = 8,
    fit_mode: str = "quartic",
) -> HessianSymmetryResult | None:
    """Hessian 非对角项分析（中心点处）"""
    from equilibrium.potential_fit_3d import fit_potential_3d_quartic, hessian_fit_3d

    dl = cfg.dl

    def _compute_V(r_norm: np.ndarray) -> np.ndarray:
        return compute_single_potential(
            potential_interps, field_interps, voltage_list, cfg, r_norm, potential_type
        )

    try:
        fit = fit_potential_3d_quartic(
            compute_V_total=_compute_V,
            um_to_norm=lambda v: um_to_norm(v, dl),
            center_um=center_um,
            range_um=range_um,
            n_pts_per_axis=n_pts_per_axis,
            fit_mode=fit_mode,
        )
        center_arr = np.array([center_um])
        hess = hessian_fit_3d(fit, center_arr)[0]  # (3, 3)
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None

    diag = np.array([hess[0, 0], hess[1, 1], hess[2, 2]])
    offdiag = np.array([hess[0, 1], hess[0, 2], hess[1, 2]])
    mean_diag = np.mean(np.abs(diag))
    offdiag_ratio = float(np.max(np.abs(offdiag)) / (mean_diag + 1e-30))

    return HessianSymmetryResult(
        potential_type=potential_type,
        kappa_xx=float(diag[0]),
        kappa_yy=float(diag[1]),
        kappa_zz=float(diag[2]),
        kappa_xy=float(offdiag[0]),
        kappa_xz=float(offdiag[1]),
        kappa_yz=float(offdiag[2]),
        offdiag_ratio=offdiag_ratio,
    )


# ---------------------------------------------------------------------------
# 5. 聚合函数
# ---------------------------------------------------------------------------
_POTENTIAL_TYPES: tuple[PotentialType, ...] = ("dc", "pseudo", "total")


def compute_potential_symmetry(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    center_um: tuple[float, float, float],
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    potential_type: PotentialType,
    *,
    n_mirror_pts: int = 10,
    n_rot_pts: int = 50,
    n_fit_pts: int = 8,
    fit_mode: str = "quartic",
) -> PotentialSymmetry:
    """单一势场类型的完整对称性分析"""
    common = dict(
        potential_interps=potential_interps,
        field_interps=field_interps,
        voltage_list=voltage_list,
        cfg=cfg,
        center_um=center_um,
        range_um=range_um,
        potential_type=potential_type,
    )

    mirror = compute_mirror_symmetry(**common, n_pts_per_axis=n_mirror_pts)
    rotational = compute_rotational_symmetry(**common, n_pts_per_axis=n_rot_pts)
    poly = compute_polynomial_symmetry(**common, n_pts_per_axis=n_fit_pts, fit_mode=fit_mode)
    hess = compute_hessian_symmetry(**common, n_pts_per_axis=n_fit_pts, fit_mode=fit_mode)

    return PotentialSymmetry(
        potential_type=potential_type,
        mirror=mirror,
        rotational=rotational,
        polynomial=poly,
        hessian=hess,
    )


def compute_symmetry_report(
    potential_interps: list,
    field_interps: list,
    voltage_list: list,
    cfg,
    center_um: tuple[float, float, float],
    range_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    n_mirror_pts: int = 10,
    n_rot_pts: int = 50,
    n_fit_pts: int = 8,
    fit_mode: str = "quartic",
) -> SymmetryReport:
    """DC / RF 赝势 / 总势场的完整对称性报告"""
    common = dict(
        potential_interps=potential_interps,
        field_interps=field_interps,
        voltage_list=voltage_list,
        cfg=cfg,
        center_um=center_um,
        range_um=range_um,
        n_mirror_pts=n_mirror_pts,
        n_rot_pts=n_rot_pts,
        n_fit_pts=n_fit_pts,
        fit_mode=fit_mode,
    )
    return SymmetryReport(
        center_um=center_um,
        range_um=range_um,
        dc=compute_potential_symmetry(potential_type="dc", **common),
        pseudo=compute_potential_symmetry(potential_type="pseudo", **common),
        total=compute_potential_symmetry(potential_type="total", **common),
    )
