"""从二维格点 CSV 拟合 RF 势的 A、B 系数（Laplace 基线性最小二乘）。

模型：V(x, y) ≈ V₀ + A·(x² − y²) + B·(x⁴ − 6x²y² + y⁴)，
即径向平面 RF 四极 + 十六极（2D Laplace 调和基，见 docs/radial_trap.md §2）。
常数项 V₀ 吸收任意零点偏移（电势只差常数不影响场）。

CSV 格式（三种，自动识别）：
1. Comsol 导出：`%` 前缀元数据行（% Model/% Version/…）+ 每行一个格点；
   `% Length unit,<unit>` 指定坐标单位（m/cm/mm/um/nm，自动换算为 µm），
   `% x,y,<表达式>` 为列头（电势列名任意，如 esbe.V (V)）。数据须恰
   3 列 x,y,V——多表达式导出请只保留一个电势表达式。
2. 简单表头 + 每行一个格点。列名大小写不敏感，x/y 取
   {x, x_um, x(um), x[um]} 之一，电势取 {v, v_rf, phi, phi_rf, potential,
   u, v_volt} 之一；恰 3 列且前两列为 x/y 时电势列名不限。
3. 无表头且恰 3 列数值：按 x, y, V 顺序解析。
单位约定：无 `% Length unit` 行时 x/y 按 µm 解析（Comsol 导出总会带该行）；
V 一律为伏特（与本模块其余部分一致）。
"""
from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from typing import IO, Iterable

import numpy as np

_X_NAMES = {"x", "x_um", "x(um)", "x(µm)", "x[um]"}
_Y_NAMES = {"y", "y_um", "y(um)", "y(µm)", "y[um]"}
_V_NAMES = {"v", "v_rf", "phi", "phi_rf", "potential", "u", "v_volt", "volt"}

# Comsol `% Length unit` 坐标单位 → µm 换算因子（µ U+00B5 / μ U+03BC 归一为 um，
# 与 FieldParser/csv_reader.py 同口径）
_UNIT_TO_UM = {"m": 1e6, "cm": 1e4, "mm": 1e3, "um": 1.0, "nm": 1e-3}


@dataclass(frozen=True)
class FitABResult:
    """A/B 拟合结果与诊断量。"""

    A_V_per_um2: float
    B_V_per_um4: float
    v_offset_V: float
    n_points: int
    rms_residual_V: float
    r_squared: float
    x_range_um: tuple[float, float]
    y_range_um: tuple[float, float]


def _match_column(header: list[str], names: set[str], what: str) -> int:
    low = [h.strip().lower() for h in header]
    for want in names:
        if want in low:
            return low.index(want)
    raise ValueError(f"CSV 表头中找不到 {what} 列（接受 {sorted(names)}），"
                     f"实际表头: {header}")


def _is_meta_row(row: list[str]) -> bool:
    """Comsol `%` 前缀元数据行（% Model/% Length unit/% x,y,… 等）。"""
    return bool(row) and row[0].lstrip().startswith("%")


def _comsol_unit_to_um(meta_rows: list[list[str]]) -> float:
    """从 `% Length unit,<unit>` 行取坐标→µm 换算因子；无该行返回 1.0（µm）。"""
    for row in meta_rows:
        key = row[0].strip().lstrip("%").strip()
        if key.lower() == "length unit":
            label = row[1].strip() if len(row) > 1 else ""
            label = label.replace("µ", "u").replace("μ", "u")
            factor = _UNIT_TO_UM.get(label.lower())
            if factor is None:
                raise ValueError(f"无法识别的坐标单位 {label!r}"
                                 f"（支持 {sorted(_UNIT_TO_UM)}）")
            return factor
    return 1.0


def _comsol_header(meta_rows: list[list[str]]) -> list[str] | None:
    """取 `% x,y,…` 列头行（去 % 前缀后的单元格）；无则 None。仅用于诊断信息。"""
    for row in meta_rows:
        cells = [c.strip() for c in row]
        cells[0] = cells[0].lstrip("%").strip()
        if cells and cells[0].lower() == "x":
            return cells
    return None


def load_radial_grid_csv(path: str | IO[str] | Iterable[str]) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """读取二维格点 CSV → (x_um, y_um, v_V) 三个等长一维数组。

    自动识别 Comsol 导出（`%` 元数据行 + `% Length unit` 单位换算）、
    简单表头与无表头三种格式，见模块 docstring。含 NaN/Inf 的行（Comsol
    在几何奇点/域外格点的典型导出形态）自动剔除并发 RuntimeWarning。
    """
    if hasattr(path, "read"):
        rows = list(csv.reader(path))
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("CSV 为空")

    meta_rows = [r for r in rows if _is_meta_row(r)]
    rows = [r for r in rows if not _is_meta_row(r)]
    unit_to_um = _comsol_unit_to_um(meta_rows)
    comsol_hdr = _comsol_header(meta_rows)
    if not rows:
        raise ValueError("CSV 无数据行（仅含 % 元数据行）")

    def _try_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    if all(_try_float(c) for c in rows[0]):
        if len(rows[0]) != 3:
            hint = ""
            if comsol_hdr is not None and len(comsol_hdr) > 3:
                hint = (f"；检测到 Comsol 列头 {comsol_hdr}"
                        "——请在导出中只保留一个电势表达式")
            raise ValueError(f"无表头 CSV 需恰 3 列 (x, y, V)，"
                             f"实际 {len(rows[0])} 列{hint}")
        ix, iy, iv = 0, 1, 2
        data = rows
    else:
        header = rows[0]
        data = rows[1:]
        ix = _match_column(header, _X_NAMES, "x 坐标")
        iy = _match_column(header, _Y_NAMES, "y 坐标")
        rest = [i for i in range(len(header)) if i not in (ix, iy)]
        try:
            iv = _match_column(header, _V_NAMES, "电势")
        except ValueError:
            # 电势列名不限的回退（Comsol 表达式名任意，如 esbe.V (V)）：
            # 恰剩 1 列 → 即电势；剩多列则无法唯一确定
            if len(rest) == 1:
                iv = rest[0]
            else:
                cands = ", ".join(header[i] for i in rest)
                raise ValueError(
                    f"无法确定电势列（剩余候选: {cands}；接受别名 "
                    f"{sorted(_V_NAMES)}，或仅保留一个电势列）") from None
    try:
        pts = np.array([[float(r[ix]), float(r[iy]), float(r[iv])] for r in data])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"CSV 数据行解析失败: {exc}") from exc
    pts[:, :2] *= unit_to_um
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        n_bad = int(finite.size - finite.sum())
        warnings.warn(
            f"已剔除 {n_bad} 个含 NaN/Inf 的格点行（Comsol 域外点/奇点常见此形态）",
            RuntimeWarning, stacklevel=2)
        pts = pts[finite]
    if pts.shape[0] < 4:
        raise ValueError(f"格点数不足（{pts.shape[0]} < 4，至少需定 3 个系数）")
    return pts[:, 0], pts[:, 1], pts[:, 2]


def fit_rf_ab(x_um: np.ndarray, y_um: np.ndarray, v_V: np.ndarray,
              fit_range_um: tuple[float, float] | None = None) -> FitABResult:
    """对 Laplace 基 [1, x²−y², x⁴−6x²y²+y⁴] 做线性最小二乘。

    x/y/v 同形状即可（meshgrid 二维阵可直接传入，内部展平）。含 NaN/Inf
    时直接报错（单个 NaN 即可使全部系数静默变 NaN）。
    fit_range_um=(rx, ry) 时只拟合 |x|≤rx、|y|≤ry 的格点（以坐标原点为
    中心）——格点覆盖远大于链展宽时，更高阶 Laplace 项（六阶/十二极等）
    会泄漏进 A/B（实测 ±300 µm 全域拟合可把 A 抬高 ~50%）；限制到离子区
    （数十~百余 µm）即可去偏。
    B 的可辨识性取决于格点横向范围：x⁴ 基函数幅度 ∝ x_max⁴，范围过小时
    B 与常数/四极项近乎简并（结果对噪声敏感）。调用方应保证覆盖链的
    预计展宽（经验 ≥ 数十 µm）。
    """
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    v = np.asarray(v_V, dtype=float)
    if not (x.shape == y.shape == v.shape):
        raise ValueError("x/y/v 需为同形状数组")
    x = x.ravel()
    y = y.ravel()
    v = v.ravel()
    if not (np.isfinite(x).all() and np.isfinite(y).all()
            and np.isfinite(v).all()):
        # 单个 NaN 会使 lstsq 全系数变 NaN 且 R² 兜底成 1.0，静默掩盖故障
        raise ValueError("x/y/v 含 NaN/Inf——请先剔除非有限值格点"
                         "（load_radial_grid_csv 会自动剔除 CSV 中的此类行）")
    if fit_range_um is not None:
        rx, ry = fit_range_um
        if not (np.isfinite(rx) and rx > 0 and np.isfinite(ry) and ry > 0):
            raise ValueError(f"fit_range_um 需为正有限值，实际 {fit_range_um}")
        m = (np.abs(x) <= rx) & (np.abs(y) <= ry)
        x, y, v = x[m], y[m], v[m]
        if x.size < 4:
            raise ValueError(f"拟合范围 |x|≤{rx:g}, |y|≤{ry:g} µm 内格点不足"
                             f"（{x.size} < 4，至少需定 3 个系数）")
    q2 = x * x - y * y
    q4 = x ** 4 - 6.0 * x * x * y * y + y ** 4
    basis = np.column_stack([np.ones_like(x), q2, q4])
    coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
    resid = v - basis @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((v - v.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return FitABResult(
        A_V_per_um2=float(coef[1]),
        B_V_per_um4=float(coef[2]),
        v_offset_V=float(coef[0]),
        n_points=int(x.size),
        rms_residual_V=float(np.sqrt(ss_res / x.size)),
        r_squared=float(r2),
        x_range_um=(float(x.min()), float(x.max())),
        y_range_um=(float(y.min()), float(y.max())),
    )
