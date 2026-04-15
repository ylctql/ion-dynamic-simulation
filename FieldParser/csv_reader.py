"""
解析 CSV 格式的电场格点数据
参考 data_standard.md，输出 grid_coord 与 grid_voltage 两个 np 数组
"""
from pathlib import Path

import numpy as np
import pandas as pd

# 长度单位换算：CSV 数值 → SI 米；表头未识别时默认按 mm
_UNITS = {"m": 1e0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6}


def _length_unit_label_to_meters(unit_label: str) -> float:
    """
    将 % Length unit 列中的写法转为「CSV 坐标值 × 该因子 = SI 米」。
    兼容 COMSOL 等导出的 µm（U+00B5）、μm（U+03BC）与 ASCII um。
    """
    s = unit_label.strip()
    s = s.replace("\u00b5", "u").replace("\u03bc", "u")
    return _UNITS.get(s.lower(), 1e-3)


def read(
    csv_filename: str | Path,
    field_settings=None,
    *,
    normalize: bool = True,
    dl: float | None = None,
    dV: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    解析 CSV 电场格点数据，仅保留数据矩阵部分

    Parameters
    ----------
    csv_filename : str | Path
        电场格点数据 CSV 文件路径
    field_settings : FieldSettings | None
        若提供则检查 voltage 列数与 voltage_list 数量一致
    normalize : bool
        是否进行无量纲化（坐标 * unit_l/dl，电势 / dV）
    dl, dV : float | None
        无量纲化常数；normalize=True 时必填

    Returns
    -------
    grid_coord : np.ndarray, shape (N, 3)
        格点坐标，每行为 (x, y, z)，已按 lexsort(x,y,z) 排序
    grid_voltage : np.ndarray, shape (N, n_voltage)
        势场值，每行对应 grid_coord 同序格点的各电极电势
    """
    csv_filename = Path(csv_filename)
    if not csv_filename.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_filename}")

    unit_l = 1e-3  # 默认 mm
    with open(csv_filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 20:
                break
            row = line.strip().split(",")
            if row and row[0].strip() == r"% Length unit" and len(row) >= 2:
                unit_l = _length_unit_label_to_meters(row[1])
                break
            if row and row[0].strip() == r"% x":
                break

    dat = pd.read_csv(csv_filename, comment="%", header=None)
    data = dat.to_numpy(dtype=float)

    if data.size == 0:
        raise ValueError(f"CSV 文件无有效数据: {csv_filename}")

    n_cols = data.shape[1]
    if n_cols < 4:
        raise ValueError(f"CSV 至少需 4 列 (x,y,z + 至少 1 个电势)，当前 {n_cols} 列")

    if field_settings is not None:
        n_expected = len(field_settings.voltage_list)
        n_actual = data.shape[1] - 3
        if n_expected > 0 and n_actual != n_expected:
            raise ValueError(
                f"势场列数 {n_actual} 与 voltage_list 数量 {n_expected} 不一致"
            )

    data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))]
    grid_coord = data[:, :3].copy()
    grid_voltage = data[:, 3:].copy()

    if normalize:
        if dl is None or dV is None:
            raise ValueError("normalize=True 时须提供 dl 和 dV")
        grid_coord = grid_coord * (unit_l / dl)
        grid_voltage = grid_voltage / dV

    return grid_coord.astype(float, order="C"), grid_voltage.astype(float, order="C")
