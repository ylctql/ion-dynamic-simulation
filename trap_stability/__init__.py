"""
trap_stability — Mathieu 稳定性参数 (a, q) 及非谐常数计算

从实际场几何 (CSV+JSON) 计算 Mathieu a/q 参数、secular 频率、
非谐常数（4阶/6阶无量纲 Taylor 系数），并判断稳定性区域。

用法:
    python -m trap_stability --csv <csv> --config <json>
"""
from .stability import StabilityResult, compute_stability_from_field
