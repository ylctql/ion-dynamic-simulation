"""
trap_stability — Mathieu 稳定性参数 (a, q) 计算

从实际场几何 (CSV+JSON) 或理想四极阱参数计算 Mathieu a/q 参数、
secural 频率，并判断稳定性区域。

用法:
    python -m trap_stability --csv <csv> --config <json>
    python -m trap_stability --direct --rf-freq 35.28 --r0 700 --V0 275
"""
from .stability import StabilityResult, compute_stability_direct, compute_stability_from_field
