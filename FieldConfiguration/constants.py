"""
无量纲化常数 dt, dl, dV
由 FieldConfiguration 中的 RF 频率决定。
init_from_config 返回 Config 对象，在需要处显式传入，避免全局可变状态。
"""
from dataclasses import dataclass

from scipy.constants import e, pi, epsilon_0, u as AMU

# 物理常数（固定）
BA135_MASS_AMU = 134.905683  # Ba-135 原子质量 (NIST AME2020), amu
m = BA135_MASS_AMU * AMU  # Ba135 离子质量 @ SI (kg)
ec = e  # 元电荷 @ SI (C)
epsl = epsilon_0  # 真空介电常数 @ SI

freq_RF_default = 35.28  # 无 RF 配置时的默认基准频率 @ MHz


def _compute_constants(
    freq_MHz: float,
    mass_amu: float = BA135_MASS_AMU,
) -> tuple[float, float, float, float]:
    """根据基准频率 (MHz) 和离子质量 (amu) 计算 Omega, dt, dl, dV"""
    m_kg = mass_amu * AMU
    Omega = freq_MHz * 2 * pi * 1e6  # rad/s
    dt_val = 2 / Omega  # 单位时间，满足 Nyquist 采样
    dl_val = (ec**2 / (4 * pi * m_kg * epsl * Omega**2)) ** (1 / 3)  # 单位长度
    dV_val = m_kg / ec * (dl_val / dt_val) ** 2  # 单位电压
    return Omega, dt_val, dl_val, dV_val


def _get_ref_freq_from_config(config: dict) -> float:
    """从配置字典提取基准 RF 频率 (MHz)"""
    raw = config.get("voltage_list", [])
    rf_freqs = [
        float(v["frequency"])
        for v in raw
        if v.get("type") == "rf" and "frequency" in v
    ]
    return max(rf_freqs) if rf_freqs else freq_RF_default


@dataclass(frozen=True)
class Config:
    """无量纲化常数配置，由 init_from_config 返回"""

    Omega: float
    dt: float
    dl: float
    dV: float
    freq_RF: float


def init_from_config(
    config_path: str,
    mass_amu: float | None = None,
) -> tuple[Config, dict | None]:
    """
    加载 JSON 配置，根据 RF 频率和离子质量计算 dt, dl, dV，返回 Config 对象。

    Parameters
    ----------
    config_path : str
        电压配置 JSON 路径
    mass_amu : float or None
        离子质量 (amu)。None 时使用 Ba135 默认值。

    Returns
    -------
    Config
        无量纲化常数
    dict | None
        配置字典；若文件不存在则使用默认频率并返回 None
    """
    import json
    from pathlib import Path

    _mass = mass_amu if mass_amu is not None else BA135_MASS_AMU

    path = Path(config_path)
    if not path.exists():
        Omega, dt, dl, dV = _compute_constants(freq_RF_default, mass_amu=_mass)
        return (
            Config(Omega=Omega, dt=dt, dl=dl, dV=dV, freq_RF=freq_RF_default),
            None,
        )

    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    ref_freq = _get_ref_freq_from_config(config)
    Omega, dt, dl, dV = _compute_constants(ref_freq, mass_amu=_mass)
    return Config(Omega=Omega, dt=dt, dl=dl, dV=dV, freq_RF=ref_freq), config
