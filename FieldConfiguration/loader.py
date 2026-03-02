"""
从 JSON 文件加载电极电压配置

用户配置的电压为 SI 单位（伏特 V），频率为 SI 单位（MHz）。
电压幅度保持 SI 单位传入 force 计算（电势基已在 csv_reader 中归一化，幅度不再归一化，
否则会导致 E_tot 双重归一化错误）。
需传入 Config 对象以获取 freq_RF 等无量纲化常数。

JSON 格式示例:
{
  "g": 0.1,
  "voltage_list": [
    {"type": "dc", "name": "U1", "V_bias": -0.2},
    {"type": "dc", "name": "U2", "V_bias": -0.3},
    {"type": "rf", "name": "RF", "V0": -8.0, "V_bias": 0.0, "frequency": 35.28}
  ]
}
"""
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .field_settings import FieldSettings, voltage_dc, voltage_rf
from utils import Voltage

if TYPE_CHECKING:
    from .constants import Config


def get_reference_frequency(config: dict[str, Any]) -> float:
    """从配置中提取基准 RF 频率 (MHz)，供 constants.init_from_config 使用"""
    from .constants import _get_ref_freq_from_config

    return _get_ref_freq_from_config(config)


def _freq_mhz_to_norm(freq_mhz: float, freq_RF: float) -> float:
    """将 SI 频率 (MHz) 转为无量纲，供 cos_time 使用"""
    return 2.0 * (freq_mhz / freq_RF)


def _parse_voltage(obj: dict[str, Any], cfg: "Config") -> Voltage:
    """
    从 JSON 对象解析单个 Voltage
    用户输入的 V_bias、V0 为 SI 单位（伏特），frequency 为 SI 单位（MHz）。
    电压幅度保持 SI 单位（不除以 dV），因电势基已归一化，field_interp 输出 (dl/dV)*E_si，
    需 coef=V_si 才能得到正确的 E_norm = V_si*E_si*dl/dV。
    """
    vtype = obj.get("type", "dc")
    name = obj.get("name", "U")
    if vtype == "dc":
        V_bias = float(obj.get("V_bias", 0.0))
        return voltage_dc(name, V_bias)
    elif vtype == "rf":
        V0 = float(obj.get("V0", 0.0))
        V_bias = float(obj.get("V_bias", 0.0))
        freq_mhz = float(obj.get("frequency", cfg.freq_RF))
        freq_norm = _freq_mhz_to_norm(freq_mhz, cfg.freq_RF)
        return voltage_rf(name, V0, V_bias, freq_norm)
    else:
        raise ValueError(f"未知电极类型: {vtype}，支持 dc / rf")


def load_field_config(config_path: Path | str) -> dict[str, Any]:
    """从 JSON 文件加载配置"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_voltage_list(
    config: dict[str, Any],
    n_voltage: int,
    cfg: "Config",
) -> list[Voltage]:
    """
    根据配置构建 voltage_list，长度与 n_voltage 匹配
    若配置条目少于 n_voltage，用 voltage_dc(f"U{i+1}", 0.0) 补齐
    若多于 n_voltage，则截断
    """
    raw = config.get("voltage_list", [])
    voltage_list = [_parse_voltage(v, cfg) for v in raw]
    while len(voltage_list) < n_voltage:
        voltage_list.append(voltage_dc(f"U{len(voltage_list)+1}", 0.0))
    return voltage_list[:n_voltage]


def field_settings_from_config(
    csv_path: str,
    config_path: Path | str,
    n_voltage: int,
    cfg: "Config",
) -> FieldSettings:
    """根据 CSV 路径、配置路径和电极数量构建 FieldSettings"""
    config = load_field_config(config_path)
    voltage_list = build_voltage_list(config, n_voltage, cfg)
    g = float(config.get("g", 0.1))
    return FieldSettings(
        csv_filename=csv_path,
        voltage_list=voltage_list,
        g=g,
    )
