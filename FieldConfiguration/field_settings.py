"""
电势场与耗散参数设置
参考 outline.md - 电势场参数、耗散参数
CSV 经 FieldParser/csv_reader 解析后得到 np.ndarray 势场数据基，电压组合直接使用 List[Voltage]
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from utils import Voltage, cos_time, constant


@dataclass
class FieldSettings:
    """
    电势场与耗散参数
    供 FieldParser 模块使用
    势场数据基由 csv_reader 从 CSV 解析得到 np.ndarray，与 voltage_list 顺序一一对应
    """

    # ----- 电势场参数 -----
    csv_filename: str = ""  # 电场格点数据 CSV 文件路径，经 csv_reader 解析后得到 np.ndarray 格式的势场数据基
    voltage_list: list[Voltage] = field(default_factory=list)  # 电压组合，与数据基顺序一致

    # ----- 耗散参数 -----
    dissipation_mode: Literal["scalar", "vector"] = "scalar"
    g: float | np.ndarray = 0.1  # 相对耗散强度，scalar 时为 float，vector 时为 (3,) 数组

    def __post_init__(self) -> None:
        if self.dissipation_mode == "scalar":
            self.g = float(self.g)
        else:
            self.g = np.asarray(self.g, dtype=float)
            if self.g.shape != (3,):
                raise ValueError("dissipation_mode='vector' 时 g 须为 (3,) 数组")

    def get_gamma(self) -> float | np.ndarray:
        """返回耗散强度，供 force 计算使用"""
        return self.g

    def get_voltage_names(self) -> list[str]:
        """返回 voltage_list 中的名称列表"""
        return [v.name for v in self.voltage_list]


def voltage_dc(name: str, V_bias: float) -> Voltage:
    """创建纯直流电极 Voltage"""
    return Voltage(name=name, V0=0.0, f=constant(1.0), V_bias=V_bias)


def voltage_rf(name: str, V0: float, V_bias: float = 0.0, frequency: float = 2.0) -> Voltage:
    """创建 RF 电极 Voltage"""
    return Voltage(name=name, V0=V0, f=cos_time(frequency), V_bias=V_bias)
