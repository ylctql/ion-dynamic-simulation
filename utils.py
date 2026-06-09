"""
类型预定义
参考 outline.md 与 ism-hybrid/utils.py
无量纲化常数由 Config 对象提供，在需要处显式传入。
"""
import logging
import os
from pathlib import Path

import numpy as np
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

# ============== 通信类型 (参照 ism-hybrid) ==============
class CommandType(Enum):
    START = 0
    PAUSE = 1
    RESUME = 2
    STOP = 3


class Frame:
    """单帧动力学数据"""

    def __init__(self, r: np.ndarray, v: np.ndarray, t: float):
        self.r = r  # 位置 (N, 3)
        self.v = v  # 速度 (N, 3)
        self.timestamp = t  # 时间戳 (dt 单位)



class Message:
    """控制通道消息"""

    def __init__(
        self,
        command: CommandType,
        r: np.ndarray | None = None,
        v: np.ndarray | None = None,
        t0: float | None = None,
        charge: np.ndarray | None = None,
        mass: np.ndarray | None = None,
        force: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None = None,
    ):
        self.command = command
        self.r = r
        self.v = v
        self.t0 = t0
        self.charge = charge
        self.mass = mass
        self.force = force


# ============== 帧数据保存 ==============
def save_frame_rv_npz(
    f: Frame, path: str | Path, dl: float, dt: float
) -> None:
    """
    保存 Frame 的 r(μm)、v(m/s)、t_us(μs) 到 NPZ 文件。

    统一替代 main.py / dataplot.py 中分散的 r/v 保存逻辑，
    避免单位转换代码重复。

    Parameters
    ----------
    f : Frame
        单帧数据（无量纲坐标）
    path : str | Path
        输出 NPZ 文件完整路径（含文件名），自动创建父目录
    dl : float
        无量纲长度单位 (SI, m)
    dt : float
        无量纲时间单位 (SI, s)
    """
    r_um = np.asarray(f.r, dtype=np.float64) * dl * 1e6
    v_m_s = np.asarray(f.v, dtype=np.float64) * dl / dt
    time_us = f.timestamp * dt * 1e6
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, r=r_um, v=v_m_s, t_us=time_us)
    logger.info("已保存 r/v: %s", path)


# ============== 电势类 Voltage ==============
# 写成 V0(r)f(t)+V_bias(r) 的形式，便于对幅度部分插值，避免引入多元函数 g(r,t)
class Voltage:
    """
    抽象电势分量
    对应形式: V0(r) * f(t) + V_bias(r)
    其中 V0、V_bias 由空间插值得到，f(t) 为时间函数
    """

    def __init__(
        self,
        name: str,
        V0: float,
        f: Callable[[float], float],
        V_bias: float = 0.0,
    ):
        self.name = name
        self.V0 = V0  # 含时分量幅度
        self.f = f  # 时间函数 f(t)
        self.V_bias = V_bias  # 直流偏置


# ============== 常用时间函数实现 ==============
def cos_time(frequency: float = 2.0) -> Callable[[float], float]:
    """cos(omega * t)，默认 frequency=2 对应 RF 一个周期"""

    def f(t: float) -> float:
        return float(np.cos(frequency * t))

    return f


def sin_time(frequency: float = 2.0) -> Callable[[float], float]:
    """sin(omega * t)"""

    def f(t: float) -> float:
        return float(np.sin(frequency * t))

    return f


def exp_decay(tau: float = 50.0) -> Callable[[float], float]:
    """指数衰减 exp(-t/tau)"""

    def f(t: float) -> float:
        return float(np.exp(-t / tau))

    return f


def exp_ramp(tau: float = 50.0) -> Callable[[float], float]:
    """指数上升 1 - exp(-t/tau)"""

    def f(t: float) -> float:
        return float(1.0 - np.exp(-t / tau))

    return f


def constant(value: float = 1.0) -> Callable[[float], float]:
    """常数"""

    def f(t: float) -> float:
        return value

    return f


def cut_off(cutoff_time: float = 1.0, *, dt_ref: float) -> Callable[[float], float]:
    """
    时间截断：t*dt_ref*1e6 < cutoff_time 时返回 1，否则返回 0

    Parameters
    ----------
    cutoff_time : float
        截断时间 (μs)
    dt_ref : float
        无量纲时间单位 (s)，用于将 t 转换为微秒
    """

    def f(t: float) -> float:
        t_us = t * dt_ref * 1e6
        return 1.0 if t_us < cutoff_time else 0.0

    return f
