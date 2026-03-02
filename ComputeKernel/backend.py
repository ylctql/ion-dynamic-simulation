"""
Python 后端计算接口
使用 pybind11 封装的 ionsim 模块，实现类似 ism-hybrid/dataplot.py 的 CalculationBackend
参考 outline.md - backend.py
"""
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Callable

import setup_path

# 须在 import ionsim 之前调用，确保 build/ 在 sys.path 中（spawn 子进程时可能未继承）
setup_path.ensure_build_in_path(Path(__file__).resolve().parent.parent)

import numpy as np

from utils import CommandType, Frame, Message

try:
    import ionsim
except ImportError as e:
    raise ImportError(
        "无法导入 ionsim 模块，请先编译 C++ 扩展：\n"
        "  cd ism-main-v1.0 && cmake -B build && cmake --build build"
    ) from e


def get_actual_device(requested: str) -> str:
    """
    返回实际使用的计算设备。
    若请求 cuda 但未编译 CUDA 支持，则回退为 cpu。
    """
    if requested == "cuda" and not getattr(ionsim, "cuda_available", False):
        return "cpu"
    return requested


def _to_ionsim_format(
    r: np.ndarray,
    v: np.ndarray,
    charge: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将数组转换为 ionsim.calculate_trajectory 所需格式
    - r, v: (N, 3) Fortran-contiguous
    - charge, mass: (N, 1) 列向量
    """
    r = np.asarray(r, dtype=np.float64, order="F")
    v = np.asarray(v, dtype=np.float64, order="F")
    charge = np.asarray(charge, dtype=np.float64).reshape(-1, 1)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1, 1)
    return r, v, charge, mass


def _wrap_force(
    force: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    包装 force 回调，确保返回 (N, 3) 且可被 ionsim 使用
    """

    def wrapped(r: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        F = force(r, v, t)
        return np.asarray(F, dtype=np.float64, order="F")

    return wrapped


class CalculationBackend:
    """
    动力学模拟后端
    从 queue_control 接收 Message，调用 ionsim.calculate_trajectory，
    将结果 Frame 写入 queue_data
    """

    def __init__(
        self,
        step: int = 100,
        interval: float = 1.0,
        batch: int = 10,
        time: float = np.inf,
        device: str = "cpu",
        calc_method: str = "VV",
        dt: float = 1.0,
        dl: float = 1.0,
    ):
        """
        Parameters
        ----------
        step : int
            每两帧之间的积分步数
        interval : float
            每两帧之间的时间间隔 (dt 单位)
        batch : int
            每次计算输出的帧数
        time : float
            总模拟时间 (dt 单位)，np.inf 表示无限
        device : str
            "cpu" 或 "cuda"
        calc_method : str
            "RK4" 或 "VV"
        dt : float
            单位时间 (SI)
        dl : float
            单位长度 (SI)
        """
        self.step = step
        self.interval = interval
        self.batch = batch
        self.time = time
        self.device = device
        self.calc_method = calc_method
        self.dt = dt
        self.dl = dl
        self.use_cuda = device == "cuda"

    def run(
        self,
        queue_data: mp.Queue,
        queue_control: mp.Queue,
    ) -> None:
        """
        后端主循环：从 queue_control 接收 Message，计算轨迹，向 queue_data 输出 Frame

        Parameters
        ----------
        queue_data : mp.Queue
            数据通道，输出 Frame
        queue_control : mp.Queue
            控制通道，接收 Message
        """
        # 等待 START 信号
        m: Message = queue_control.get()
        while m.command != CommandType.START:
            m = queue_control.get()

        if m.r is None or m.v is None or m.charge is None or m.mass is None or m.force is None:
            raise RuntimeError("START 消息须包含 r, v, charge, mass, force")

        r0 = m.r
        v0 = m.v
        charge = m.charge
        mass = m.mass
        force = _wrap_force(m.force)
        t: float = m.t0 if m.t0 is not None else 0.0

        paused = False

        while True:
            # 先消费控制消息
            while not queue_control.empty():
                m = queue_control.get()
                if m.command == CommandType.START:
                    pass  # 忽略重复 START
                elif m.command == CommandType.PAUSE:
                    paused = True
                elif m.command == CommandType.RESUME:
                    paused = False
                    if m.r is not None:
                        r0 = m.r
                    if m.v is not None:
                        v0 = m.v
                    if m.t0 is not None:
                        t = m.t0
                    if m.charge is not None:
                        charge = m.charge
                    if m.mass is not None:
                        mass = m.mass
                    if m.force is not None:
                        force = _wrap_force(m.force)
                elif m.command == CommandType.STOP:
                    queue_data.put(False)
                    return

            if paused:
                time.sleep(0.1)
                continue

            # 转换格式并调用 ionsim
            r_in, v_in, q_in, m_in = _to_ionsim_format(r0, v0, charge, mass)

            total_steps = self.step * self.batch
            time_end = t + self.interval * self.batch

            # 排查用：SKIP_FORCE_CALLBACK=1 时跳过 Python force 回调
            skip_force = os.environ.get("SKIP_FORCE_CALLBACK") == "1"

            r_list, v_list = ionsim.calculate_trajectory(
                r_in,
                v_in,
                q_in,
                m_in,
                step=total_steps,
                time_start=t,
                time_end=time_end,
                force=force,
                use_cuda=self.use_cuda,
                calc_method=self.calc_method,
                use_zero_force=skip_force,
            )

            # 输出每 batch 帧
            for i in range(self.batch):
                t += self.interval
                if t > self.time:
                    queue_control.put(Message(CommandType.STOP))
                    queue_data.put(False)
                    return

                idx = (i + 1) * self.step - 1
                r_frame = np.asarray(r_list[idx], dtype=np.float64)
                v_frame = np.asarray(v_list[idx], dtype=np.float64)
                queue_data.put(Frame(r_frame, v_frame, t))

            r0 = np.asarray(r_list[-1], dtype=np.float64)
            v0 = np.asarray(v_list[-1], dtype=np.float64)
