"""
物理系统与计算模式参数设置
参考 outline.md 输入模块 - 离子参数、演化信息、计算模式、动力学模拟方法
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Parameters:
    """
    物理系统与计算模式参数
    坐标、速度、时间均采用无量纲单位（dl, dl/dt, dt）
    """

    # ----- 离子参数 -----
    N: int = 50  # 离子数量
    m: np.ndarray | None = None  # 质量 (N,)，相对参考质量，未设置时全为 1
    q: np.ndarray | None = None  # 电荷量 (N,)，单位元电荷，未设置时全为 1

    # ----- 演化信息 -----
    r0: np.ndarray | None = None  # 初始坐标 (N, 3)，无量纲，未设置则随机生成
    v0: np.ndarray | None = None  # 初始速度 (N, 3)，无量纲，未设置则为全 0

    # 随机初始化时的分布范围（当 r0 未设置时使用）
    init_center: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 中心坐标 (x0, y0, z0)
    init_range: float | tuple[float, float, float] = 150.0  # 偏离最大值，标量则三轴同值

    t0: float = 0.0  # 演化起始时间 (dt 单位)
    duration: float = np.inf  # 演化持续时间 (dt 单位)

    # ----- 计算模式 -----
    device: Literal["cpu", "cuda"] = "cpu"

    # ----- 动力学模拟方法 -----
    calc_method: Literal["RK4", "VV"] = "VV"

    # ----- 同位素参杂（可选，用于自动分配质量）-----
    alpha: float = 0.0  # 参杂比例，>0 时按 Ba133/134/135/136/137/138 分配质量
    isotope_type: str | None = None  # 单同位素模式：指定种类时 alpha 为该同位素丰度，其余为 Ba135

    def __post_init__(self) -> None:
        """根据 N 和 alpha 补全 m, q；根据 r0, v0 决定是否随机初始化"""
        self._ensure_arrays()

    def _ensure_arrays(self) -> None:
        """确保 m, q 为长度为 N 的数组"""
        if self.m is None:
            self.m = np.ones(self.N)
        else:
            self.m = np.asarray(self.m, dtype=float)
            if self.m.size != self.N:
                raise ValueError(f"m 长度 {self.m.size} 与 N={self.N} 不一致")

        if self.q is None:
            self.q = np.ones(self.N)
        else:
            self.q = np.asarray(self.q, dtype=float)
            if self.q.size != self.N:
                raise ValueError(f"q 长度 {self.q.size} 与 N={self.N} 不一致")

        if self.alpha > 0 or self.isotope_type is not None:
            self._apply_isotope_doping()

    def _apply_isotope_doping(self) -> None:
        """按 alpha 分配质量：单同位素模式（isotope_type 指定）或混合模式"""
        n = self.N
        a = self.alpha

        if self.isotope_type is not None:
            # 单同位素模式：alpha 为该同位素丰度，其余为 Ba135
            mass_map = {
                "Ba133": 133 / 135,
                "Ba134": 134 / 135,
                "Ba135": 1.0,
                "Ba136": 136 / 135,
                "Ba137": 137 / 135,
                "Ba138": 138 / 135,
            }
            m_doped = mass_map.get(self.isotope_type)
            if m_doped is None:
                raise ValueError(f"未知同位素 {self.isotope_type}，支持: {list(mass_map.keys())}")
            n_doped = int(n * a) if a > 0 else 0
            self.m[:n_doped] = m_doped
            self.m[n_doped:] = 1.0  # Ba135
            return

        # 混合模式：Ba133/134/135/136/137/138 各占 alpha，Ba135 占剩余
        i0, i1, i2, i3, i4, i5 = (
            int(n * a),
            int(2 * n * a),
            int(n * (1 - 3 * a)),  # Ba135 结束
            int(n * (1 - 3 * a) + n * a),
            int(n * (1 - 3 * a) + 2 * n * a),
            n,
        )
        self.m[:i0] = 133 / 135  # Ba133
        self.m[i0:i1] = 134 / 135  # Ba134
        self.m[i1:i2] = 1.0  # Ba135
        self.m[i2:i3] = 136 / 135  # Ba136
        self.m[i3:i4] = 137 / 135  # Ba137
        self.m[i4:i5] = 138 / 135  # Ba138

    def get_r0(self) -> np.ndarray:
        """返回初始坐标，若未设置则按 init_center、init_range 随机生成"""
        if self.r0 is not None:
            return np.asarray(self.r0, dtype=float, order="C")

        cx, cy, cz = self.init_center
        if isinstance(self.init_range, (int, float)):
            rx = ry = rz = float(self.init_range)
        else:
            rx, ry, rz = self.init_range

        r0 = (np.random.rand(self.N, 3) - 0.5) * np.array([rx, ry, rz])
        r0[:, 0] += cx
        r0[:, 1] += cy
        r0[:, 2] += cz
        return r0.astype(float, order="C")

    def get_v0(self) -> np.ndarray:
        """返回初始速度，若未设置则为全 0"""
        if self.v0 is not None:
            return np.asarray(self.v0, dtype=float, order="C")
        return np.zeros((self.N, 3), dtype=float, order="C")

    def duration_in_dt(self) -> float:
        """演化持续时间 (dt 单位)"""
        return self.duration

    def t0_in_dt(self) -> float:
        """演化起始时间 (dt 单位)"""
        return self.t0


def from_argparse(args, dt: float, *, n_ions: int | None = None) -> Parameters:
    """
    从 argparse.Namespace 构建 Parameters
    兼容 ism-hybrid main.py 的常用参数名

    Parameters
    ----------
    args : argparse.Namespace
    dt : float
        无量纲时间单位 (s)，用于将微秒转换为 dt 单位
    n_ions : int | None
        若指定则覆盖 args 中的离子数（多 --N 时由 CLI 逐场传入）
    """
    if n_ions is not None:
        N = n_ions
    else:
        nv = getattr(args, "N", 50)
        if isinstance(nv, list):
            if not nv:
                raise ValueError("args.N 离子数列表为空")
            N = nv[0]
        else:
            N = int(nv)
    alpha = getattr(args, "alpha", getattr(args, "isotope_ratio", 0.0))
    isotope_type = getattr(args, "isotope", None)
    t0 = getattr(args, "t0", 0.0)
    time_end_us = getattr(args, "time", np.inf)
    if time_end_us is None:
        time_end_us = np.inf

    # 将微秒转换为 dt 单位；--time 为模拟终止时刻 (μs)
    t0_dt = t0 / (dt * 1e6)
    if time_end_us == np.inf:
        duration_dt = np.inf
    else:
        if t0 >= time_end_us:
            raise ValueError(
                f"--t0 ({t0} μs) 必须小于 --time ({time_end_us} μs)，"
                "终止时刻不能早于或等于起始时刻"
            )
        time_end_dt = time_end_us / (dt * 1e6)
        duration_dt = time_end_dt - t0_dt

    return Parameters(
        N=N,
        alpha=alpha,
        isotope_type=isotope_type,
        t0=t0_dt,
        duration=duration_dt,
        device="cuda" if getattr(args, "device", "cpu") == "cuda" else "cpu",
        calc_method=getattr(args, "calc_method", "VV"),
    )
