"""DataPlotter σ_y 标题解耦测试。

验证 σ_y 实时数值在任意 --plot-fig 组合下都可见：
- 仅 zox（无 zoy）→ σ_y 落在首个 zox 子图标题（新增能力）
- zoy,zox（默认）→ σ_y 仍在 zoy、zox 不带（向后兼容）
- xoy,zox（无 zoy）→ σ_y 落在首个 xoy 子图标题

纯 Python（不 import ionsim），可在 conda base 跑。
"""
import matplotlib

matplotlib.use("Agg")  # 必须在导入 pyplot / DataPlotter 之前

import multiprocessing as mp

import numpy as np

from Plotter.dataplot import DataPlotter
from utils import Frame


def _make_plotter(plot_fig):
    """构造一个 headless DataPlotter（show_plot=False），r 的 y 方向有非零 spread。"""
    q_data = mp.Queue()
    q_control = mp.Queue()
    n = 4
    r0 = np.zeros((n, 3), dtype=float)
    r0[:, 1] = np.array([-3.0, -1.0, 1.0, 3.0])  # σ_y > 0
    v0 = np.zeros((n, 3), dtype=float)
    frame_init = Frame(r0, v0, 0.0)
    plotter = DataPlotter(
        q_data,
        q_control,
        frame_init,
        plot_fig=plot_fig,
        show_plot=False,
        dl=1e-6,  # → _dl_um = 1.0，r 直接当 µm 处理
        dt=1e-6,
    )
    return plotter, frame_init


def test_sigma_y_on_zox_when_no_zoy():
    """--plot-fig zox：无 zoy 时 σ_y 应显示在唯一的 zox 子图标题。"""
    plotter, frame = _make_plotter(["zox"])
    plotter.plot(frame=frame)
    assert "σ_y" in plotter.ax[0].get_title()


def test_sigma_y_on_zoy_when_present():
    """--plot-fig zoy,zox（默认）：σ_y 仍在 zoy（首图），zox 不带——向后兼容。"""
    plotter, frame = _make_plotter(["zoy", "zox"])
    plotter.plot(frame=frame)
    assert "σ_y" in plotter.ax[0].get_title()
    assert "σ_y" not in plotter.ax[1].get_title()


def test_sigma_y_on_first_axis_when_no_zoy():
    """--plot-fig xoy,zox：无 zoy 时 σ_y 落在第一个子图（xoy）。"""
    plotter, frame = _make_plotter(["xoy", "zox"])
    plotter.plot(frame=frame)
    assert "σ_y" in plotter.ax[0].get_title()
    assert "σ_y" not in plotter.ax[1].get_title()
