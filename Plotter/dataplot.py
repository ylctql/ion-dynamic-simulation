"""
DataPlotter：实时绘图类
参考 outline.md 与 ism-hybrid/dataplot.py
从 queue_data 接收 Frame，使用 BlitManager 与 color 模块实现高效动画
"""
import logging
import math
import multiprocessing as mp
import os
import queue
import time
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from utils import CommandType, Frame, Message

logger = logging.getLogger(__name__)

# 心跳超时（秒）：用于检测 backend 子进程异常，不影响模拟速度
_QUEUE_GET_TIMEOUT = 1.0

from Plotter.blit import BlitManager
from Plotter.color import (
    ISOTOPE_COLORS,
    ISOTOPE_LABELS,
    get_colors,
    get_colors_layerwise_y_pos,
    get_mass_indices,
)

PlotFig = Literal["xoy", "zoy", "zox"]
ColorMode = Literal["y_pos", "v2", "isotope"] | None


class DataPlotter:
    """
    实时绘图器
    从 queue_data 接收 Frame，更新 scatter 并 blit 显示
    """

    def __init__(
        self,
        queue_data: mp.Queue,
        queue_control: mp.Queue,
        frame_init: Frame,
        *,
        interval: float = 0.04,
        plot_fig: list[PlotFig] | None = None,
        bilayer: bool = False,
        color_mode: ColorMode = None,
        ion_size: float = 5.0,
        x_range: float = 100.0,
        y_range: float = 20.0,
        z_range: float = 200.0,
        x0_plot: float = 0.0,
        y0_plot: float = 0.0,
        z0_plot: float = 0.0,
        dl: float = 1.0,
        dt: float = 1.0,
        mass: np.ndarray | None = None,
        save_times_us: list[float] | None = None,
        save_fig_dir: str = "saves/images/traj",
        save_rv_traj_dir: str | None = None,
        save_rv_status_dir: str | None = None,
        save_final_image: str | None = None,
        target_time_dt: float | None = None,
        show_plot: bool = True,
        device: str = "cpu",
        backend_interval: float = 1.0,
    ):
        """
        Parameters
        ----------
        queue_data : mp.Queue
            数据通道，接收 Frame
        queue_control : mp.Queue
            控制通道，可向 Backend 发送 Message
        frame_init : Frame
            初始帧，用于初始化坐标范围与 scatter
        interval : float
            绘图刷新间隔（秒）
        plot_fig : list[str] | None
            子图视角，如 ["zoy", "zox"] 表示两子图；None 则默认 ["zoy", "zox"]
        bilayer : bool
            True 时使用两片 z-x 投影，分别显示索引 0:N/2 与 N/2:N
        color_mode : None | "y_pos" | "v2" | "isotope"
            离子颜色模式
        ion_size : float
            散点大小
        x_range, y_range, z_range : float
            显示范围半宽（um，已乘 dl）
        x0_plot, y0_plot, z0_plot : float
            显示中心（um）
        dl, dt : float
            单位长度、单位时间（SI）
        mass : np.ndarray
            质量，isotope 模式需要
        save_times_us : list[float] | None
            需保存的时刻（μs）；None 不保存；[] 仅保存最后一帧
        save_fig_dir : str
            save_times_us 保存的根目录，结构为 {save_fig_dir}/{device}/{离子数}/t{时间}us.png
        save_rv_traj_dir : str | None
            指定时刻 r/v 保存根目录
        save_rv_status_dir : str | None
            最后一帧 r/v 保存根目录
        save_final_image : str | None
            最后一帧保存路径
        target_time_dt : float | None
            目标演化时间（dt 单位），达到后发送 STOP
        show_plot : bool
            是否弹出窗口实时显示
        backend_interval : float
            与 ComputeKernel 一致：相邻输出帧在无量纲时间上的间隔（--interval，单位 dt）
        """
        self.queue_data = queue_data
        self.queue_control = queue_control
        self.frame_init = frame_init
        self.interval = interval
        self.bilayer = bilayer
        if self.bilayer:
            n_ions = len(frame_init.r)
            self._bilayer_split = n_ions // 2
            if self._bilayer_split * 2 != n_ions:
                raise ValueError("bilayer 绘图需要离子数 N 为偶数")
            self.plot_fig = ["zox", "zox"]
        else:
            self._bilayer_split = None
            self.plot_fig = plot_fig or ["zoy", "zox"]
        self.color_mode = color_mode
        self.ion_size = ion_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.x0_plot = x0_plot
        self.y0_plot = y0_plot
        self.z0_plot = z0_plot
        self.dl = dl
        self.dt = dt
        self.mass = mass
        self.save_times_us = save_times_us
        self.save_fig_dir = save_fig_dir
        self.save_rv_traj_dir = save_rv_traj_dir
        self.save_rv_status_dir = save_rv_status_dir
        self.device = device
        self.save_final_image = save_final_image
        self.target_time_dt = target_time_dt
        self.show_plot = show_plot
        self.final_frame_saved = False
        self.last_output_time_us = -10.0

        # 无量纲坐标转 um
        self._dl_um = dl * 1e6
        self._dt_us = dt * 1e6

        self.backend_interval = float(backend_interval)
        self._output_stride_us = self.backend_interval * self._dt_us
        self._save_times_fulfilled: set[int] = set()
        if save_times_us and len(save_times_us) >= 2:
            st = sorted(float(x) for x in save_times_us)
            self._min_save_gap_us = min(
                st[i + 1] - st[i] for i in range(len(st) - 1)
            )
            if self._min_save_gap_us <= 0:
                self._min_save_gap_us = math.inf
        else:
            self._min_save_gap_us = math.inf
        half_stride = 0.5 * self._output_stride_us
        if math.isfinite(self._min_save_gap_us):
            self._save_capture_tol_us = half_stride + 0.45 * self._min_save_gap_us
        else:
            self._save_capture_tol_us = max(half_stride, 1e-6)
        if (
            save_times_us
            and len(save_times_us) >= 2
            and math.isfinite(self._min_save_gap_us)
            and self._output_stride_us > self._min_save_gap_us * 1.01
        ):
            logger.warning(
                "输出帧间隔 (%.4g μs) 大于 --save-times-us 最小间隔 (%.4g μs)；"
                "部分目标时刻可能没有足够近的输出帧，请减小 --interval（dt 单位）使 "
                "interval×dt×1e6 不大于保存步长，或接受最近帧快照。",
                self._output_stride_us,
                self._min_save_gap_us,
            )

        # 质量 -> 同位素索引（用于图例）
        self.mass_indices = get_mass_indices(mass) if mass is not None else None

        # 创建子图
        n_axes = len(self.plot_fig)
        if n_axes == 1:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 5))
            self.ax = [self.ax]
        else:
            self.fig, self.ax = plt.subplots(n_axes, 1, figsize=(10, 5 * n_axes))

        colors_init = self._compute_colors(frame_init.r, frame_init.v)

        self.artists = []
        if self.bilayer:
            split = self._bilayer_split
            assert split is not None
            r_um_init = frame_init.r * self._dl_um
            xy0 = np.column_stack((r_um_init[:split, 2], r_um_init[:split, 0]))
            xy1 = np.column_stack((r_um_init[split:, 2], r_um_init[split:, 0]))
            ax0, ax1 = self.ax[0], self.ax[1]
            sc0 = ax0.scatter(
                xy0[:, 0], xy0[:, 1], s=ion_size, c=colors_init[:split]
            )
            sc1 = ax1.scatter(
                xy1[:, 0], xy1[:, 1], s=ion_size, c=colors_init[split:]
            )
            self.artists = [sc0, sc1]
            for ax in (ax0, ax1):
                self._set_axis_limits(ax, "zox")
                self._set_axis_labels(ax, "zox")
            if mass is not None and self.mass_indices is not None:
                self._add_isotope_legend(ax0)
            t0_us = frame_init.timestamp * self._dt_us
            n_ions = r_um_init.shape[0]
            title0, title1 = self._bilayer_titles_for_r_um(
                t0_us, split, n_ions, r_um_init
            )
            ax0.set_title(title0, fontsize=14)
            ax1.set_title(title1, fontsize=14)
        else:
            for i, view in enumerate(self.plot_fig):
                ax = self.ax[i]
                xy = self._get_xy(frame_init.r, view)
                sc = ax.scatter(xy[:, 0], xy[:, 1], s=ion_size, c=colors_init)
                self.artists.append(sc)
                self._set_axis_limits(ax, view)
                self._set_axis_labels(ax, view)
                if mass is not None and self.mass_indices is not None:
                    self._add_isotope_legend(ax)

        # 出界统计标注（参与 blit 以保证实时刷新）
        self._oob_texts: list = []
        _bbox = dict(boxstyle="round", facecolor="wheat", alpha=0.85)
        for ax in self.ax:
            txt = ax.text(
                0.02,
                0.98,
                "",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11,
                bbox=_bbox,
            )
            self._oob_texts.append(txt)
        self._update_oob_labels(frame_init.r * self._dl_um)

        time.sleep(0.5)
        self.bm = BlitManager(self.fig.canvas, (*self.artists, *self._oob_texts))

        if show_plot:
            plt.show(block=False)

    def _oob_counts_3d(
        self,
        r_um: np.ndarray,
        *,
        y_center_um: float | None = None,
    ) -> tuple[int, int]:
        """
        统计在给定长方体视域之外离子数 (Out-of-bounds count, total).

        x/z: x0_plot±x_range, z0_plot±z_range（μm）.
        y: y0_plot±y_range 当 y_center_um 为 None；
        否则为 y_center_um±y_range（供 bilayer 每层以该层 y 均值为基准）。
        """
        if r_um.size == 0:
            return 0, 0
        r_um = np.asarray(r_um, dtype=float)
        x, y, z = r_um[:, 0], r_um[:, 1], r_um[:, 2]
        xmin, xmax = self.x0_plot - self.x_range, self.x0_plot + self.x_range
        if y_center_um is None:
            ymin, ymax = self.y0_plot - self.y_range, self.y0_plot + self.y_range
        else:
            yc = float(y_center_um)
            ymin, ymax = yc - self.y_range, yc + self.y_range
        zmin, zmax = self.z0_plot - self.z_range, self.z0_plot + self.z_range
        inside = (
            (x >= xmin)
            & (x <= xmax)
            & (y >= ymin)
            & (y <= ymax)
            & (z >= zmin)
            & (z <= zmax)
        )
        n_total = int(r_um.shape[0])
        n_out = int(n_total - np.sum(inside))
        return n_out, n_total

    def _update_oob_labels(self, r_um: np.ndarray) -> None:
        """更新各子图上的出界计数标注；bilayer 时每层 y 窗口以该层 y 均值为中心。"""
        if self.bilayer:
            split = self._bilayer_split
            assert split is not None
            layer0 = np.asarray(r_um[:split], dtype=float)
            layer1 = np.asarray(r_um[split:], dtype=float)
            mean_y0 = float(np.mean(layer0[:, 1])) if layer0.size else 0.0
            mean_y1 = float(np.mean(layer1[:, 1])) if layer1.size else 0.0
            n0, t0 = self._oob_counts_3d(layer0, y_center_um=mean_y0)
            n1, t1 = self._oob_counts_3d(layer1, y_center_um=mean_y1)
            self._oob_texts[0].set_text(f"Out of bounds {n0}/{t0}")
            self._oob_texts[1].set_text(f"Out of bounds {n1}/{t1}")
        else:
            n, t = self._oob_counts_3d(r_um)
            label = f"Out of bounds {n}/{t}"
            for txt in self._oob_texts:
                txt.set_text(label)

    @staticmethod
    def _bilayer_titles_for_r_um(
        t_us: float,
        split: int,
        n_ions: int,
        r_um: np.ndarray,
    ) -> tuple[str, str]:
        """双层图子标题：每层 y（μm）的均值与标准差（对齐 bilayer_prompt）。"""
        y0 = np.asarray(r_um[:split, 1], dtype=float)
        y1 = np.asarray(r_um[split:, 1], dtype=float)
        m0, s0 = float(np.mean(y0)), float(np.std(y0))
        m1, s1 = float(np.mean(y1)), float(np.std(y1))
        t0 = (
            f"t = {t_us:.3f} μs  |  layer [0:{split})  "
            f"μ_y = {m0:.3f} μm, σ_y = {s0:.3f} μm"
        )
        t1 = (
            f"t = {t_us:.3f} μs  |  layer [{split}:{n_ions})  "
            f"μ_y = {m1:.3f} μm, σ_y = {s1:.3f} μm"
        )
        return t0, t1

    def _compute_colors(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.bilayer and self.color_mode == "y_pos":
            assert self._bilayer_split is not None
            return get_colors_layerwise_y_pos(
                r, self._bilayer_split, cmap_name="RdBu"
            )
        return get_colors(
            r,
            v,
            cast(ColorMode, self.color_mode),
            self.mass,
            cmap_name="RdBu",
        )

    def _get_xy(self, r: np.ndarray, view: PlotFig) -> np.ndarray:
        """根据视角返回 (x_display, y_display)，单位 um"""
        r_um = r * self._dl_um
        if view == "xoy":
            return np.column_stack((r_um[:, 0], r_um[:, 1]))
        if view == "zoy":
            return np.column_stack((r_um[:, 2], r_um[:, 1]))
        if view == "zox":
            return np.column_stack((r_um[:, 2], r_um[:, 0]))
        return np.column_stack((r_um[:, 2], r_um[:, 0]))

    def _set_axis_limits(self, ax, view: PlotFig) -> None:
        """设置坐标轴范围"""
        if view == "xoy":
            ax.set_xlim(self.x0_plot - self.x_range, self.x0_plot + self.x_range)
            ax.set_ylim(self.y0_plot - self.y_range, self.y0_plot + self.y_range)
        elif view == "zoy":
            ax.set_xlim(self.z0_plot - self.z_range, self.z0_plot + self.z_range)
            ax.set_ylim(self.y0_plot - self.y_range, self.y0_plot + self.y_range)
        else:  # zox
            ax.set_xlim(self.z0_plot - self.z_range, self.z0_plot + self.z_range)
            ax.set_ylim(self.x0_plot - self.x_range, self.x0_plot + self.x_range)
        ax.set_aspect("equal")

    def _set_axis_labels(self, ax, view: PlotFig) -> None:
        """设置坐标轴标签"""
        if view == "xoy":
            ax.set_xlabel("x (μm)", fontsize=14)
            ax.set_ylabel("y (μm)", fontsize=14)
        elif view == "zoy":
            ax.set_xlabel("z (μm)", fontsize=14)
            ax.set_ylabel("y (μm)", fontsize=14)
        else:
            ax.set_xlabel("z (μm)", fontsize=14)
            ax.set_ylabel("x (μm)", fontsize=14)

    def _save_rv(self, f: Frame, n_ions: int, out_dir: str, basename: str) -> None:
        """保存 r(μm)、v(m/s)、t_us(μs) 到 npz 文件，t_us 供续跑时正确设置 RF 相位"""
        r_um = np.asarray(f.r, dtype=np.float64) * self._dl_um
        v_m_s = np.asarray(f.v, dtype=np.float64) * self.dl / self.dt
        time_us = f.timestamp * self._dt_us
        dir_path = os.path.join(out_dir, str(n_ions))
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{basename}.npz")
        np.savez(path, r=r_um, v=v_m_s, t_us=time_us)
        logger.info("已保存 r/v: %s", path)

    def _add_isotope_legend(self, ax) -> None:
        """添加同位素图例"""
        if self.mass_indices is None:
            return
        unique_indices = np.unique(self.mass_indices)
        legend_labels = [ISOTOPE_LABELS[i] for i in unique_indices]
        legend_colors = [ISOTOPE_COLORS[i] for i in unique_indices]
        if legend_labels:
            ax.legend(
                [
                    plt.Line2D(
                        [0], [0], color=c, marker="o", linestyle="", markersize=8
                    )
                    for c in legend_colors
                ],
                legend_labels,
                loc="upper right",
                ncol=1,
                frameon=True,
                fontsize=12,
            )

    def plot(self, frame: Frame | None = None) -> bool:
        """
        绘制一帧
        frame 为 None 时从 queue_data 获取
        Returns False 表示应停止（收到结束信号或窗口关闭）
        """
        if self.show_plot and not plt.fignum_exists(self.fig.number):
            return False

        if frame is not None:
            f = frame
        else:
            if self.queue_data.empty():
                return True
            item = self.queue_data.get()
            if item is False:
                return False
            f = item

        # 更新各子图
        colors = self._compute_colors(f.r, f.v)
        t_us = f.timestamp * self._dt_us
        if self.bilayer:
            split = self._bilayer_split
            assert split is not None
            r_um = f.r * self._dl_um
            xy0 = np.column_stack((r_um[:split, 2], r_um[:split, 0]))
            xy1 = np.column_stack((r_um[split:, 2], r_um[split:, 0]))
            self.artists[0].set_offsets(xy0)
            self.artists[1].set_offsets(xy1)
            self.artists[0].set_facecolor(colors[:split])
            self.artists[1].set_facecolor(colors[split:])
            n_ions = r_um.shape[0]
            title0, title1 = self._bilayer_titles_for_r_um(
                t_us, split, n_ions, r_um
            )
            self.ax[0].set_title(title0, fontsize=14)
            self.ax[1].set_title(title1, fontsize=14)
            self._update_oob_labels(r_um)
        else:
            for i, view in enumerate(self.plot_fig):
                xy = self._get_xy(f.r, view)
                self.artists[i].set_offsets(xy)
                self.artists[i].set_facecolor(colors)
                self.ax[i].set_title(
                    f"t = {t_us:.3f} μs",
                    fontsize=14,
                )
            self._update_oob_labels(f.r * self._dl_um)

        # 输出时间
        time_us = f.timestamp * self._dt_us
        if time_us - self.last_output_time_us >= 10.0:
            logger.info("Simulation Time: %.3f μs", time_us)
            self.last_output_time_us = time_us

        # 保存指定时刻（save_times_us 为需保存的时刻列表，μs）
        # 原实现用 max(0.5*dt,0.01)μs 作容差：dt 为积分步时常导致容差≈0.01，与输出帧间隔
        # (backend_interval×dt×1e6) 无关。若输出间隔为 0.11 μs 而目标间距 0.1 μs，会出现
        # 连续文件时间差 0.11 μs；若容差过大还会在单帧上错误匹配多个目标。现改为：在尚未
        # 命中的目标中选与当前仿真时刻最近者，且距离不超过由输出步长与保存步长共同决定的捕获半径。
        if self.save_times_us is not None:
            pending = [
                i
                for i in range(len(self.save_times_us))
                if i not in self._save_times_fulfilled
            ]
            if pending:
                i_best = min(
                    pending,
                    key=lambda i: (
                        abs(time_us - float(self.save_times_us[i])),
                        i,
                    ),
                )
                t_nom = float(self.save_times_us[i_best])
                d = abs(time_us - t_nom)
                if d <= self._save_capture_tol_us:
                    n_ions = len(f.r)
                    out_dir = os.path.join(self.save_fig_dir, self.device, str(n_ions))
                    os.makedirs(out_dir, exist_ok=True)
                    tag = f"t{t_nom:.2f}us"
                    path = os.path.join(out_dir, f"{tag}.png")
                    self.fig.savefig(path, dpi=150, bbox_inches="tight")
                    logger.info(
                        "已保存: %s (目标 %.4f μs, 仿真时刻 %.4f μs)",
                        path,
                        t_nom,
                        time_us,
                    )
                    if self.save_rv_traj_dir:
                        rv_dir = os.path.join(self.save_rv_traj_dir, self.device)
                        self._save_rv(f, n_ions, rv_dir, tag)
                    self._save_times_fulfilled.add(i_best)

        # 达到目标时间：保存最后一帧并停止
        if (
            self.target_time_dt is not None
            and f.timestamp >= self.target_time_dt
            and not self.final_frame_saved
        ):
            if self.save_final_image:
                os.makedirs(os.path.dirname(self.save_final_image) or ".", exist_ok=True)
                self.fig.savefig(
                    self.save_final_image, dpi=150, bbox_inches="tight"
                )
                logger.info("已保存最后一帧: %s", self.save_final_image)
                self.final_frame_saved = True
            self.queue_control.put(Message(CommandType.STOP))

        self.bm.update()
        return True

    def start(
        self,
        save_path: str | None = None,
        proc: mp.Process | None = None,
    ) -> Frame | None:
        """
        主循环：从 queue_data 取帧并绘制，直到收到结束信号
        save_path: 若提供，结束时保存最后一帧到该路径
        proc: 若提供，get 超时后检查子进程存活，异常退出时抛出 RuntimeError
        """
        if self.show_plot:
            logger.info("starting plotter...")
        else:
            logger.info("collecting frames (no real-time display)...")

        f: Frame | None = None
        while True:
            try:
                item = self.queue_data.get(
                    timeout=_QUEUE_GET_TIMEOUT if proc is not None else None
                )
            except queue.Empty:
                if proc is not None and not proc.is_alive():
                    exitcode = proc.exitcode or -1
                    raise RuntimeError(
                        f"Backend 子进程异常退出 (exitcode={exitcode})，请检查上述错误信息"
                    ) from None
                continue
            if item is False:
                break
            f = item
            if not self.plot(frame=f):
                break
            if self.show_plot:
                plt.pause(self.interval)

        logger.info("stopping plotter...")
        if save_path and f is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            if self.show_plot and plt.fignum_exists(self.fig.number):
                self.plot(frame=f)
                plt.pause(0.1)
            else:
                self.plot(frame=f)
            self.fig.text(
                0.5,
                0.02,
                f"Simulation Time: {f.timestamp * self._dt_us:.3f} μs",
                ha="center",
                va="bottom",
                fontsize=14,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved final frame to %s", save_path)
        if self.save_rv_status_dir and f is not None:
            time_us = f.timestamp * self._dt_us
            rv_dir = os.path.join(self.save_rv_status_dir, self.device)
            self._save_rv(f, len(f.r), rv_dir, f"t{time_us:.2f}us")
        return f
