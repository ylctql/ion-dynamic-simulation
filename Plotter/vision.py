"""
可视化参数设置
参考 outline.md - 实时绘图选项
供 Plotter/DataPlotter 使用，单位采用 μm、μs 等常用单位
"""
from dataclasses import dataclass, field
from typing import Literal

PlotFig = Literal["xoy", "zoy", "zox"]
ColorMode = Literal["y_pos", "v2", "isotope"] | None


@dataclass
class Vision:
    """
    可视化参数
    供 Plotter/DataPlotter 使用
    """

    # ----- 保存选项 -----
    save_times_us: list[float] | None = None
    """
    需保存轨迹图的时刻（国际单位制，μs）
    - None: 不保存
    - []: 仅保存最后一帧
    - [t1, t2, ...]: 在指定时刻（μs）保存帧，如 [10.0, 50.0, 100.0]
    """
    save_fig_dir: str = "saves/images/traj"
    """save_times_us 保存的根目录，结构为 {save_fig_dir}/{device}/{离子数}/t{时间}us.png"""

    save_rv_traj_dir: str | None = None
    """
    指定时刻 r/v 保存根目录；None 不保存
    指定该变量但未传参时默认 saves/rv/traj；结构为 {dir}/{device}/{离子数}/t{时间}us.npz
    需同时设置 save_times_us
    """
    save_rv_status_dir: str | None = None
    """
    最后一帧 r/v 保存根目录；None 不保存
    指定该变量但未传参时默认 saves/rv/status；结构为 {dir}/{device}/{离子数}/t{时间}us.npz（以最后一帧时间命名）
    不要求保存最后一帧图片
    """

    # ----- 绘图开关与视角 -----
    plot_fig: list[PlotFig] | None = field(default_factory=lambda: ["zoy", "zox"])
    """
    子图视角，决定是否实时绘图及布局
    - None: 不进行实时绘图
    - ["zoy", "zox"]: 两子图，z-y 与 z-x 视图
    - ["xoy"], ["zox"] 等: 单子图
    """
    show_plot: bool | None = None
    """是否弹出窗口实时显示；None 时由 plot_fig 推导（plot_fig 非空则为 True）"""

    # ----- 离子显示 -----
    ion_size: float = 5.0  # 散点大小

    # ----- 显示区域范围（单位 μm）-----
    x0_plot: float = 0.0  # 区域中心 x
    y0_plot: float = 0.0  # 区域中心 y
    z0_plot: float = 0.0  # 区域中心 z
    xm_plot: float = 100.0  # x 方向偏离中心最大值（半宽）
    ym_plot: float = 20.0  # y 方向偏离中心最大值（半宽）
    zm_plot: float = 200.0  # z 方向偏离中心最大值（半宽）

    # ----- 双层晶格绘图 -----
    bilayer: bool = False
    """True：两子图均为 z-x 投影，分别显示索引两半离子；不绘 zoy"""

    # ----- 离子颜色 -----
    color_mode: ColorMode = None
    """
    None: 全部红色
    "y_pos": 按 y 坐标上色
    "v2": 按速度模平方上色
    "isotope": 按同位素种类上色
    """

    # ----- 其他 -----
    save_final_image: str | None = None  # 最后一帧保存路径
    plot_interval: float = 0.001  # 绘图刷新间隔（秒），0.04 会每帧固定等待 40ms 导致模拟卡顿

    def to_dataplot_kwargs(self, dl: float, dt: float, mass=None) -> dict:
        """
        转换为 DataPlotter 的参数字典
        供 main.py 等调用时使用
        """
        return {
            "plot_fig": self.plot_fig,
            "bilayer": self.bilayer,
            "color_mode": self.color_mode,
            "ion_size": self.ion_size,
            "x_range": self.xm_plot,
            "y_range": self.ym_plot,
            "z_range": self.zm_plot,
            "x0_plot": self.x0_plot,
            "y0_plot": self.y0_plot,
            "z0_plot": self.z0_plot,
            "save_times_us": self.save_times_us,
            "save_fig_dir": self.save_fig_dir,
            "save_rv_traj_dir": self.save_rv_traj_dir,
            "save_rv_status_dir": self.save_rv_status_dir,
            "save_final_image": self.save_final_image,
            "interval": self.plot_interval,
            "dl": dl,
            "dt": dt,
            "mass": mass,
            "show_plot": self.show_plot if self.show_plot is not None else (self.plot_fig is not None),
        }
