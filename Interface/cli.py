"""
CLI 参数定义与解析
集中管理 argparse 参数、路径解析，以及 Parameters / FieldSettings / Vision 的构建。
init_from_config 返回 Config 对象，在需要处显式传入，无导入顺序依赖。
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from FieldConfiguration.constants import Config, init_from_config
from FieldConfiguration.field_settings import FieldSettings
from Plotter.vision import PlotFig, Vision

logger = logging.getLogger(__name__)

# 默认路径，可通过环境变量 ISM_DEFAULT_CONFIG / ISM_DEFAULT_CSV 覆盖
DEFAULT_CONFIG_PATH = os.environ.get("ISM_DEFAULT_CONFIG", "FieldConfiguration/default.json")
DEFAULT_CSV_PATH = os.environ.get("ISM_DEFAULT_CSV", "data/monolithic20241118.csv")

if TYPE_CHECKING:
    from Interface.parameters import Parameters


@dataclass
class ParsedRun:
    """解析后的运行参数，供 main.run 使用"""

    params: "Parameters"
    field_settings: FieldSettings
    vision: Vision
    step: int
    interval: float
    batch: int
    config: Config


def create_parser() -> argparse.ArgumentParser:
    """创建 ArgumentParser 并添加所有参数"""
    parser = argparse.ArgumentParser(description="离子阱动力学模拟")
    parser.add_argument("--N", type=int, default=50, help="离子数")
    parser.add_argument("--t0", type=float, default=0.0, help="起始时间 (μs)")
    parser.add_argument("--time", type=float, default=None, help="演化时间 (μs)，默认无限")
    parser.add_argument("--plot", action="store_true", help="启用实时绘图")
    parser.add_argument("--interval", type=float, default=1.0, help="帧间隔 (dt 单位)")
    parser.add_argument("--step", type=int, default=10, help="每帧积分步数")
    parser.add_argument("--batch", type=int, default=50, help="每批帧数，增大可减少 batch 边界等待造成的卡顿")
    parser.add_argument("--alpha", type=float, default=0.0, help="同位素参杂比例")
    parser.add_argument(
        "--init_file",
        type=str,
        default="",
        help="初始 r0/v0 的 .npz 文件路径，须含 'r'(μm) 和 'v'(m/s) 键，形状 (N,3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备: cpu 或 cuda",
    )
    parser.add_argument(
        "--calc_method",
        type=str,
        default="VV",
        choices=["RK4", "VV"],
        help="积分算法: RK4 或 VV",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="电场 CSV 路径，默认 data/monolithic20241118.csv，可设 ISM_DEFAULT_CSV 覆盖",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="电极电压配置 JSON 路径，默认 FieldConfiguration/default.json，可设 ISM_DEFAULT_CONFIG 覆盖",
    )
    parser.add_argument("--save_final_image", type=str, default=None, help="最后一帧保存路径")
    # 绘图选项
    parser.add_argument(
        "--color_mode",
        type=str,
        default=None,
        choices=["y_pos", "v2", "isotope", "none"],
        help="离子着色模式: y_pos/v2/isotope/none，默认 alpha>0 时 isotope 否则 none",
    )
    parser.add_argument(
        "--plot_fig",
        type=str,
        default=None,
        help="子图视角，逗号分隔如 zoy,zox；默认 --plot 时为 zoy,zox",
    )
    parser.add_argument("--ion_size", type=float, default=5.0, help="散点大小")
    parser.add_argument("--x_range", type=float, default=100.0, help="x 方向显示半宽 (μm)")
    parser.add_argument("--y_range", type=float, default=20.0, help="y 方向显示半宽 (μm)")
    parser.add_argument("--z_range", type=float, default=200.0, help="z 方向显示半宽 (μm)")
    return parser


def parse_and_build(args: argparse.Namespace, root: Path) -> ParsedRun:
    """
    根据 args 构造 Parameters、FieldSettings、Vision，并返回 ParsedRun。
    """
    # 1. 解析路径并加载配置
    config_path = args.config or str(root / DEFAULT_CONFIG_PATH)
    if not Path(config_path).is_absolute():
        config_path = str(root / config_path)
    cfg, _ = init_from_config(config_path)

    # 2. 构建 Parameters
    from Interface.parameters import from_argparse
    from FieldConfiguration.loader import build_voltage_list, field_settings_from_config

    params = from_argparse(args, cfg.dt)

    # 3. 若指定 --init_file，从文件加载 r0、v0（单位 μm、m/s，转为无量纲）
    if getattr(args, "init_file", ""):
        init_path = args.init_file
        if not Path(init_path).is_absolute():
            init_path = str(root / init_path)
        init_path = Path(init_path)
        if not init_path.exists():
            raise FileNotFoundError(f"初始状态文件不存在: {init_path}")
        data = dict(np.load(init_path, allow_pickle=True))
        if "r" not in data or "v" not in data:
            raise ValueError(
                f"--init_file 须为含 'r' 和 'v' 键的 .npz 文件，当前键: {list(data.keys())}"
            )
        r_um = np.asarray(data["r"], dtype=float)
        v_si = np.asarray(data["v"], dtype=float)
        if r_um.ndim != 2 or r_um.shape[1] != 3:
            raise ValueError(f"r 须为 (N, 3)，当前形状: {r_um.shape}")
        if v_si.ndim != 2 or v_si.shape[1] != 3:
            raise ValueError(f"v 须为 (N, 3)，当前形状: {v_si.shape}")
        if r_um.shape[0] != v_si.shape[0]:
            raise ValueError(f"r 与 v 的 N 不一致: {r_um.shape[0]} vs {v_si.shape[0]}")
        if r_um.shape[0] != params.N:
            raise ValueError(
                f"--init_file 中 N={r_um.shape[0]} 与 --N={params.N} 不一致"
            )
        # μm -> 无量纲: r_dim = r_um * 1e-6 / dl
        # m/s -> 无量纲: v_dim = v_si * dt / dl
        params.r0 = (r_um * 1e-6 / cfg.dl).astype(float, order="C")
        params.v0 = (v_si * cfg.dt / cfg.dl).astype(float, order="C")

    # 4. 构建 FieldSettings
    csv_path = args.csv or str(root / DEFAULT_CSV_PATH)
    if not Path(csv_path).is_absolute():
        csv_path = str(root / csv_path)
    csv_exists = Path(csv_path).exists()

    if csv_exists:
        import pandas as pd

        dat = pd.read_csv(csv_path, comment="%", header=None)
        n_voltage = dat.shape[1] - 3
        if n_voltage > 0:
            try:
                field_settings = field_settings_from_config(
                    csv_path, config_path, n_voltage, cfg
                )
            except FileNotFoundError as e:
                logger.warning("%s，使用零电压默认配置", e)
                config = {"g": 0.1, "voltage_list": []}
                voltage_list = build_voltage_list(config, n_voltage, cfg)
                field_settings = FieldSettings(
                    csv_filename=csv_path,
                    voltage_list=voltage_list,
                    g=0.1,
                )
        else:
            field_settings = FieldSettings(
                csv_filename=csv_path,
                voltage_list=[],
                g=0.1,
            )
    else:
        field_settings = FieldSettings(csv_filename="", voltage_list=[], g=0.1)
        if args.csv:
            logger.warning("CSV 文件不存在 %s，使用零外力", csv_path)

    # 5. 解析 color_mode
    if args.color_mode is not None:
        color_mode = None if args.color_mode == "none" else args.color_mode
    else:
        color_mode = "isotope" if params.alpha > 0 else None

    # 6. 解析 plot_fig（有效值: xoy, zoy, zox）
    valid_views: set[str] = {"xoy", "zoy", "zox"}
    if args.plot_fig is not None:
        raw = [s.strip().lower() for s in args.plot_fig.split(",") if s.strip()]
        filtered = [v for v in raw if v in valid_views] or ["zoy", "zox"]
        plot_fig = cast(list[PlotFig], filtered)
    else:
        plot_fig = cast(list[PlotFig], ["zoy", "zox"]) if args.plot else None

    # 7. 构建 Vision
    vision = Vision(
        plot_fig=plot_fig,
        color_mode=color_mode,
        ion_size=args.ion_size,
        xm_plot=args.x_range,
        ym_plot=args.y_range,
        zm_plot=args.z_range,
        save_final_image=args.save_final_image,
    )

    return ParsedRun(
        params=params,
        field_settings=field_settings,
        vision=vision,
        step=args.step,
        interval=args.interval,
        batch=args.batch,
        config=cfg,
    )
