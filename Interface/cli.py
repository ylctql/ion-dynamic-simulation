"""
CLI 参数定义与解析
集中管理 argparse 参数、路径解析，以及 Parameters / FieldSettings / Vision 的构建。
init_from_config 返回 Config 对象，在需要处显式传入，无导入顺序依赖。
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from FieldConfiguration.constants import Config, init_from_config
from FieldConfiguration.field_settings import FieldSettings
from Plotter.vision import PlotFig, Vision

logger = logging.getLogger(__name__)


def _parse_n_list(value: str) -> list[int]:
    """
    --N 解析：单个整数或逗号分隔列表，如 500 或 500,2000,10000。
    """
    s = value.strip()
    if not s:
        raise argparse.ArgumentTypeError("--N 不能为空")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            n = int(p, 10)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"--N 中含非法整数: {p!r}") from e
        if n < 1:
            raise argparse.ArgumentTypeError(f"--N 须为正整数，当前: {n}")
        out.append(n)
    return out


_MAX_SAVE_TIME_RANGE_POINTS = 1_000_000


def _split_top_level_commas(s: str) -> list[str]:
    """按逗号切分，忽略括号内的逗号（供 range(a,b,c) 与列表混写）。"""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(c)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _is_whole_number(x: float) -> bool:
    return math.isfinite(x) and abs(x - round(x)) < 1e-9


def _expand_range_times(start: float, stop: float, step: float) -> list[float]:
    """
    与 Python range 相同语义：stop 不包含；step 可为负。
    起止与步长均为整数时走内置 range；否则为浮点序列。
    """
    if step == 0:
        raise ValueError("range 的 step 不能为 0")
    if _is_whole_number(start) and _is_whole_number(stop) and _is_whole_number(step):
        ia, ib, ic = int(round(start)), int(round(stop)), int(round(step))
        n = len(range(ia, ib, ic))
        if n > _MAX_SAVE_TIME_RANGE_POINTS:
            raise ValueError(
                f"range 展开后点数过多 (>{_MAX_SAVE_TIME_RANGE_POINTS})"
            )
        return [float(t) for t in range(ia, ib, ic)]
    out: list[float] = []
    n = 0
    if step > 0:
        x = start
        while x < stop - 1e-12 * max(1.0, abs(stop)) and n < _MAX_SAVE_TIME_RANGE_POINTS:
            out.append(x)
            x += step
            n += 1
    else:
        x = start
        while x > stop + 1e-12 * max(1.0, abs(stop)) and n < _MAX_SAVE_TIME_RANGE_POINTS:
            out.append(x)
            x += step
            n += 1
    if n >= _MAX_SAVE_TIME_RANGE_POINTS:
        raise ValueError(
            f"range 展开后点数过多 (>{_MAX_SAVE_TIME_RANGE_POINTS})"
        )
    return out


def _parse_save_times_segment(seg: str) -> list[float]:
    seg = seg.strip()
    m = re.fullmatch(
        r"range\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)",
        seg,
        flags=re.IGNORECASE,
    )
    if m:
        try:
            start = float(m.group(1).strip())
            stop = float(m.group(2).strip())
            step = float(m.group(3).strip())
        except ValueError as e:
            raise ValueError(f"range(...) 中数值无效: {seg!r}") from e
        return _expand_range_times(start, stop, step)
    # start:stop:step — 无括号，避免 bash 将 ( 当作语法（未加引号的 range(...) 会报错）
    if seg.count(":") == 2:
        a, b, c = seg.split(":", 2)
        try:
            start, stop, step = float(a.strip()), float(b.strip()), float(c.strip())
        except ValueError as e:
            raise ValueError(f"无效 start:stop:step: {seg!r}") from e
        return _expand_range_times(start, stop, step)
    try:
        return [float(seg)]
    except ValueError as e:
        raise ValueError(f"无效时刻: {seg!r}") from e


def parse_save_times_us(raw: str) -> list[float]:
    """
    解析 --save_times_us：逗号分隔的微秒时刻；或 range(start,stop,step)（须加引号以免 bash 解析括号）；
    或 start:stop:step（与 Python range 相同，stop 不含，无需引号）。

    示例：10,20,30 或 'range(100,1100,100)' 或 100:1100:100 或 50,200:500:100,600
    """
    s = raw.strip()
    if not s:
        raise ValueError("内容为空")
    out: list[float] = []
    for part in _split_top_level_commas(s):
        out.extend(_parse_save_times_segment(part))
    if not out:
        raise ValueError("未得到任何保存时刻")
    return out


# 默认路径，可通过环境变量覆盖
DEFAULT_CONFIG_PATH = os.environ.get("ISM_DEFAULT_CONFIG", "FieldConfiguration/configs/default.json")
DEFAULT_CSV_PATH = os.environ.get("ISM_DEFAULT_CSV", "data/monolithic20241118.csv")
DEFAULT_SAVE_FIG_DIR = os.environ.get("ISM_DEFAULT_SAVE_FIG_DIR", "saves/images/traj")
# 仅传文件名时使用的默认目录（写死，便于日常只传文件名）
DEFAULT_CSV_DIR = "data"
DEFAULT_CONFIG_DIR = "FieldConfiguration/configs"

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
    smooth_axes: tuple[str, ...] | None = None
    smooth_sg: tuple[int, int] = (11, 3)


def create_parser() -> argparse.ArgumentParser:
    """创建 ArgumentParser 并添加所有参数"""
    parser = argparse.ArgumentParser(description="离子阱动力学模拟")
    parser.add_argument(
        "--N",
        type=_parse_n_list,
        default=_parse_n_list("50"),
        metavar="N[,N,...]",
        help="离子数；可逗号分隔一次跑多场，如 500,2000,10000（按顺序依次模拟）",
    )
    parser.add_argument("--t0", type=float, default=0.0, help="起始时间 (μs)")
    parser.add_argument("--time", type=float, default=None, help="模拟终止时刻 (μs)，不传则无限运行")
    parser.add_argument("--plot", action="store_true", help="启用实时绘图")
    parser.add_argument("--interval", type=float, default=1.0, help="帧间隔 (dt 单位)")
    parser.add_argument("--step", type=int, default=10, help="每帧积分步数")
    parser.add_argument("--batch", type=int, default=50, help="每批帧数，增大可减少 batch 边界等待造成的卡顿")
    parser.add_argument("--alpha", type=float, default=0.0, help="同位素参杂比例；单同位素模式下为该同位素丰度")
    parser.add_argument(
        "--isotope",
        type=str,
        default=None,
        choices=["Ba133", "Ba134", "Ba135", "Ba136", "Ba137", "Ba138"],
        help="单同位素模式：指定同位素种类，alpha 为该同位素丰度，其余为 Ba135；不指定则使用混合模式",
    )
    parser.add_argument(
        "--init_file",
        type=str,
        default="",
        help="初始 r0/v0 的 .npz 路径，须含 r/v；含 t_us 或文件名 t{X}us.npz 时自动设 t0 以衔接 RF 相位",
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
        "--smooth-axes",
        type=str,
        default="z",
        metavar="AXES",
        help="势场平滑方向：默认 z；可指定 x,y,z 或 x,y 或 x；指定 none 关闭滤波",
    )
    parser.add_argument(
        "--smooth-sg",
        type=str,
        default="11,3",
        metavar="WINDOW,POLY",
        help="Savitzky-Golay 滤波器参数：窗口长度与多项式阶数，逗号分隔，默认 11,3",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="电场 CSV；可仅传文件名(如 monolithic20241118.csv)则自动在 data/ 下查找；可设 ISM_DEFAULT_CSV 覆盖默认",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="电极电压 JSON；可仅传文件名(如 default.json)则自动在 FieldConfiguration/configs/ 下查找；可设 ISM_DEFAULT_CONFIG 覆盖默认",
    )
    parser.add_argument("--save_final_image", type=str, default=None, help="最后一帧保存路径")
    parser.add_argument(
        "--save_times_us",
        type=str,
        default=None,
        help=(
            "需保存轨迹图的时刻 (μs)：逗号分隔如 10,20,30；"
            "或 start:stop:step（同 Python range，stop 不含），如 100:1100:100 即 100..1000，无括号、bash 安全；"
            "或 range(start,stop,step)（整段须加引号，否则 bash 会报括号语法错）；可混写"
        ),
    )
    parser.add_argument(
        "--save_fig_dir",
        type=str,
        nargs="?",
        default=None,
        const=DEFAULT_SAVE_FIG_DIR,
        metavar="DIR",
        help="轨迹帧保存根目录，结构为 {dir}/{离子数}/t{时间}us.png；指定但未传参时默认 saves/images/traj；可设 ISM_DEFAULT_SAVE_FIG_DIR 覆盖默认",
    )
    parser.add_argument(
        "--save_rv_traj_dir",
        type=str,
        nargs="?",
        default=None,
        const="saves/rv/traj",
        metavar="DIR",
        help="指定时刻 r/v 保存根目录；指定但未传参时默认 saves/rv/traj；需 --save_times_us；不指定则不保存",
    )
    parser.add_argument(
        "--save_rv_status_dir",
        type=str,
        nargs="?",
        default=None,
        const="saves/rv/status",
        metavar="DIR",
        help="最后一帧 r/v 保存根目录；指定但未传参时默认 saves/rv/status；不指定则不保存",
    )
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


def parse_and_build(
    args: argparse.Namespace,
    root: Path,
    *,
    n_override: int | None = None,
) -> ParsedRun:
    """
    根据 args 构造 Parameters、FieldSettings、Vision，并返回 ParsedRun。

    Parameters
    ----------
    n_override
        若指定，则本场模拟使用该离子数（用于 --N 多值时由 main 循环传入）。
    """
    # 1. 解析路径并加载配置（仅传文件名时用默认目录）
    def _resolve_path(arg: str, default_full: str, default_dir: str) -> str:
        if not arg:
            return str(root / default_full)
        p = Path(arg)
        if not p.is_absolute() and "/" not in arg and "\\" not in arg:
            return str(root / default_dir / arg)
        return str(root / arg) if not p.is_absolute() else arg

    config_path = _resolve_path(args.config, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_DIR)
    cfg, _ = init_from_config(config_path)

    n_values = getattr(args, "N", None)
    if isinstance(n_values, list) and len(n_values) > 1 and getattr(args, "init_file", ""):
        raise ValueError("指定 --init_file 时不支持多个 --N（初始态离子数固定）")

    # 2. 构建 Parameters
    from Interface.parameters import from_argparse
    from FieldConfiguration.loader import build_voltage_list, field_settings_from_config

    params = from_argparse(args, cfg.dt, n_ions=n_override)

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

        # 演化起点 t0：优先用 npz 内 t_us（确保 RF 相位正确衔接），其次文件名，最后 --t0
        t0_us = None
        if "t_us" in data:
            t0_us = float(np.asarray(data["t_us"]).item())
        elif "t" in data:
            t0_us = float(np.asarray(data["t"]).item())
        if t0_us is None:
            m = re.match(r"t(\d+(?:\.\d+)?)us\.npz$", init_path.name, re.IGNORECASE)
            if m is not None:
                t0_us = float(m.group(1))
        if t0_us is not None:
            params.t0 = t0_us / (cfg.dt * 1e6)
            logger.info("init_file 演化起点 t0 = %.3f μs（RF 相位衔接）", t0_us)
            # 若指定了 --time，需用更新后的 t0 重算 duration 并校验 t0 < time
            time_end_us = getattr(args, "time", None)
            if time_end_us is not None and time_end_us != np.inf:
                if t0_us >= time_end_us:
                    raise ValueError(
                        f"init_file 演化起点 t0={t0_us:.3f} μs 必须小于 --time ({time_end_us} μs)，"
                        "终止时刻不能早于或等于起始时刻"
                    )
                time_end_dt = time_end_us / (cfg.dt * 1e6)
                params.duration = time_end_dt - params.t0
        elif getattr(args, "t0", 0.0) == 0.0:
            logger.warning(
                "init_file 未包含时刻信息且未指定 --t0，使用 t0=0；"
                "若 checkpoint 来自其他时刻，请指定 --t0 以确保 RF 相位正确衔接"
            )

    # 4. 构建 FieldSettings
    csv_path = _resolve_path(args.csv, DEFAULT_CSV_PATH, DEFAULT_CSV_DIR)
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

    # 5. 解析 color_mode（单同位素或混合模式时默认 isotope 着色）
    if args.color_mode is not None:
        color_mode = None if args.color_mode == "none" else args.color_mode
    else:
        use_isotope = params.alpha > 0 or getattr(args, "isotope", None) is not None
        color_mode = "isotope" if use_isotope else None

    # 6. 解析 plot_fig（有效值: xoy, zoy, zox）
    valid_views: set[str] = {"xoy", "zoy", "zox"}
    if args.plot_fig is not None:
        raw = [s.strip().lower() for s in args.plot_fig.split(",") if s.strip()]
        filtered = [v for v in raw if v in valid_views] or ["zoy", "zox"]
        plot_fig = cast(list[PlotFig], filtered)
    else:
        # save_times_us 需 plotter 才能保存，指定时启用 plot_fig（无窗口模式）
        plot_fig = cast(list[PlotFig], ["zoy", "zox"]) if (args.plot or args.save_times_us) else None

    # show_plot：仅 --plot 时弹窗；仅 save_times_us 时无窗口
    show_plot = args.plot

    # 7. 解析 save_times_us
    save_times_us: list[float] | None = None
    if args.save_times_us:
        try:
            save_times_us = parse_save_times_us(args.save_times_us)
        except ValueError as e:
            raise ValueError(
                f"--save_times_us 解析失败 ({args.save_times_us!r}): {e}"
            ) from e

    # 8. 解析 save_fig_dir（未指定时用环境变量或默认）
    save_fig_dir = args.save_fig_dir or DEFAULT_SAVE_FIG_DIR

    # 9. 解析 save_rv 相关（指定但未传参时 nargs='?' 的 const 已生效）
    save_rv_traj_dir: str | None = (args.save_rv_traj_dir or "").strip() or None
    if save_rv_traj_dir == "":
        save_rv_traj_dir = None
    save_rv_status_dir: str | None = (args.save_rv_status_dir or "").strip() or None
    if save_rv_status_dir == "":
        save_rv_status_dir = None

    # 10. 解析 smooth 选项（默认沿 z 方向滤波，--smooth-axes none 关闭）
    smooth_axes: tuple[str, ...] | None = None
    smooth_sg: tuple[int, int] = (11, 3)
    raw_smooth = getattr(args, "smooth_axes", "z")
    if raw_smooth and raw_smooth.strip().lower() != "none":
        axes_parts = [a.strip().lower() for a in raw_smooth.split(",") if a.strip()]
        valid_axes = [a for a in axes_parts if a in "xyz"]
        if valid_axes:
            smooth_axes = tuple(valid_axes)
    if smooth_axes is not None:
        try:
            sg_parts = [p.strip() for p in getattr(args, "smooth_sg", "11,3").split(",")]
            smooth_sg = (
                int(sg_parts[0]) if sg_parts else 11,
                int(sg_parts[1]) if len(sg_parts) >= 2 else 3,
            )
        except (ValueError, IndexError):
            smooth_sg = (11, 3)

    # 11. 构建 Vision
    vision = Vision(
        plot_fig=plot_fig,
        show_plot=show_plot if plot_fig is not None else None,
        color_mode=color_mode,
        ion_size=args.ion_size,
        xm_plot=args.x_range,
        ym_plot=args.y_range,
        zm_plot=args.z_range,
        save_final_image=args.save_final_image,
        save_times_us=save_times_us,
        save_fig_dir=save_fig_dir,
        save_rv_traj_dir=save_rv_traj_dir,
        save_rv_status_dir=save_rv_status_dir,
    )

    return ParsedRun(
        params=params,
        field_settings=field_settings,
        vision=vision,
        step=args.step,
        interval=args.interval,
        batch=args.batch,
        config=cfg,
        smooth_axes=smooth_axes,
        smooth_sg=smooth_sg,
    )
