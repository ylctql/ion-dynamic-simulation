"""
主函数：整合各模块，创建 queue_control/queue_data，启动 backend 与 plotter
参考 outline.md 与 ism-hybrid

dt/dl/dV 由 parse_and_build 返回的 Config 对象提供，无导入顺序依赖。
"""
from __future__ import annotations

import logging
import os
import queue
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import multiprocessing as mp

if TYPE_CHECKING:
    from Interface.cli import ParsedRun

import setup_path

# 日志配置：可通过 ISM_LOG_LEVEL 环境变量覆盖（DEBUG/INFO/WARNING/ERROR）
_LOG_LEVEL = getattr(logging, os.environ.get("ISM_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 心跳超时（秒）：get 超时后检查子进程，不影响模拟速度（模拟在 backend 进程中）
_QUEUE_GET_TIMEOUT = 1.0

# 统一 path 设置（须在 import backend 之前），setup_path 为唯一修改 sys.path 的模块
_ROOT = setup_path.ensure_build_in_path(Path(__file__).resolve().parent)

import numpy as np

from FieldConfiguration.field_settings import FieldSettings
from FieldParser.csv_reader import read as read_csv
from FieldParser.force import build_force, _zero_force
from Plotter.vision import Vision
from utils import CommandType, Frame, Message
from ComputeKernel.backend import CalculationBackend, get_actual_device
from Plotter.dataplot import DataPlotter


def _get_from_queue(
    queue_data: mp.Queue,
    proc: mp.Process,
) -> object:
    """
    从 queue_data 获取项，带超时。若子进程异常退出则抛出 RuntimeError。
    超时仅用于定期检查 proc 存活，不影响模拟运行速度。
    """
    try:
        return queue_data.get(timeout=_QUEUE_GET_TIMEOUT)
    except queue.Empty:
        if not proc.is_alive():
            exitcode = proc.exitcode or -1
            raise RuntimeError(
                f"Backend 子进程异常退出 (exitcode={exitcode})，请检查上述错误信息"
            ) from None
        return None  # 超时但进程存活，调用方需重试


def _consume_queue_until_done(
    queue_data: mp.Queue,
    queue_control: mp.Queue,
    proc: mp.Process,
    *,
    target_time_dt: float | None = None,
    dt_si: float | None = None,
) -> Frame | None:
    """
    消费 queue_data 直到收到 False。
    若 target_time_dt 非 None 且某帧 timestamp >= target_time_dt，则发送 STOP 后继续消费至结束。
    dt_si: 单位时间 (s)，提供时每 10 μs 输出 Simulation Time 日志。
    """
    f_last = None
    last_output_time_us = -10.0  # 确保首帧可输出
    while True:
        item = _get_from_queue(queue_data, proc)
        if item is None:
            continue
        if item is False:
            break
        f_last = item
        if dt_si is not None:
            time_us = f_last.timestamp * dt_si * 1e6
            if time_us - last_output_time_us >= 10.0:
                logger.info("Simulation Time: %.3f μs", time_us)
                last_output_time_us = time_us
        if target_time_dt is not None and f_last.timestamp >= target_time_dt:
            queue_control.put(Message(CommandType.STOP))
            f_last = _consume_queue_until_done(
                queue_data, queue_control, proc,
                target_time_dt=target_time_dt, dt_si=dt_si,
            )
            break
    return f_last


def _drain_queue_after_stop(
    queue_data: mp.Queue,
    queue_control: mp.Queue,
    proc: mp.Process,
) -> Frame | None:
    """发送 STOP 后消费队列直至结束，返回最后一帧"""
    if not proc.is_alive():
        return None
    queue_control.put(Message(CommandType.STOP))
    return _consume_queue_until_done(queue_data, queue_control, proc)


def _build_force(
    field_settings: FieldSettings,
    cfg,
    charge: np.ndarray,
    root: Path,
) -> Callable:
    """根据 field_settings 构建 force 回调"""
    if not field_settings.csv_filename:
        return _zero_force
    csv_path = Path(field_settings.csv_filename)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    grid_coord, grid_voltage = read_csv(
        csv_path, field_settings, normalize=True, dl=cfg.dl, dV=cfg.dV
    )
    return build_force(field_settings, grid_coord, grid_voltage, charge)


def _create_backend_and_start(
    parsed: ParsedRun,
    force: Callable,
) -> tuple[mp.Process, Frame, mp.Queue, mp.Queue]:
    """
    创建队列、backend、启动子进程，返回 (proc, frame_init, queue_control, queue_data)
    """
    p = parsed.params
    cfg = parsed.config
    r0 = p.get_r0()
    v0 = p.get_v0()
    charge = np.asarray(p.q, dtype=np.float64)
    mass = np.asarray(p.m, dtype=np.float64)
    t0 = p.t0_in_dt()
    duration = p.duration_in_dt()

    queue_control = mp.Queue()
    queue_data = mp.Queue(maxsize=50)

    backend = CalculationBackend(
        step=parsed.step,
        interval=parsed.interval,
        batch=parsed.batch,
        time=duration,
        device=p.device,
        calc_method=p.calc_method,
        dt=cfg.dt,
        dl=cfg.dl,
    )

    frame_init = Frame(r0, v0, t0)
    queue_control.put(Message(CommandType.START, r0, v0, t0, charge, mass, force))
    queue_data.put(frame_init)

    proc = mp.Process(
        target=backend.run,
        args=(queue_data, queue_control),
        daemon=True,
    )
    proc.start()
    return proc, frame_init, queue_control, queue_data


def _save_final_image(
    vision: Vision,
    f_last: Frame,
    queue_data: mp.Queue,
    queue_control: mp.Queue,
    cfg,
    mass: np.ndarray,
) -> None:
    """保存最后一帧到 vision.save_final_image"""
    save_path = vision.save_final_image
    if not save_path:
        return
    import matplotlib

    matplotlib.use("Agg")
    plotter_kwargs = vision.to_dataplot_kwargs(cfg.dl, cfg.dt, mass)
    plotter_kwargs["show_plot"] = False
    plotter = DataPlotter(queue_data, queue_control, f_last, **plotter_kwargs)
    plotter.plot(frame=f_last)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plotter.fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )
    logger.info("已保存最后一帧: %s", save_path)


def run(parsed: ParsedRun) -> Frame | None:
    """
    运行动力学模拟

    Parameters
    ----------
    parsed : ParsedRun
        由 cli.parse_and_build 返回的解析结果，包含 params、field_settings、
        vision、step、interval、batch

    Returns
    -------
    Frame | None
        最后一帧，若无则 None
    """
    cfg = parsed.config
    mass = np.asarray(parsed.params.m, dtype=np.float64)

    actual_device = get_actual_device(parsed.params.device)
    if parsed.params.device == "cuda" and actual_device == "cpu":
        logger.warning("请求使用 cuda，但未编译 CUDA 支持，实际使用 cpu")
    logger.info("积分算法: %s", parsed.params.calc_method)
    logger.info("计算设备: %s", actual_device)

    force = _build_force(
        parsed.field_settings, cfg, np.asarray(parsed.params.q, dtype=np.float64), _ROOT
    )
    proc, frame_init, queue_control, queue_data = _create_backend_and_start(
        parsed, force
    )

    t0 = parsed.params.t0_in_dt()
    duration = parsed.params.duration_in_dt()
    target_time_dt = t0 + duration if np.isfinite(duration) else None

    if parsed.vision.plot_fig is not None:
        plotter_kwargs = parsed.vision.to_dataplot_kwargs(cfg.dl, cfg.dt, mass)
        plotter_kwargs["target_time_dt"] = target_time_dt
        if not plotter_kwargs.get("show_plot", True):
            import matplotlib
            matplotlib.use("Agg")
        plotter = DataPlotter(
            queue_data,
            queue_control,
            frame_init,
            **plotter_kwargs,
        )
        f_last = plotter.start(
            save_path=parsed.vision.save_final_image, proc=proc
        )
        drained = _drain_queue_after_stop(queue_data, queue_control, proc)
        if drained is not None:
            f_last = drained
        proc.join()
    else:
        logger.info("绘图已禁用，仅进行计算...")
        f_last = _consume_queue_until_done(
            queue_data, queue_control, proc,
            target_time_dt=target_time_dt, dt_si=cfg.dt,
        )
        proc.join()

        if f_last is not None:
            logger.info("Simulation Time: %.3f μs", f_last.timestamp * cfg.dt * 1e6)

        if parsed.vision.save_final_image and f_last is not None:
            _save_final_image(
                parsed.vision, f_last, queue_data, queue_control, cfg, mass
            )

        _drain_queue_after_stop(queue_data, queue_control, proc)
        proc.join()

    return f_last


def main():
    from Interface import cli

    parser = cli.create_parser()
    args = parser.parse_args()
    parsed = cli.parse_and_build(args, _ROOT)
    run(parsed)


if __name__ == "__main__":
    # 优先使用 fork，避免 spawn 时子进程需重新导入且可能找不到 ionsim
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # 已设置过则忽略
    main()
