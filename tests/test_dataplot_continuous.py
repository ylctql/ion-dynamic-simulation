"""DataPlotter continuous-sampling 边看边存单元测试。

验证 plot() 在 continuous_save_dir/continuous_n_frames 启用时：
- 逐帧存 frame{idx}.npz（frame0=初始帧，与纯 --continuous-sampling 同格式）；
- 计数达 n_frames 后停止保存且 STOP 恰发一次（覆盖 batch 残余帧场景）；
- 未启用时（无 continuous_* kwargs）不写盘（回归保护）。
"""
import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot / dataplot 导入前

import multiprocessing as mp

import numpy as np

from utils import CommandType, Frame
from Plotter.dataplot import DataPlotter


def _make_plotter(tmp_path, n_frames, *, enable=True):
    """构造一个 continuous 模式的 DataPlotter（show_plot=False，Agg 后端）。"""
    qd = mp.Queue()
    qc = mp.Queue()
    # 4 离子，小非零位置避免退化；frame_init 即将作为 frame0 存盘
    r0 = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 2.0],
                   [0.5, 0.0, -1.0], [-0.5, 0.0, -2.0]])
    f0 = Frame(r0, np.zeros_like(r0), 0.0)
    kwargs = dict(
        show_plot=False, dl=1e-7, dt=1e-7, target_time_dt=None,
    )
    if enable:
        kwargs["continuous_save_dir"] = str(tmp_path)
        kwargs["continuous_n_frames"] = n_frames
    return DataPlotter(qd, qc, f0, **kwargs), qd, qc, f0


def _drain_commands(qc):
    out = []
    while not qc.empty():
        out.append(qc.get().command)
    return out


def _frame(t):
    r = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 2.0],
                  [0.5, 0.0, -1.0], [-0.5, 0.0, -2.0]])
    return Frame(r, np.zeros_like(r), float(t))


def test_continuous_saves_each_frame_init_is_frame0(tmp_path):
    """frame0 = frame_init；逐帧存盘 + 计数递增；达 n_frames 后下一帧不存、STOP 一次。"""
    p, qd, qc, f0 = _make_plotter(tmp_path, n_frames=3)
    # frame0 = 初始帧
    assert p.plot(frame=f0) is True
    assert (tmp_path / "frame0.npz").exists()
    assert p._continuous_frame_count == 1
    # frame1, frame2
    assert p.plot(frame=_frame(1)) is True
    assert p.plot(frame=_frame(2)) is True
    assert (tmp_path / "frame1.npz").exists()
    assert (tmp_path / "frame2.npz").exists()
    assert p._continuous_frame_count == 3
    # 达上限：下一帧不存，STOP 恰一次
    assert p.plot(frame=_frame(3)) is True
    assert not (tmp_path / "frame3.npz").exists()
    assert _drain_commands(qc).count(CommandType.STOP) == 1
    # 存盘内容键正确
    d = np.load(tmp_path / "frame0.npz")
    assert set(d.keys()) == {"r", "v", "t_us"}
    assert d["r"].shape == (4, 3)


def test_continuous_single_stop_under_leftover_batch(tmp_path):
    """覆盖边界 A：达 n_frames 后仍有残余帧涌入（batch 未排空），STOP 绝不重复。"""
    p, qd, qc, f0 = _make_plotter(tmp_path, n_frames=2)
    for i in range(6):  # 远超 n_frames，模拟残余 batch 帧
        assert p.plot(frame=_frame(i)) is True
    assert _drain_commands(qc).count(CommandType.STOP) == 1
    assert sorted(tmp_path.glob("frame*.npz")) == sorted(
        [tmp_path / "frame0.npz", tmp_path / "frame1.npz"]
    )


def test_continuous_disabled_does_not_save(tmp_path):
    """回归：无 continuous_* kwargs 时 plot() 不写盘。"""
    p, qd, qc, f0 = _make_plotter(tmp_path, n_frames=None, enable=False)
    assert p.plot(frame=_frame(1)) is True
    assert p.plot(frame=_frame(2)) is True
    assert list(tmp_path.glob("frame*.npz")) == []
    assert _drain_commands(qc) == []
