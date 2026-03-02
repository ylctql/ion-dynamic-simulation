"""
验证 backend 与 ionsim 的集成
运行方式：从项目根目录 python ComputeKernel/test_backend.py 或 python -m ComputeKernel.test_backend
"""
import multiprocessing as mp
import sys
from pathlib import Path

# 从子目录运行时需将项目根加入 path，才能 import setup_path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import setup_path

setup_path.ensure_build_in_path(_root)

import numpy as np

from utils import CommandType, Frame, Message
from ComputeKernel.backend import CalculationBackend


def _zero_force(r, v, t):
    """顶层函数，可被 multiprocessing 序列化"""
    return np.zeros_like(r)


def main():
    N = 5
    r0 = np.random.randn(N, 3).astype(np.float64) * 10
    v0 = np.zeros((N, 3), dtype=np.float64)
    charge = np.ones(N, dtype=np.float64)
    mass = np.ones(N, dtype=np.float64)

    queue_control = mp.Queue()
    queue_data = mp.Queue(maxsize=20)

    backend = CalculationBackend(
        step=20,
        interval=1.0,
        batch=3,
        time=10.0,
        device="cpu",
        calc_method="RK4",
    )

    queue_control.put(Message(CommandType.START, r0, v0, 0.0, charge, mass, _zero_force))
    queue_data.put(Frame(r0, v0, 0.0))

    proc = mp.Process(target=backend.run, args=(queue_data, queue_control), daemon=True)
    proc.start()

    frames = []
    while True:
        f = queue_data.get()
        if f is False:
            break
        frames.append(f)
        print(f"  t={f.timestamp:.1f}, r[0]={f.r[0]}")

    proc.join()
    print(f"收到 {len(frames)} 帧，测试通过")


if __name__ == "__main__":
    main()
