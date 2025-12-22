import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd   
import ionsim
from utils import *
from dataplot import *    
import math,csv,traceback,json
from scipy.signal import savgol_filter
import os
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument('--N', type=int, help='number of ions', default=2)
Parser.add_argument('--time', type=float, help='total simulation time in microseconds', default=np.inf)
Parser.add_argument('--CUDA', action='store_true', help='use CUDA for computation')
Parser.add_argument('--plot', action='store_true', help='enable plotting')
args = Parser.parse_args()

def force(r: np.ndarray, v: np.ndarray, t: float):
    return np.zeros((args.N, 3))

if __name__ == "__main__":
    device = 1 if args.CUDA else 0
    print("Using %s for computation."%( "CUDA" if device==1 else "CPU"))
    # backend = CalculationBackend(step=100, interval=5, batch=50)#step越大精度越高
    backend = CalculationBackend(device=device, step=10000, interval=0.1, batch=50, time=args.time)#step越大精度越高
    N = 2  #离子数
    charge = np.ones(N)
    charge[0] = -1  #2个离子间相互吸引
    mass = np.ones(N) #每个离子质量都是1m，具体大小见下面的m

    r0 = np.array([[1., 0, 0],
                   [-1, 0, 0]])
    v0 = np.array([[0, 0, 0.5],
                   [0, 0, -0.5]])
    # v0 = np.zeros((N, 3))
    
    t0 = 0.0

    q1 = mp.Queue()
    q2 = mp.Queue(maxsize=50)

    q1.put(Message(CommandType.START, r0, v0, t0, charge, mass, force))
    q2.put(Frame(r0, v0, 0))

    proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)   
    proc.start()

    f = None
    if args.plot:
        plot = DataPlotter(q2, q1, Frame(r0, v0, 0), interval=0.04, x_range=1.5,y_range=1.5,z_range=1.5)
        f = plot.start()#True表示只画图 #False表示只输出轨迹到/traj/文件夹# "both"表示二者都做    
        if not plot.is_alive():
            q1.put(Message(CommandType.STOP))
            proc.join()
    else:
        while True:
            new_f = q2.get()
            if isinstance(new_f, bool) and not new_f:
                break
            f = new_f
        proc.join()