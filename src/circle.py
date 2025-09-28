import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd   
from utils import *
from dataplot import *    
from scipy.signal import savgol_filter

backend = CalculationBackend(step=10, interval=0.01, batch=1)

# all parameters are float!!!
r0 = np.array([[-1., 0., 0.],
               [1, 0., 0.]]) 
v0 = np.array([[0., 1, 0],
               [0, -1, 0]])
mass = np.array([1., 1])
charge = np.array([2., -2.])


# No other forces except Coulomb force
def force(r: np.ndarray, v: np.ndarray, t: float):
    return np.zeros_like(r)

q1 = mp.Queue()
q2 = mp.Queue(maxsize=50)

q1.put(Message(CommandType.START, r0, v0, charge, mass, force))
q2.put(Frame(r0, v0, 0))

plot = DataPlotter(q2, q1, Frame(r0, v0, 0), interval=0.01, z_range=1.5, x_range=1.5, y_range=1.5, dl=1,dt=1)
proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)

proc.start()
plot.start()#True表示只画图 #False表示只输出轨迹到/traj/文件夹# "both"表示二者都做

if proc.is_alive():
    q1.put(Message(CommandType.STOP))
    while True:
        f = q2.get()
        if isinstance(f, bool) and not f:
            break
    proc.join()