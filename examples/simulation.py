import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

import ionsim

from utils import *
from dataplot import *

N = 50
# rand = [np.array([]), -100]
charge = np.ones(N)
mass = np.ones(N)
# mass[-1] = 0.8

def force(r: np.ndarray, v: np.ndarray, t: float):

	aa = np.array([-0.00005, 0.0001, 0.4])
	q = np.array([0.1, 0, -0.1])

	gamma = (0.7 - t / 30 * 0.6) if t < 30 else 0.05
	# gamma = 0.5

	f: np.ndarray = - 2000 * r * (1 + 0.001 * np.sum(r * r, axis=0)) * charge.reshape(-1, 1) * (aa - q * np.cos(120 * t)) - gamma * v
	return f

if __name__ == "__main__":
	
	

	backend = CalculationBackend(step=120, interval=0.04, batch=50)

	r0 = np.random.rand(N, 3) * 10 - 5
	# r0[:, -1] = np.array([30, 0, 0])
	v0 = np.zeros((N, 3))

	q1 = mp.Queue()
	q2 = mp.Queue()

	q1.put(Message(CommandType.START, r0, v0, mass, charge, force))
	q2.put(Frame(r0, v0, 0))

	plot = DataPlotter(q2, q1, Frame(r0, v0, 0), interval=0.04)
	proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)
	
	proc.start()
	plot.start()

	q1.put(Message(CommandType.STOP))
	while True:
		f = q2.get()
		if isinstance(f, bool) and not f:
			break
	proc.join()