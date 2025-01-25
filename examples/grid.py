import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import ionsim

from utils import *
from dataplot import *


N = 50
charge = np.ones(N)
mass = np.ones(N)

# load grid
data = pd.read_csv('examples/out.csv', comment='%', header=None).to_numpy()
data = data[np.lexsort([data[:, 2], data[:, 1], data[:, 0]])]
x, y, z = [np.unique(data[:, i]) for i in range(3)]
potential = data[:, 3].reshape(len(x), len(y), len(z))
fieldx, fieldy, fieldz = np.gradient(-potential, x, y, z, edge_order=2)

gridx = ionsim.Grid(x, y, z, value=fieldx)
gridy = ionsim.Grid(x, y, z, value=fieldy)
gridz = ionsim.Grid(x, y, z, value=fieldz)

def force(r: np.ndarray, v: np.ndarray, t: float):
	f = np.zeros_like(r)
	mask = gridx.in_bounds(r)
	# make a copy for faster vectorization
	r_mask = r[mask].copy(order='F')
	r_nmask = r[~mask].copy(order='F')

	# inside bounds
	coord = gridx.get_coord(r_mask)
	f = np.vstack((gridx.interpolate(coord), gridy.interpolate(coord), gridz.interpolate(coord))).transpose()
	# outside bounds
	f[~mask] = np.zeros_like(r_nmask)
	
	return f

if __name__ == "__main__":
	backend = CalculationBackend(step=120, interval=0.04, batch=50)

	r0 = np.random.rand(N, 3) * 10 - 5
	v0 = np.zeros((N, 3))

	q1 = mp.Queue()
	q2 = mp.Queue(maxsize=50)

	q1.put(Message(CommandType.START, r0, v0, mass, charge, force))
	q2.put(Frame(r0, v0, 0))

	plot = DataPlotter(q2, q1, Frame(r0, v0, 0), interval=0.04)
	proc = mp.Process(target=backend.run, args=(q2, q1,), daemon=True)
	
	proc.start()
	plot.start()

	if proc.is_alive():
		q1.put(Message(CommandType.STOP))
		while True:
			f = q2.get()
			if isinstance(f, bool) and not f:
				break
		proc.join()