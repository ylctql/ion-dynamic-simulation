import time
import ionsim 
import numpy as np
import pandas as pd

r0 = np.array([
	[1, 0, 0],
	[-1, 0, 0],
], dtype=np.float64)
v0 = np.array([
	[0, 1, 0],
	[0, -1, 0],
], dtype=np.float64)

charge = np.array([1, -1], dtype=np.float64)
mass = np.array([1, 1], dtype=np.float64)

data = pd.read_csv('tests/test.csv', comment='%', header=None).to_numpy()
data = data[np.lexsort([data[:, 2], data[:, 1], data[:, 0]])]
x, y, z = [np.unique(data[:, i]) for i in range(3)]
potential = data[:, 3].reshape(len(x), len(y), len(z))
fieldx, fieldy, fieldz = np.gradient(-potential, x, y, z, edge_order=2)

gridx = ionsim.Grid(x, y, z, value=fieldx)
gridy = ionsim.Grid(x, y, z, value=fieldy)
gridz = ionsim.Grid(x, y, z, value=fieldz)

def force(r: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
	# return - r * 0.75 * mass.reshape(-1, 1)
	coord = gridx.get_coord(r)
	return np.vstack((gridx.interpolate(coord), gridy.interpolate(coord), gridz.interpolate(coord))).transpose() * 0.75

r = r0
v = v0

error = np.array([])

for i in range(0, 100, 10):
	r_list, v_list = ionsim.calculate_trajectory(
		r, v, 
		charge, mass, 
		1000, i, i + 10, 
		force
	)

	print(np.abs(np.array([np.linalg.norm(r[0]) for r in r_list]).max() - 1), i)

	r = r_list[-1]
	v = v_list[-1]

	error = np.append(error, np.abs([np.linalg.norm(r[0]) - 1 for r in r_list]))

# plt.plot(np.linspace(0, 100, error.shape[0]), error)
# plt.show()

assert(len(r_list) == 1000)
assert(len(v_list) == 1000)
assert(np.abs(np.array([r[0] for r in r_list]).max() - 1) < 0.005)
assert(np.abs(np.array([np.linalg.norm(v[0]) for v in v_list]).max() - 1) < 0.005)

# data = pd.read_csv('tests/test.csv', comment='%', header=None).to_numpy()
# data = data[np.lexsort([data[:, 2], data[:, 1], data[:, 0]])]
# r = [np.unique(data[:, i]) for i in range(3)]
# a = ionsim.Grid(x=r[0], y=r[1], z=r[2], value=data[:, 3].reshape(len(r[0]), len(r[1]), len(r[2])))

# tmp = np.zeros((1, 3))
# t1 = time.perf_counter()
# g: np.ndarray = a.interpolate(tmp)
# t2 = time.perf_counter()
# print(t2 - t1)
# print(g, g.flags)

# t1 = time.perf_counter()
# c = a.get_coord(tmp)
# g2 = a.interpolate(c)
# t2 = time.perf_counter()
# print(t2 - t1)
# print(g.flags)