import multiprocessing as mp
import time
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import ionsim
import matplotlib.cm as cm

from utils import *

# Data calculating backend
class CalculationBackend:
	def __init__(self, step: int = 1000, interval: float = 1, batch: int = 10, save_traj: bool = False, t0: float = 0, g: float = 0.1, dt: float = 1, dl: float = 1):
		"""
		:param step: number of steps to calculate in an interval
		:param interval: time interval
		:param batch: number of data to be sent each time
		"""
		self.step = step
		self.interval = interval
		self.batch = batch
		self.t0 = t0
		self.g = g
		self.dt = dt
		self.dl = dl
		self.save_traj = save_traj

	def run(self, queue_out: mp.Queue, queue_in: mp.Queue):
		"""
		starts backend calculation

		:param queue_out: output channel for data
		:param queue_in: input channel for controls
		"""
		# wait for start signal
		m: Message = queue_in.get()
		while m.command != CommandType.START:
			m = queue_in.get()
		
		# START message should contain everything
		if m.r is None or m.v is None or m.charge is None or m.mass is None or m.force is None:
			raise RuntimeError('Not enough param provided to start calculation')

		r0 = m.r
		v0 = m.v
		charge = m.charge
		mass = m.mass
		force = m.force
		t: float = m.t0

		paused = False

		while(True):
			# consume message first
			while not queue_in.empty():
				m = queue_in.get()
				if m.command == CommandType.START:
					# Ignored. Use CommandType.RESUME for changing params
					pass

				elif m.command == CommandType.PAUSE:
					paused = True

				elif m.command == CommandType.RESUME:
					paused = False
					if m.r is not None:
						r0 = m.r
					if m.v is not None:
						v0 = m.v
					if m.t0 is not None:
						t0 = m.t0
					if m.charge is not None:
						charge = m.charge
					if m.mass is not None:
						mass = m.mass
					if m.force is not None:
						force = m.force

				elif m.command == CommandType.STOP:
					queue_out.put(False)
					return
				
			if paused:
				time.sleep(1)
				continue
			
			start = time.process_time()
			r_list, v_list = ionsim.calculate_trajectory(
				r0, v0, 
				charge, mass,
				self.step * self.batch,
				t,
				t + self.interval * self.batch,
				force
			)
			end = time.process_time()
			print(f't = {t}, dt = {self.interval * self.batch}, elapsed = {end - start}')

			for i in range(self.batch):
				t += self.interval
				queue_out.put(Frame(
					r_list[(i + 1) * self.step - 1],
					v_list[(i + 1) * self.step - 1],
					t
				))
				if self.save_traj:
					dir_name = f"../data_cache/traj/N={charge.shape[0]:d},t0={self.t0:.3f},g={self.g:.10g}/"
					if not os.path.exists(dir_name + "r/"):
						os.makedirs(dir_name +"r/")
						os.makedirs(dir_name +"v/")
					np.save(dir_name+f"r/{t*self.dt*1e6:.3f}us.npy", r_list[(i + 1) * self.step - 1]*self.dl*1e6)
					np.save(dir_name+f"v/{t*self.dt*1e6:.3f}us.npy", v_list[(i + 1) * self.step - 1]*self.dl/self.dt)

			r0 = r_list[-1]
			v0 = v_list[-1]

# Result Plotter
class DataPlotter:
	def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, frame_init : Frame, interval: float, fig_num=2, gamma=0.1, x_range=50, y_range=50, z_range=50, x_bias=0, y_bias=0, z_bias=0, dl=1, dt=1, photos=0, target_ion=0, isotope_ratio=0):
		"""
		:param queue_in: input channel for data
		:param queue_out: not used (reserved)
		:param frame_init: initial frame
		:param interval: plotting interval. Should match data generating speed to avoid stuttering
		"""
		self.queue_in = queue_in
		self.queue_out = queue_out
		self.fig_num = fig_num

		self.dl = dl
		self.dt = dt
		self.gamma = gamma
		self.photos = photos
		self.isotope_ratio = isotope_ratio

		colors = np.full(frame_init.r.shape[0], 'b')
		colors[target_ion] = 'r'

		if self.fig_num == 2:
			self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))

			self.ax[0].set_xlim(-x_range+x_bias, x_range+x_bias)
			self.ax[0].set_ylim(-y_range+y_bias, y_range+y_bias)
			self.ax[0].set_aspect('equal')
			self.ax[0].set_xlabel('x/um', fontsize=14)
			self.ax[0].set_ylabel('y/um', fontsize=14)
			self.ax[0].tick_params(axis='x', labelsize=14)
			self.ax[0].tick_params(axis='y', labelsize=14)

			self.ax[1].set_xlim(-x_range+x_bias, x_range+x_bias)
			self.ax[1].set_ylim(-z_range+z_bias, z_range+z_bias)
			self.ax[1].set_aspect('equal')
			self.ax[1].set_xlabel('x/um', fontsize=14)
			self.ax[1].set_ylabel('z/um', fontsize=14)
			self.ax[1].tick_params(axis='x', labelsize=14)
			self.ax[1].tick_params(axis='y', labelsize=14)

			# self.indices = np.arange(frame_init.r.shape[0])
			self.artists = (
				
				self.ax[0].scatter(frame_init.r[:, 0]*self.dl, frame_init.r[:, 1]*self.dl, 50, colors),
				self.ax[1].scatter(frame_init.r[:, 0]*self.dl, frame_init.r[:, 2]*self.dl, 50, colors),
			)
		elif self.fig_num == 1:
			self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 5))

			self.ax.set_xlim(-z_range+z_bias, z_range+z_bias)
			self.ax.set_ylim(-x_range+x_bias, x_range+x_bias)
			self.ax.set_aspect('equal')
			self.ax.set_xlabel('z/um', fontsize=14)
			self.ax.set_ylabel('x/um', fontsize=14)
			self.ax.tick_params(axis='x', labelsize=14)
			self.ax.tick_params(axis='y', labelsize=14)

			# self.indices = np.arange(frame_init.r.shape[0])
			
			self.artists = (
				self.ax.scatter(frame_init.r[:, 0]*self.dl, frame_init.r[:, 1]*self.dl, 50, colors),
			)
			print("start time:", frame_init.timestamp*self.dt*1e6)
		
		# colorbar
		# norm = Normalize(vmin=np.min(frame_init.r[:, 1]), vmax=np.max(frame_init.r[:, 1]))
		# # 使用颜色映射（'RdBu'：小值蓝，大值红）
		# cmap = cm.RdBu
		# sm = ScalarMappable(cmap=cmap, norm=norm)
		# sm.set_array([])  # 必须调用，否则可能报错
		# self.cbar = self.fig.colorbar(sm, ax=self.ax[1])
		# self.cbar.set_label('y/um', fontsize=14)

		# 同位素的颜色标记
		isotope_labels = ['133', '134', '135', '136', '137', '138']
		color_labels = ['yellow', 'green', 'red', 'blue', 'purple', 'black']
		ax_legend = self.ax[0] if self.fig_num == 2 else self.ax
		ax_legend.legend([plt.Line2D([0], [0], color=c, lw=1) for c in color_labels],
    				isotope_labels,
    				loc='upper right',
    				ncol=2,
    				frameon=False
						)

		time.sleep(0.5)

		self.bm = BlitManager(self.fig.canvas, self.artists)

		self.interval = interval
		self.count = 0

		plt.show(block=False)

	def plot(self):
		if not plt.fignum_exists(self.fig.number):
			return False
		self.count += 1
		if self.queue_in.empty():
			return True
	
		f: Frame = self.queue_in.get()
		if f.timestamp < 1e-5:
			self.count = 0
		else:
			print(self.count * self.interval, f.timestamp)
		
		# After 10us, sort the r & v data
		# if f.timestamp*self.dt> 10:
		# 	self.indices = np.lexsort((f.r[:,1], f.r[:,0], f.r[:,2]))
		# 	f.r = f.r[self.indices]
		# 	f.v = f.v[self.indices]
		
		
		'''
		self.artists[0]._offsets = np.vstack((f.r[:, 0]*self.dl, f.r[:, 1]*self.dl)).T
		self.artists[1]._offsets = np.vstack((f.r[:, 0]*self.dl, f.r[:, 2]*self.dl)).T'''

		# 颜色表示y方向坐标
		# norm = Normalize(vmin=np.min(f.r[:, 1]), vmax=np.max(f.r[:, 1]))
		# # 使用颜色映射（'RdBu'：小值蓝，大值红）
		# cmap = cm.RdBu
		# colors = cmap(norm(f.r[:, 1]))  # 转换为 RGBA 颜色数组
		# self.artists[0].set_facecolor(colors)
		# self.artists[1].set_facecolor(colors)

		# sm = ScalarMappable(cmap=cmap, norm=norm)
		# sm.set_array([])  # 必须调用，否则可能报错
		# self.cbar.update_normal(sm)

		# 颜色表示同位素
		N = f.r.shape[0]
		colors = np.full(N, 'red', dtype=object) 
		colors[:int(N*self.isotope_ratio)] = 'yellow'
		colors[int(N*self.isotope_ratio):2*int(N*self.isotope_ratio)] = 'green'
		colors[2*int(N*self.isotope_ratio):3*int(N*self.isotope_ratio)] = 'blue'
		colors[3*int(N*self.isotope_ratio):4*int(N*self.isotope_ratio)] = 'purple'
		colors[4*int(N*self.isotope_ratio):5*int(N*self.isotope_ratio)] = 'black'
		self.artists[0].set_facecolor(colors)
		self.artists[1].set_facecolor(colors)

		if self.fig_num == 2:
			self.artists[0].set_offsets(np.vstack((f.r[:, 0]*self.dl, f.r[:, 1]*self.dl)).T)
			self.artists[1].set_offsets(np.vstack((f.r[:, 0]*self.dl, f.r[:, 2]*self.dl)).T)

			self.ax[1].set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt), fontsize=14)
		elif self.fig_num == 1:
			self.artists[0].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 0]*self.dl)).T)
			self.ax.set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt), fontsize=14)
		
		np.save("../data_cache/r.npy", f.r*self.dl)

		if self.photos > 0.01:
			n_photo = f.timestamp*self.dt // self.photos
			if not os.path.exists(f"../data_cache/photos/{f.r.shape[0]:d}/{n_photo*self.photos:.10g}us.png"):
				self.fig.savefig(f"../data_cache/photos/{f.r.shape[0]:d}/{n_photo*self.photos:.10g}us.png")

		self.bm.update()

		return True

	def start(self,):
		print('starting plotter...')

		while True:
			if not self.plot():
				break
			plt.pause(self.interval)

		print('stopping plotter...')
