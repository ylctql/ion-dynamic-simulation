import multiprocessing as mp
import time
from matplotlib.cm import ScalarMappable
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import ionsim
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from utils import *

# Data calculating backend
class CalculationBackend:
	def __init__(self, step: int = 1000, interval: float = 1, batch: int = 10, time: float = np.inf, device: int = 0):
		"""
		:param step: number of steps to calculate in an interval
		:param interval: time interval
		:param batch: number of data to be sent each time
		"""
		self.device = device
		self.step = step
		self.interval = interval
		self.batch = batch
		self.time = time

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
		t: float = 0

		paused = False

		while True:
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
				self.device,
				r0, v0, 
				charge, mass,
				self.step * self.batch,
				t,
				t + self.interval * self.batch,
				force
			)
			end = time.process_time()
			#print(f't = {t}, dt = {self.interval * self.batch}, elapsed = {end - start}')

			for i in range(self.batch):
				t += self.interval
				if t > self.time:
					queue_in.put(Message(CommandType.STOP))
					queue_out.put(False)
					break
				queue_out.put(Frame(
					r_list[(i + 1) * self.step - 1],
					v_list[(i + 1) * self.step - 1],
					t
				))

			r0 = r_list[-1]
			v0 = v_list[-1]
# Result Plotter
class DataPlotter:
	def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, frame_init : Frame, interval: float, gamma=0.1, x_range=50, y_range=50, z_range=50, x_bias=0, y_bias=0, z_bias=0, dl=1, dt: float = 1):
		"""
		:param queue_in: input channel for data
		:param queue_out: not used (reserved)
		:param frame_init: initial frame
		:param interval: plotting interval. Should match data generating speed to avoid stuttering
		"""
		self.queue_in = queue_in
		self.queue_out = queue_out
		self.dl = dl
		self.dt = dt
		self.gamma = gamma

		self.fig = plt.figure() 
		# self.ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1, rowspan=1, fig=self.fig)
		# self.ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=1, fig=self.fig)
		# self.ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2, fig=self.fig)

		self.ax2 = plt.subplot2grid((6, 1), (0, 0), colspan=1, rowspan=1, fig=self.fig)
		self.ax3 = plt.subplot2grid((6, 1), (1, 0), colspan=1, rowspan=5, fig=self.fig)

		# self.ax1.set_xlim(-x_range+x_bias, x_range+x_bias)
		# self.ax1.set_ylim(-y_range+y_bias, y_range+y_bias)
		# self.ax1.set_aspect('equal')
		# self.ax1.set_xlabel('x/um', fontsize=14)
		# self.ax1.set_ylabel('y/um', fontsize=14)
		# self.ax1.tick_params(axis='x', labelsize=14)
		# self.ax1.tick_params(axis='y', labelsize=14)

		self.ax2.set_xlim(-z_range+z_bias, z_range+z_bias)
		self.ax2.set_ylim(-y_range+y_bias, y_range+y_bias)
		self.ax2.set_aspect('equal')
		self.ax2.set_xlabel('z/um', fontsize=14)
		self.ax2.set_ylabel('y/um', fontsize=14)
		self.ax2.tick_params(axis='x', labelsize=14)
		self.ax2.tick_params(axis='y', labelsize=14)

		self.ax3.set_xlim(-z_range+z_bias, z_range+z_bias)
		self.ax3.set_ylim(-x_range+x_bias, x_range+x_bias)
		self.ax3.set_aspect('equal')
		self.ax3.set_xlabel('z/um', fontsize=14)
		self.ax3.set_ylabel('x/um', fontsize=14)
		self.ax3.tick_params(axis='x', labelsize=14)
		self.ax3.tick_params(axis='y', labelsize=14)

		self.indices = np.arange(frame_init.r.shape[0])
		self.artists = (

			# self.ax1.scatter(frame_init.r[:, 0]*self.dl, frame_init.r[:, 1]*self.dl, 5, 'r'),
			self.ax2.scatter(frame_init.r[:, 2]*self.dl, frame_init.r[:, 1]*self.dl, 5, 'r'),
			self.ax3.scatter(frame_init.r[:, 2]*self.dl, frame_init.r[:, 0]*self.dl, 5, 'r'),
			
		)

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
		if f is False:
			return False
		# if f.timestamp < 1e-5:
		# 	self.count = 0
		# else:
		#	print(self.count * self.interval, f.timestamp)
		
		# After 10us, sort the r & v data
		# if f.timestamp*self.dt> 10:
		# 	self.indices = np.lexsort((f.r[:,1], f.r[:,0], f.r[:,2]))
		# 	f.r = f.r[self.indices]
		# 	f.v = f.v[self.indices]
		
		
		'''
		self.artists[0]._offsets = np.vstack((f.r[:, 0]*self.dl, f.r[:, 1]*self.dl)).T
		self.artists[1]._offsets = np.vstack((f.r[:, 0]*self.dl, f.r[:, 2]*self.dl)).T'''


		# self.artists[0].set_offsets(np.vstack((f.r[:, 0]*self.dl, f.r[:, 1]*self.dl)).T)
		# self.artists[1].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 1]*self.dl)).T)
		# self.artists[2].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 0]*self.dl)).T)

		self.artists[0].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 1]*self.dl)).T)
		self.artists[1].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 0]*self.dl)).T)

		# 归一化 y 值到 [0, 1]
		norm = Normalize(vmin=np.min(f.r[:, 1]), vmax=np.max(f.r[:, 1]))

		# 使用颜色映射（'RdBu'：小值蓝，大值红）
		cmap = cm.RdBu
		colors = cmap(norm(f.r[:, 1]))  # 转换为 RGBA 颜色数组

		self.artists[1].set_facecolor(colors)

		self.ax3.set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt), fontsize=14)

		self.bm.update()

		return True

	def start(self):
		print('starting plotter...')

		f = None
		while True:
			new_f = self.queue_in.get()
			if not self.plot():
				break
			plt.pause(self.interval)
			f = new_f
		
		print('stopping plotter...')
		return f


	def is_alive(self):
		return self.plot()
