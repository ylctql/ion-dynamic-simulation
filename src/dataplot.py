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
	def __init__(self, step: int = 1000, interval: float = 1, batch: int = 10, time: float = np.inf, device: int = 0, dt: float = 1.0, dl: float = 1.0, config_name: str = "flat_28", save_traj: bool = False, isotope: str = "Ba135"):
		"""
		:param step: number of steps to calculate in an interval
		:param interval: time interval between 2 adjacent frames
		:param batch: number of data to be sent each time
		"""
		self.device = device
		self.step = step
		self.interval = interval
		self.batch = batch
		self.time = time
		self.dt = dt
		self.dl = dl
		self.config_name = config_name
		self.save_traj = save_traj
		self.isotope = isotope

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
		t: float = m.t_satrt

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
					if m.t_satrt is not None:
						t_start = m.t_satrt
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
				if self.save_traj:
					traj_dir = f"./data_cache/{charge.shape[0]:d}/traj/{self.config_name}/{self.isotope}/"
					np.save(traj_dir+f"r/{t*self.dt*1e6:.3f}us.npy", r_list[(i + 1) * self.step - 1]*self.dl*1e6)
					np.save(traj_dir+f"v/{t*self.dt*1e6:.3f}us.npy", v_list[(i + 1) * self.step - 1]*self.dl/self.dt)
					# queue_out中两帧之间的时间差应该是interval个dt，即绘制的两帧间隔

			r0 = r_list[-1]
			v0 = v_list[-1]
# Result Plotter
class DataPlotter:
	def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, frame_init : Frame, interval: float, x_range=50, y_range=50, z_range=50, x_bias=0, y_bias=0, z_bias=0, dl=1, dt: float = 1, bilayer: bool = False):
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
		self.bilayer = bilayer

		self.fig = plt.figure(figsize=(30, 20)) 
		# self.ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1, rowspan=1, fig=self.fig)
		# self.ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=1, fig=self.fig)
		# self.ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2, fig=self.fig)

		if self.bilayer:
			# bilayer
			self.ax2 = plt.subplot2grid((1, 6), (0, 0), colspan=1, rowspan=1, fig=self.fig)
			self.ax3 = plt.subplot2grid((1, 6), (0, 2), colspan=4, rowspan=1, fig=self.fig)

			self.ax2.set_ylim(-z_range+z_bias, z_range+z_bias)
			self.ax2.set_xlim(-x_range+x_bias, x_range+x_bias)
			self.ax2.set_aspect('equal')
			self.ax2.set_ylabel('z/um', fontsize=14)
			self.ax2.set_xlabel('x/um', fontsize=14)
			self.ax2.tick_params(axis='x', labelsize=14)
			self.ax2.tick_params(axis='y', labelsize=14)

			self.ax3.set_ylim(-z_range+z_bias, z_range+z_bias)
			self.ax3.set_xlim(-y_range+y_bias, y_range+y_bias)
			self.ax3.set_aspect('equal')
			self.ax3.set_ylabel('z/um', fontsize=14)
			self.ax3.set_xlabel('y/um', fontsize=14)
			self.ax3.tick_params(axis='x', labelsize=14)
			self.ax3.tick_params(axis='y', labelsize=14)
			self.artists = (
			self.ax2.scatter(frame_init.r[:, 0]*self.dl, frame_init.r[:, 2]*self.dl, 5, 'r'),
			self.ax3.scatter(frame_init.r[:, 1]*self.dl, frame_init.r[:, 2]*self.dl, 5, 'r'),
			
		)
		else:
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

			# self.indices = np.arange(frame_init.r.shape[0])
			self.artists = (
				# self.ax1.scatter(frame_init.r[:, 0]*self.dl, frame_init.r[:, 1]*self.dl, 5, 'r'),
				self.ax2.scatter(frame_init.r[:, 2]*self.dl, frame_init.r[:, 1]*self.dl, 5, 'r'),
				self.ax3.scatter(frame_init.r[:, 2]*self.dl, frame_init.r[:, 0]*self.dl, 5, 'r'),
			)

		# isotope_labels = ['133', '134', '135', '136', '137', '138']
		# color_labels = ['y', 'g', 'r', 'b', 'c', 'k']
		# self.ax3.legend([plt.Line2D([0], [0], color=c, lw=1) for c in color_labels],
    	# 			isotope_labels,
    	# 			loc='upper right',
    	# 			ncol=2,
    	# 			frameon=False
		# 				)

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

		if self.bilayer:
			# bilayer
			self.artists[0].set_offsets(np.vstack((f.r[:, 0]*self.dl, f.r[:, 2]*self.dl)).T)
			self.artists[1].set_offsets(np.vstack((f.r[:, 1]*self.dl, f.r[:, 2]*self.dl)).T)
			# Bilayer: +为蓝色，-为红色
			colors = np.full(f.r.shape[0], 'r')
			colors[f.r[:,1]>0] = 'b'
		else:
			self.artists[0].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 1]*self.dl)).T)
			self.artists[1].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 0]*self.dl)).T)
			
			# 深度决定颜色
			mask = (np.abs(f.r[:, 0]*self.dl)<50) & (np.abs(f.r[:, 1]*self.dl)<20)
			# 归一化 y 值到 [0, 1]
			norm = Normalize(vmin=np.min(f.r[mask, 1]), vmax=np.max(f.r[mask, 1]))
			# 使用颜色映射（'RdBu'：小值蓝，大值红）
			cmap = cm.RdBu
			colors = cmap(norm(f.r[:, 1]))  # 转换为 RGBA 颜色数组

			#颜色区分不同同位素
			# colors = np.full(f.r.shape[0], 'r')
			# colors[:100] = 'y' #133
			# colors[100:200] = 'g' #134
			# colors[200:300] = 'b' #136
			# colors[300:400] = 'c' #137
			# colors[400:500] = 'k' #138

		
		# colors = np.where(f.r.shape[:,1]>0, 'b', colors)
		self.artists[0].set_facecolor(colors)
		self.artists[1].set_facecolor(colors)

		self.ax3.set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt), fontsize=14)

		print(f.timestamp, f.timestamp*self.dt)

		# if not os.path.exists("./data_cache/%d/ion_pos"%f.r.shape[0]):
		# 	os.makedirs("./data_cache/%d/ion_pos/"%f.r.shape[0])
		
		# duration = 10

		# if f.timestamp*self.dt >100 and not os.path.exists("./data_cache/%d/ion_pos/%dus.npy"%(f.r.shape[0], duration*int(f.timestamp*self.dt/duration))):
		# 	np.save("./data_cache/%d/ion_pos/%dus.npy"%(f.r.shape[0], duration*int(f.timestamp*self.dt/duration)), f.r*self.dl)

		# photos = 1
		# n_photo = f.timestamp*self.dt // photos
		# if not os.path.exists(f"./data_cache/10000/photos/{n_photo*photos:.10g}us.png"):
		# 	self.fig.savefig(f"./data_cache/10000/photos/{n_photo*photos:.10g}us.png")


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
