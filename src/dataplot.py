import multiprocessing as mp
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import ionsim
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from utils import *

# Data calculating backend
class CalculationBackend:
	def __init__(self, step: int = 1000, interval: float = 1, batch: int = 10, time: float = np.inf, device: int = 0, dt: float = 1.0, dl: float = 1.0, dV: float = 1.0, config_name: str = "flat_28", save_traj: bool = False, isotope: str = "Ba135", g: float = 0.1):
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
		self.dV = dV
		self.config_name = config_name
		self.save_traj = save_traj
		self.isotope = isotope
		self.g = g

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
						t = m.t_satrt
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
					coulombpotential = ionsim.calculate_coulombpotential(self.device, r_list[(i + 1) * self.step - 1], charge)*self.dV
					traj_dir = f"./data_cache/{charge.shape[0]:d}/traj/{self.config_name}/{self.isotope}/g={self.g:.6g}/"
					if not os.path.exists(traj_dir):
						os.makedirs(traj_dir+"r")
						os.makedirs(traj_dir+"v")
						os.makedirs(traj_dir+"Vc")
					np.save(traj_dir+f"r/{t*self.dt*1e6:.3f}us.npy", r_list[(i + 1) * self.step - 1]*self.dl*1e6)
					np.save(traj_dir+f"v/{t*self.dt*1e6:.3f}us.npy", v_list[(i + 1) * self.step - 1]*self.dl/self.dt)
					np.save(traj_dir+f"Vc/{t*self.dt*1e6:.3f}us.npy", coulombpotential)
					# queue_out中两帧之间的时间差应该是interval个dt，即绘制的两帧间隔

			r0 = r_list[-1]
			v0 = v_list[-1]
# Result Plotter
class DataPlotter:
	def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, frame_init : Frame, interval: float, x_range: float=50, y_range: float=50, z_range: float=50, x_bias: float=0, y_bias: float=0, z_bias: float=0, dl=1, dt: float = 1, bilayer: bool = False, mass: np.ndarray = None, show_plot: bool = True):
		"""
		:param queue_in: input channel for data
		:param queue_out: not used (reserved)
		:param frame_init: initial frame
		:param interval: plotting interval. Should match data generating speed to avoid stuttering
		:param mass: mass array for isotope identification
		:param show_plot: whether to show plot in real-time (False means only save final frame)
		"""
		self.queue_in = queue_in
		self.queue_out = queue_out
		self.dl = dl
		self.dt = dt
		self.bilayer = bilayer
		self.mass = mass
		self.show_plot = show_plot
		# 用于记录上次输出的时间，每10us输出一次
		self.last_output_time_us = -10.0
		
		# 定义同位素颜色映射：更轻的离子用亮色（黄色、绿色），更重的离子用深色（蓝色、紫色、黑色）
		self.isotope_masses = np.array([133/135, 134/135, 1.0, 136/135, 137/135, 138/135])
		self.isotope_colors = ['yellow', 'lime', 'red', 'blue', 'purple', 'black']
		self.isotope_labels = ['Ba133', 'Ba134', 'Ba135', 'Ba136', 'Ba137', 'Ba138']
		
		# 为每个质量值找到最接近的同位素类型
		if self.mass is not None:
			self.mass_indices = np.array([np.argmin(np.abs(self.isotope_masses - mass_val)) 
				for mass_val in self.mass], dtype=int)
		else:
			self.mass_indices = None

		self.fig = plt.figure(figsize=(30, 20))

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

			self.artists = (
				self.ax2.scatter(frame_init.r[:, 2]*self.dl, frame_init.r[:, 1]*self.dl, 5, 'r'),
				self.ax3.scatter(frame_init.r[:, 2]*self.dl, frame_init.r[:, 0]*self.dl, 5, 'r'),
			)

		# 添加同位素图例
		if self.mass is not None and not self.bilayer:
			unique_indices = np.unique(self.mass_indices)
			legend_labels = [self.isotope_labels[idx] for idx in unique_indices]
			legend_colors = [self.isotope_colors[idx] for idx in unique_indices]
			if legend_labels:
				self.ax3.legend([plt.Line2D([0], [0], color=c, marker='o', linestyle='', markersize=8) 
					for c in legend_colors], legend_labels, loc='upper right', ncol=1, frameon=True, fontsize=12)

		self.bm = BlitManager(self.fig.canvas, self.artists)
		self.interval = interval

		if self.show_plot:
			plt.show(block=False)

	def plot(self, frame=None):
		"""
		绘制一帧数据
		:param frame: 可选的Frame对象，如果提供则直接使用，否则从队列获取
		"""
		# 如果show_plot=False，不需要检查窗口是否存在
		if self.show_plot and not plt.fignum_exists(self.fig.number):
			return False
		
		# 如果提供了frame参数，直接使用；否则从队列获取
		if frame is not None:
			f = frame
		else:
			if self.queue_in.empty():
				return True
			f: Frame = self.queue_in.get()
			if f is False:
				return False

		if self.bilayer:
			# bilayer
			self.artists[0].set_offsets(np.vstack((f.r[:, 0]*self.dl, f.r[:, 2]*self.dl)).T)
			self.artists[1].set_offsets(np.vstack((f.r[:, 1]*self.dl, f.r[:, 2]*self.dl)).T)
			# Bilayer: +为蓝色，-为红色
			colors = np.full(f.r.shape[0], 'r')
			colors[f.r[:,1]>0] = 'b'
			self.artists[0].set_facecolor(colors)
			self.artists[1].set_facecolor(colors)
		else:
			# ax2 (z-y平面): 使用深度颜色映射
			self.artists[0].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 1]*self.dl)).T)
			norm = Normalize(vmin=np.min(f.r[:, 1]), vmax=np.max(f.r[:, 1]))
			cmap = cm.RdBu
			colors_ax2 = cmap(norm(f.r[:, 1]))
			self.artists[0].set_facecolor(colors_ax2)
			
			# ax3 (z-x平面，xoz平面): 使用同位素颜色标记
			self.artists[1].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 0]*self.dl)).T)
			if self.mass is not None and self.mass_indices is not None and len(self.mass_indices) == f.r.shape[0]:
				colors_ax3 = np.array([self.isotope_colors[idx] for idx in self.mass_indices])
			else:
				norm = Normalize(vmin=np.min(f.r[:, 0]), vmax=np.max(f.r[:, 0]))
				colors_ax3 = cm.RdBu(norm(f.r[:, 0]))
			self.artists[1].set_facecolor(colors_ax3)

		self.ax3.set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt), fontsize=14)
		# f.timestamp*self.dt 已经是微秒单位（因为self.dt在configure.py中已经乘以1e6）
		time_us = f.timestamp * self.dt
		# 每过10us输出一次时间
		if time_us - self.last_output_time_us >= 10.0:
			print(f"Simulation Time: {time_us:.3f} μs")
			self.last_output_time_us = time_us

		if self.show_plot:
			self.bm.update()

		return True

	def start(self, save_path=None):
		if self.show_plot:
			print('starting plotter...')
		else:
			print('collecting frames (no real-time display)...')

		f = None
		while True:
			new_f = self.queue_in.get()
			if isinstance(new_f, bool) and not new_f:
				# 收到结束信号
				break
			# 保存当前帧
			f = new_f
			# 无论是否显示，都要调用plot()来处理数据和输出时间
			# 直接传递frame参数，避免从队列重复获取
			if not self.plot(frame=new_f):
				# plot()返回False表示窗口关闭或出错
				break
			# 如果实时显示，暂停一下
			if self.show_plot:
				plt.pause(self.interval)
		
		print('stopping plotter...')
		if save_path is not None and f is not None:
			save_dir = os.path.dirname(save_path)
			if save_dir:
				os.makedirs(save_dir, exist_ok=True)
			# 绘制最后一帧
			if self.show_plot and plt.fignum_exists(self.fig.number):
				self.plot()
				plt.pause(0.1)
			else:
				self.plot()
			# 在图片上添加时间信息
			# f应该是Frame对象，包含timestamp属性
			from utils import Frame
			if isinstance(f, Frame):
				time_us = f.timestamp * self.dt
				# 在ax3标题中已经显示了时间，但为了更明显，可以在图片上添加文本
				# 使用fig.text在图片上添加时间信息（位于图片底部中央）
				self.fig.text(0.5, 0.02, f'Simulation Time: {time_us:.3f} μs', 
				             ha='center', va='bottom', fontsize=14, 
				             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
			# 保存图片
			self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
			print(f'Saved final frame to {save_path}')
		return f


	def is_alive(self):
		return self.plot()
