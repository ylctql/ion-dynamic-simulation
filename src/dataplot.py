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
	def __init__(self, step: int = 1000, interval: float = 1, batch: int = 10,
	             time: float = np.inf, device: int = 0, dt: float = 1.0, 
	             dl: float = 1.0, dV: float = 1.0, config_name: str = "flat_28",
	             save_traj: bool = False, isotope_type: str = "Ba135", g: float = 0.1):
		"""
		:param step: number of steps to calculate in an interval
		:param interval: time interval between 2 adjacent frames
		:param batch: number of data to be sent each time
		:param time: total simulation time (in dt units)
		:param device: 计算设备标志，0 表示 CPU；目前 ism-cpu 仅支持 CPU
		:param dt: unit time
		:param dl: unit length
		:param dV: unit voltage
		:param config_name: configuration name for saving trajectory
		:param save_traj: whether to save trajectory
		:param isotope_type: isotope type for saving trajectory
		:param g: cooling rate
		"""
		# 目前只支持 CPU，若传入非 0 设备则直接报错，避免误用
		if device not in (0, False, None):
			raise ValueError("ism-cpu 目前只支持 CPU 计算（device=0），请不要传入 CUDA 设备。")
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
		self.isotope_type = isotope_type
		self.g = g
		self.last_print_time_us = None  # 上次输出时间（微秒）
		self.print_interval_us = 10.0  # 输出间隔（微秒）

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
		t: float = m.t0 if hasattr(m, 't0') and m.t0 is not None else 0.0

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
					if hasattr(m, 't0') and m.t0 is not None:
						t = m.t0
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

			for i in range(self.batch):
				t += self.interval
				# 检查是否达到总模拟时间
				if t > self.time:
					queue_in.put(Message(CommandType.STOP))
					queue_out.put(False)
					break
				# 将时间转换为微秒
				t_us = t * self.dt * 1e6
				
				# 检查是否需要输出（每10us输出一次）
				if self.last_print_time_us is None or (t_us - self.last_print_time_us) >= self.print_interval_us:
					print(f't = {t_us:.3f}us, elapsed = {end - start:.3f}s')
					self.last_print_time_us = t_us
				
				queue_out.put(Frame(
					r_list[(i + 1) * self.step - 1],
					v_list[(i + 1) * self.step - 1],
					t
				))
				if self.save_traj:
					traj_dir = f"./data_cache/{charge.shape[0]:d}/traj/{self.config_name}/{self.isotope_type}/g={self.g:.6g}/"
					if not os.path.exists(traj_dir):
						os.makedirs(traj_dir+"r")
						os.makedirs(traj_dir+"v")
					np.save(traj_dir+f"r/{t*self.dt*1e6:.3f}us.npy", r_list[(i + 1) * self.step - 1]*self.dl*1e6)
					np.save(traj_dir+f"v/{t*self.dt*1e6:.3f}us.npy", v_list[(i + 1) * self.step - 1]*self.dl/self.dt)

			r0 = r_list[-1]
			v0 = v_list[-1]

# Result Plotter
class DataPlotter:
	def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, frame_init : Frame,
	             interval: float, fig_num=2, gamma=0.1, x_range=50, y_range=50,
	             z_range=50, x_bias=0, y_bias=0, z_bias=0, dl=1, dt=1,
	             photos=0, target_ion=0, isotope_ratio=0, save_final_image=None,
	             target_time_dt=None, bilayer: bool = False,
	             mass: np.ndarray | None = None, show_plot: bool = True):
		"""
		:param queue_in: input channel for data
		:param queue_out: not used (reserved)
		:param frame_init: initial frame
		:param interval: plotting interval. Should match data generating speed to avoid stuttering
		:param save_final_image: path to save the final frame image
		:param target_time_dt: target evolution time in dt units, if None then run indefinitely
		"""
		self.queue_in = queue_in
		self.queue_out = queue_out
		self.fig_num = fig_num

		self.dl = dl
		self.dt = dt
		self.gamma = gamma
		self.photos = photos
		self.isotope_ratio = isotope_ratio
		self.save_final_image = save_final_image
		self.target_time_dt = target_time_dt
		self.final_frame_saved = False
		# 与 ism-cuda 对齐的新参数
		self.bilayer = bilayer
		self.mass = mass
		self.show_plot = show_plot
		
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
		
		# 用于控制时间输出的变量
		self.last_output_time_us = -10.0

		colors = np.full(frame_init.r.shape[0], 'b')
		colors[target_ion] = 'r'

		if self.fig_num == 2:
			# 改为上下布局 (2, 1) 而不是 (1, 2)
			self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 10))

			# 第一个子图：z-y视图，横坐标z轴，纵坐标y轴
			self.ax[0].set_xlim(-z_range+z_bias, z_range+z_bias)
			self.ax[0].set_ylim(-y_range+y_bias, y_range+y_bias)
			self.ax[0].set_aspect('equal')
			self.ax[0].set_xlabel('z/um', fontsize=14)
			self.ax[0].set_ylabel('y/um', fontsize=14)
			self.ax[0].tick_params(axis='x', labelsize=14)
			self.ax[0].tick_params(axis='y', labelsize=14)

			# 第二个子图：z-x视图，横坐标z轴，纵坐标x轴
			self.ax[1].set_xlim(-z_range+z_bias, z_range+z_bias)
			self.ax[1].set_ylim(-x_range+x_bias, x_range+x_bias)
			self.ax[1].set_aspect('equal')
			self.ax[1].set_xlabel('z/um', fontsize=14)
			self.ax[1].set_ylabel('x/um', fontsize=14)
			self.ax[1].tick_params(axis='x', labelsize=14)
			self.ax[1].tick_params(axis='y', labelsize=14)

			# self.indices = np.arange(frame_init.r.shape[0])
			self.artists = (
				
				self.ax[0].scatter(frame_init.r[:, 2]*self.dl, frame_init.r[:, 1]*self.dl, 5, colors),
				self.ax[1].scatter(frame_init.r[:, 2]*self.dl, frame_init.r[:, 0]*self.dl, 5, colors),
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
				self.ax.scatter(frame_init.r[:, 0]*self.dl, frame_init.r[:, 1]*self.dl, 5, colors),
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

		# 添加同位素图例（只显示实际存在的同位素类型）
		if self.mass is not None and self.mass_indices is not None and not self.bilayer:
			unique_indices = np.unique(self.mass_indices)
			legend_labels = [self.isotope_labels[idx] for idx in unique_indices]
			legend_colors = [self.isotope_colors[idx] for idx in unique_indices]
			if legend_labels:
				ax_legend = self.ax[1] if self.fig_num == 2 else self.ax
				ax_legend.legend([plt.Line2D([0], [0], color=c, marker='o', linestyle='', markersize=8) 
					for c in legend_colors], legend_labels, loc='upper right', ncol=1, frameon=True, fontsize=12)

		time.sleep(0.5)

		self.bm = BlitManager(self.fig.canvas, self.artists)

		self.interval = interval
		self.count = 0

		# show_plot=False 时不弹出窗口，只用于收集并保存最后一帧
		if self.show_plot:
			plt.show(block=False)

	def plot(self, frame=None):
		"""
		绘制一帧数据
		:param frame: 可选的Frame对象，如果提供则直接使用，否则从队列获取
		"""
		# 只有在需要实时显示时才检查窗口是否关闭
		if self.show_plot and not plt.fignum_exists(self.fig.number):
			return False
		
		# 如果提供了frame参数，直接使用；否则从队列获取
		if frame is not None:
			f = frame
		else:
			self.count += 1
			if self.queue_in.empty():
				return True
			f: Frame = self.queue_in.get()
			if f is False:
				return False
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

		if self.bilayer:
			# bilayer模式：+为蓝色，-为红色
			if self.fig_num == 2:
				self.artists[0].set_offsets(np.vstack((f.r[:, 0]*self.dl, f.r[:, 2]*self.dl)).T)
				self.artists[1].set_offsets(np.vstack((f.r[:, 1]*self.dl, f.r[:, 2]*self.dl)).T)
			else:
				self.artists[0].set_offsets(np.vstack((f.r[:, 0]*self.dl, f.r[:, 2]*self.dl)).T)
			colors = np.full(f.r.shape[0], 'r')
			colors[f.r[:,1]>0] = 'b'
			if self.fig_num == 2:
				self.artists[0].set_facecolor(colors)
				self.artists[1].set_facecolor(colors)
			else:
				self.artists[0].set_facecolor(colors)
		else:
			# 非bilayer模式
			if self.fig_num == 2:
				# 第一个子图：z-y视图（横坐标z，纵坐标y）- 使用深度颜色映射
				self.artists[0].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 1]*self.dl)).T)
				norm = Normalize(vmin=np.min(f.r[:, 1]), vmax=np.max(f.r[:, 1]))
				cmap = cm.RdBu
				colors_ax0 = cmap(norm(f.r[:, 1]))
				self.artists[0].set_facecolor(colors_ax0)
				
				# 第二个子图：z-x视图（横坐标z，纵坐标x）- 使用同位素颜色标记
				self.artists[1].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 0]*self.dl)).T)
				if self.mass is not None and self.mass_indices is not None and len(self.mass_indices) == f.r.shape[0]:
					colors_ax1 = np.array([self.isotope_colors[idx] for idx in self.mass_indices])
				else:
					norm = Normalize(vmin=np.min(f.r[:, 0]), vmax=np.max(f.r[:, 0]))
					colors_ax1 = cm.RdBu(norm(f.r[:, 0]))
				self.artists[1].set_facecolor(colors_ax1)
				
				# 上下布局时，两个子图都显示标题
				self.ax[0].set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt), fontsize=14)
				self.ax[1].set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt), fontsize=14)
			elif self.fig_num == 1:
				# 单个子图：z-x视图（横坐标z，纵坐标x）- 使用同位素颜色标记
				self.artists[0].set_offsets(np.vstack((f.r[:, 2]*self.dl, f.r[:, 0]*self.dl)).T)
				if self.mass is not None and self.mass_indices is not None and len(self.mass_indices) == f.r.shape[0]:
					colors_ax = np.array([self.isotope_colors[idx] for idx in self.mass_indices])
				else:
					norm = Normalize(vmin=np.min(f.r[:, 0]), vmax=np.max(f.r[:, 0]))
					colors_ax = cm.RdBu(norm(f.r[:, 0]))
				self.artists[0].set_facecolor(colors_ax)
				self.ax.set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt), fontsize=14)
		
		# 每过10us输出一次时间
		time_us = f.timestamp * self.dt
		if time_us - self.last_output_time_us >= 10.0:
			print(f"Simulation Time: {time_us:.3f} μs")
			self.last_output_time_us = time_us
		
		np.save("../data_cache/r.npy", f.r*self.dl)

		if self.photos > 0.01:
			n_photo = f.timestamp*self.dt // self.photos
			if not os.path.exists(f"../data_cache/photos/{f.r.shape[0]:d}/{n_photo*self.photos:.10g}us.png"):
				self.fig.savefig(f"../data_cache/photos/{f.r.shape[0]:d}/{n_photo*self.photos:.10g}us.png")

		# 检查是否达到目标时间，如果是则保存图片并停止
		if self.target_time_dt is not None and f.timestamp >= self.target_time_dt and not self.final_frame_saved:
			if self.save_final_image is not None:
				# 确保目录存在
				image_dir = os.path.dirname(self.save_final_image)
				if image_dir:  # 如果路径包含目录
					os.makedirs(image_dir, exist_ok=True)
				self.fig.savefig(self.save_final_image, dpi=150, bbox_inches='tight')
				print(f"已保存最后一帧图片到: {self.save_final_image}")
				self.final_frame_saved = True
			# 发送停止信号
			self.queue_out.put(Message(CommandType.STOP))

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
