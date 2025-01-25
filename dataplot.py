import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

import ionsim

from utils import *

# Data calculating backend
class CalculationBackend:
	def __init__(self, step: int = 1000, interval: float = 1, batch: int = 10):
		"""
		:param step: number of steps to calculate in an interval
		:param interval: time interval
		:param batch: number of data to be sent each time
		"""
		self.step = step
		self.interval = interval
		self.batch = batch


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
			
			# start = time.process_time()
			r_list, v_list = ionsim.calculate_trajectory(
				r0, v0, 
				charge, mass,
				self.step * self.batch,
				t,
				t + self.interval * self.batch,
				force
			)
			# end = time.process_time()
			# print(f't = {t}, dt = {self.interval * self.batch}, elapsed = {end - start}')

			for i in range(self.batch):
				t += self.interval
				queue_out.put(Frame(
					r_list[(i + 1) * self.step - 1],
					v_list[(i + 1) * self.step - 1],
					t
				))

			r0 = r_list[-1]
			v0 = v_list[-1]

# Result Plotter
class DataPlotter:
	def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, frame_init : Frame, interval: float,ini_range:float =10,dl=1,dt=1):
		"""
		:param queue_in: input channel for data
		:param queue_out: not used (reserved)
		:param frame_init: initial frame
		:param interval: plotting interval. Should match data generating speed to avoid stuttering
		"""

		self.queue_in = queue_in
		self.queue_out = queue_out
		self.fig, self.ax = plt.subplots(1, 2, figsize=(15, 10))
		xyzmap=["x","y","z"]
		axisid=[[0,1],[0,2]]
		self.dl=dl
		self.dt=dt
		self.plot_range=ini_range*2*self.dl		
		for i in range(len(self.ax)):
			ax=self.ax[i]
			# ax.set_xlim(-self.plot_range, self.plot_range)
			# ax.set_ylim(-self.plot_range, self.plot_range)
			ax.set_xlim(-50, 50)
			ax.set_ylim(-200, 200)
			ax.set_xlabel(xyzmap[axisid[i][0]]+" (um)")
			ax.set_ylabel(xyzmap[axisid[i][1]]+" (um)")
			ax.set_aspect('equal')

		self.artists = (
			self.ax[0].plot(frame_init.r[:, 0]*self.dl, frame_init.r[:, 1]*self.dl, 'ro', animated=True)[0],
			self.ax[1].plot(frame_init.r[:, 0]*self.dl, frame_init.r[:, 2]*self.dl, 'ro', animated=True)[0],
		)
		self.ax[0]
		self.bm = BlitManager(self.fig.canvas, self.artists)

		self.interval = interval
		self.count = 0

		plt.show(block=False)

	def plot(self,plotFlag=True):
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
		if plotFlag:
		
				
			self.artists[0].set_data(f.r[:, 0]*self.dl, f.r[:, 1]*self.dl)
			self.artists[1].set_data(f.r[:, 0]*self.dl, f.r[:, 2]*self.dl)
			self.ax[1].set_title("timestamp=%.2f, t=%.3fus"%(f.timestamp,f.timestamp*self.dt))
			
			self.bm.update()
		if not plotFlag or plotFlag=="both":
			np.savetxt("./traj/%s.txt"%int(self.count),f.r)
		else:
			np.savetxt("./traj/r.txt",f.r)
		return True

	def start(self,plotFlag=True):
		print('starting plotter...')

		while True:
			if not self.plot(plotFlag=plotFlag):
				break
			if plotFlag:
				plt.pause(self.interval)

		print('stopping plotter...')
