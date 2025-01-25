import multiprocessing as mp
import numpy as np
from enum import Enum
from typing import Callable

import ionsim


class BlitManager:
	def __init__(self, canvas, animated_artists=()):
		"""
		Parameters
		----------
		canvas : FigureCanvasAgg
		The canvas to work with, this only works for subclasses of the Agg
		canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
		`~FigureCanvasAgg.restore_region` methods.

		animated_artists : Iterable[Artist]
		List of the artists to manage
		"""
		self.canvas = canvas
		self._bg = None
		self._artists = []

		for a in animated_artists:
			self.add_artist(a)
		# grab the background on every draw
		self.cid = canvas.mpl_connect("draw_event", self.on_draw)

	def on_draw(self, event):
		"""Callback to register with 'draw_event'."""
		cv = self.canvas
		if event is not None:
			if event.canvas != cv:
				raise RuntimeError
		self._bg = cv.copy_from_bbox(cv.figure.bbox)
		self._draw_animated()

	def add_artist(self, art):
		"""
		Add an artist to be managed.

		Parameters
		----------
		art : Artist

		The artist to be added.  Will be set to 'animated' (just
		to be safe).  *art* must be in the figure associated with
		the canvas this class is managing.

		"""
		if art.figure != self.canvas.figure:
			raise RuntimeError
		art.set_animated(True)
		self._artists.append(art)

	def _draw_animated(self):
		"""Draw all of the animated artists."""
		fig = self.canvas.figure
		for a in self._artists:
			fig.draw_artist(a)

	def update(self):
		"""Update the screen with animated artists."""
		cv = self.canvas
		fig = cv.figure
		# paranoia in case we missed the draw event,
		if self._bg is None:
			self.on_draw(None)
		else:
			# restore the background
			cv.restore_region(self._bg)
			# draw all of the animated artists
			self._draw_animated()
			# update the GUI state
			cv.blit(fig.bbox)
		# let the GUI event loop process anything it has to do
		cv.flush_events()

class Frame:
	def __init__(self, r: np.ndarray, v: np.ndarray, t: float):
		self.r = r
		self.v = v
		self.timestamp = t

	def kinetics(self):
		return np.sum(np.square(self.v)) * 0.5
class CommandType(Enum):
	START = 0
	PAUSE = 1
	RESUME = 2
	STOP = 3

class Message:
	def __init__(self):
		self.command: CommandType
		self.r: np.ndarray | None = None
		self.v: np.ndarray | None = None
		self.charge: np.ndarray | None = None
		self.mass: np.ndarray | None = None
		self.force: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None = None

class Producer:
	def __init__(self, step: int = 1000, interval: float = 1, batch: int = 10):
		self.step = step
		self.interval = interval
		self.batch = batch

	def run(self, queue_out: mp.Queue, queue_in: mp.Queue):
		# wait for start signal
		m: Message = queue_in.get()
		while m.command != CommandType.START:
			m = queue_in.get()
		
		assert(m.r is not None)
		assert(m.v is not None)
		assert(m.charge is not None)
		assert(m.mass is not None)
		assert(m.force is not None)

		r0 = m.r
		v0 = m.v
		charge = m.charge
		mass = m.mass
		force = m.force
		t: float = 0

		paused = False

		while(True):
			while paused or not queue_in.empty():
				m = queue_in.get()
				match m.command:
					case CommandType.START:
						break
					case CommandType.PAUSE:
						paused = True
					case CommandType.RESUME:
						paused = False
						if m.r:
							r0 = m.r
						if m.v:
							v0 = m.v
						if m.charge:
							charge = m.charge
						if m.mass:
							mass = m.mass
						if m.force:
							force = m.force
					case CommandType.STOP:
						return
			
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