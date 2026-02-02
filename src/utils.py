import numpy as np
from enum import Enum
from typing import Callable
from scipy.spatial import Delaunay
import math

# For updating graphs

class Const:
	pi = math.pi
	Vrf = 550/2 #RF电压振幅
	freq_RF = 35.28 #RF射频频率@MHz
	Omega = freq_RF*2*pi*10**6 #RF射频角频率@SI
	epsl = 8.854*10**(-12)#真空介电常数@SI
	m = 137.327*(1.66053904*10**(-27))#离子的质量 @SI #Yb171=170.936323；Yb174=173.938859; Ba135=137.327
	ec = 1.6*10**(-19)#元电荷@SI
	dt = 2/Omega#单位时间dt #dt*T*Omega=2pi ->  T=pi
	dl = (( ec**2)/(4*pi*m*epsl*(Omega)**2))**(1/3)#单位长度dl
	dV = m/ec*(dl/dt)**2 #单位电压dV

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

# Data frame
class Frame:
	def __init__(self, r: np.ndarray, v: np.ndarray, t: float):
		self.r = r
		self.v = v
		self.timestamp = t

	def kinetics(self):
		return np.sum(np.square(self.v)) * 0.5
	
	def std_y(self):
		return self.r[:, 1].std() * Const.dl * 1e6
	
	def len_z(self):
		return np.abs(self.r[:, 2].max() - self.r[:, 2].min()) * Const.dl * 1e6
	
	def equi_cell(self) -> float:
		points = np.array([self.r[:, 0], self.r[:, 2]]).T
		delaunay = Delaunay(points)
		edges = set([e for n in delaunay.simplices for e in [(n[0], n[1]), (n[1], n[2]), (n[2], n[0])]])
		l_edges = [((points[e[0]][0] - points[e[1]][0]) ** 2 + (points[e[0]][1] - points[e[1]][1]) ** 2) ** 0.5 for e in edges]
		mean_edge = np.mean(l_edges)
		std_edge = np.std(l_edges)
		ratio = std_edge / mean_edge

		return float(ratio)
	
	def save_position(self, filename)->None:
		np.save(filename, self.r * Const.dl * 1e3)
	

class CommandType(Enum):
	START = 0
	PAUSE = 1
	RESUME = 2
	STOP = 3

class Message:
	def __init__(self, 
		command: CommandType,
		r: np.ndarray | None = None, 
		v: np.ndarray | None = None,
		charge: np.ndarray | None = None,
		mass: np.ndarray | None = None,
		force: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None = None
	):
		
		self.command = command
		self.r = r
		self.v = v
		self.charge = charge
		self.mass = mass
		self.force = force