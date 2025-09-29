from __future__ import annotations
import numpy
import typing
__all__ = ['Grid', 'calculate_trajectory']
M = typing.TypeVar("M", bound=int)

__version__: str = '0.0.4'

def calculate_trajectory(
        device: int,
        init_r: numpy.ndarray[tuple[M, typing.Literal[3]], numpy.dtype[numpy.float64]], 
        init_v: numpy.ndarray[tuple[M, typing.Literal[3]], numpy.dtype[numpy.float64]], 
        charge: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]], 
        mass: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]], 
        step: int, time_start: float, time_end: float, force: typing.Callable
    ) -> tuple[list[numpy.ndarray[tuple[M, typing.Literal[3]], numpy.dtype[numpy.float64]]], list[numpy.ndarray[tuple[M, typing.Literal[3]], numpy.dtype[numpy.float64]]]]:
    ...

class Grid:

    class GridCoord:
        px: float
        py: float
        pz: float
        x: int
        y: int
        z: int
        @staticmethod
        def _pybind11_conduit_v1_(*args, **kwargs):
            ...
        @staticmethod
        def from_numpy(a: numpy.ndarray[Grid.GridCoord]) -> list:
            ...
        @staticmethod
        def to_numpy(a: list) -> numpy.ndarray[Grid.GridCoord]:
            ...
        def __init__(self, x: int = 0, y: int = 0, z: int = 0, px: float = 0.0, py: float = 0.0, pz: float = 0.0) -> None:
            ...
    
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(
            self, 
            x: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], 
            y: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], 
            z: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], 
            value: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        ) -> None:
        ...
    def get_coord(self, xi: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        ...
    def in_bounds(self, xi: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        ...
    @typing.overload
    def interpolate(self, xi: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        ...
    @typing.overload
    def interpolate(self, xi: numpy.ndarray[Grid.GridCoord]) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        ...