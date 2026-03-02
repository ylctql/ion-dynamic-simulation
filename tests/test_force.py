"""force 构建模块测试"""
import numpy as np

from FieldConfiguration.field_settings import FieldSettings
from FieldParser.force import _zero_force, build_force


def test_build_force_empty_voltage_list_returns_zero_force():
    fs = FieldSettings(csv_filename="", voltage_list=[], g=0.1)
    grid_coord = np.random.rand(10, 3)
    grid_voltage = np.random.rand(10, 1)
    charge = np.ones(5)
    f = build_force(fs, grid_coord, grid_voltage, charge)
    assert f is _zero_force


def test_zero_force_returns_zeros():
    r = np.random.rand(5, 3)
    v = np.random.rand(5, 3)
    F = _zero_force(r, v, 0.0)
    np.testing.assert_array_equal(F, np.zeros_like(r))
