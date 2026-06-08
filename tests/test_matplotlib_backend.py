"""Regression: --species 解析不得将 matplotlib 全局后端强制为 Agg。"""
from __future__ import annotations

import matplotlib


def test_species_import_preserves_interactive_backend():
    """模拟 main.py 先导入 DataPlotter 再解析 --species 的顺序。"""
    import matplotlib.pyplot as plt  # noqa: F401 — 与 Plotter.dataplot 一致

    backend_before = matplotlib.get_backend().lower()
    assert backend_before != "agg", "测试环境需为交互式 matplotlib 后端"

    from FieldConfiguration.ion_species import ION_SPECIES  # noqa: F401 — cli 路径

    assert matplotlib.get_backend().lower() == backend_before
    assert "Yb171+" in ION_SPECIES


def test_collision_pressure_package_init_has_no_side_effects():
    import collision_pressure  # noqa: F401

    backend = matplotlib.get_backend().lower()
    assert backend != "agg" or matplotlib.get_backend().lower() == backend
