"""collision_pressure 保存图片时使用的无头 matplotlib 后端（延迟设置，避免污染全局）。"""
from __future__ import annotations

_NON_INTERACTIVE = frozenset({"agg", "template", "svg", "pdf", "ps"})


def ensure_agg_backend() -> None:
    """仅在当前后端为交互式时切换到 Agg，供 savefig 使用。"""
    import matplotlib

    if matplotlib.get_backend().lower() in _NON_INTERACTIVE:
        return
    matplotlib.use("Agg", force=True)


def pyplot():
    """返回已配置 Agg 后端的 pyplot 模块。"""
    ensure_agg_backend()
    import matplotlib.pyplot as plt

    return plt
