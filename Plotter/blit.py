"""
BlitManager：基于 matplotlib blitting 的高效动画更新
参考 outline.md 与 ism-hybrid/utils.py 中的 BlitManager
适用于 Agg 后端，通过 copy_from_bbox / restore_region 实现快速重绘
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.backend_bases import FigureCanvasBase


class BlitManager:
    """
    管理一组 matplotlib Artist 的 blit 动画更新

    仅适用于 FigureCanvasAgg 及其子类（需有 copy_from_bbox、restore_region 方法）。
    在 draw_event 时捕获背景，update() 时仅重绘动画元素并 blit 到画布。
    """

    def __init__(
        self,
        canvas: FigureCanvasBase,
        animated_artists: Iterable[Artist] = (),
    ):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            画布，须为 Agg 子类，具备 copy_from_bbox、restore_region 方法
        animated_artists : Iterable[Artist]
            需要参与 blit 的 Artist 列表
        """
        self.canvas = canvas
        self._bg = None
        self._artists: list[Artist] = []

        for a in animated_artists:
            self.add_artist(a)

        self._cid = canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event) -> None:
        """draw_event 回调：捕获背景并绘制动画元素"""
        cv = self.canvas
        if event is not None and event.canvas != cv:
            raise RuntimeError("draw_event 来自非当前画布")
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art: Artist) -> None:
        """
        添加需管理的 Artist

        Parameters
        ----------
        art : Artist
            要管理的 Artist，须属于当前画布对应的 figure
        """
        if art.figure != self.canvas.figure:
            raise RuntimeError("Artist 须属于当前画布对应的 figure")
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self) -> None:
        """绘制所有动画 Artist"""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self) -> None:
        """
        更新显示：恢复背景，重绘动画元素，blit 到画布
        若尚未捕获背景则先触发 on_draw
        """
        cv = self.canvas
        fig = cv.figure
        if self._bg is None:
            self._on_draw(None)
        else:
            cv.restore_region(self._bg)
            self._draw_animated()
            cv.blit(fig.bbox)
        cv.flush_events()
