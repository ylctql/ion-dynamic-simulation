"""
Display single-frame simulation images. Figure text is in English.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from .types import CameraParams


def _select_interactive_backend() -> None:
    """
    If the process is still on a non-interactive backend (e.g. ``Agg``), switch to
    a GUI backend *before* ``pyplot`` is imported so ``plt.show()`` can open a window.

    Tries, in order: ``TkAgg`` (often available via system tkinter), ``QtAgg``,
    ``macosx``. Falls back to ``Agg`` if no GUI can load. Does not override an
    already interactive backend (e.g. Jupyter inline).
    """
    import importlib

    import matplotlib

    b = matplotlib.get_backend().lower()
    if b not in ("agg", "template", "svg", "pdf", "ps"):
        return
    # Prefer Tcl/Tk (lighter dependency) over Qt, which may be missing in minimal envs.
    candidates: list[tuple[str, str]] = [
        ("TkAgg", "matplotlib.backends.backend_tkagg"),
        ("QtAgg", "matplotlib.backends.backend_qtagg"),
        ("MacOSX", "matplotlib.backends.backend_macosx"),
    ]
    for name, mod_name in candidates:
        try:
            importlib.import_module(mod_name)
            matplotlib.use(name, force=True)
            return
        except Exception:  # noqa: BLE001
            continue
    try:
        matplotlib.use("Agg", force=True)
    except Exception:  # noqa: BLE001
        pass


def show_ion_frame(
    image: np.ndarray,
    *,
    camera: CameraParams | None = None,
    title: str = "Ion image (simulation)",
    cmap: str = "gray",
    figsize: tuple[float, float] = (6.0, 5.5),
    show_colorbar: bool = True,
    block: bool = True,
    save_path: str | Path | None = None,
    dpi: float = 120.0,
) -> "Figure":
    """
    Show a single 2D frame in an **interactive** matplotlib window (default).

    The array must have shape (h, l): first axis row (y), second axis column (x),
    as produced by :func:`ImgSimulation.pipeline.render_single_frame`.

    If ``image`` is shown under the ``Agg`` backend (e.g. some headless runs), this
    function tries to switch to ``QtAgg`` / ``TkAgg`` / ``MacOSX`` when possible
    so a window can open. For notebooks, use ``%matplotlib inline`` and pass
    ``save_path`` or capture the returned figure.

    Parameters
    ----------
    block : bool
        Passed to :func:`matplotlib.pyplot.show`. ``True`` (default) keeps the
        window open until closed when running a script; use ``False`` in rare
        non-blocking cases.
    save_path : path-like, optional
        If set, also save the figure to this file (e.g. ``.png`` or ``.pdf``).

    Returns
    -------
    matplotlib.figure.Figure or None
        The ``matplotlib`` figure. If only ``Agg`` is available, no window will appear;
    set ``save_path`` to still persist the result.
    """
    # Backend must be set before pyplot; prefer a GUI so plt.show() opens a window.
    _select_interactive_backend()
    import matplotlib.pyplot as plt

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("image must be 2D with shape (h, l)")

    try:
        fig, ax = plt.subplots(figsize=figsize, layout="tight")
    except Exception:  # noqa: BLE001
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if camera is not None:
        px = float(camera.pixel_um)
        half_x = 0.5 * (camera.l - 1) * px
        half_y = 0.5 * (camera.h - 1) * px
        x0, y0 = float(camera.x0_um), float(camera.y0_um)
        extent = (x0 - half_x, x0 + half_x, y0 - half_y, y0 + half_y)
        im = ax.imshow(
            arr,
            origin="lower",
            cmap=cmap,
            aspect="equal",
            extent=extent,
        )
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    else:
        im = ax.imshow(
            arr,
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        ax.set_xlabel("Column index (x)")
        ax.set_ylabel("Row index (y)")

    ax.set_title(title)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Intensity (a.u.)", fraction=0.046, pad=0.04)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    plt.show(block=block)
    return fig
