"""
Browse image-like files under ``Imgs/`` in an interactive window (English UI labels).

Run::

    python -m ImgSimulation.gallery
    python ImgSimulation/gallery.py
    python -m ImgSimulation.gallery --root /path/to/Imgs

With optional custom root (default: ``ImgSimulation/Imgs`` next to this package)::

    python -m ImgSimulation.gallery --root .

Controls:

- **Left / Right arrow** only — previous / next (no wrap at ends).
- **Home / End** — first / last image.
- **Slider** — jump by index.
- **Text box** — enter **0-based** index, then **Enter** to apply.
- **Eq pos** button — toggles turquoise ``+`` markers: loads ``pos_zx/<traj_stem>.npy``
  (time-averaged z–x in µm) and ``meta/<traj_stem>/<frame>.json`` (for ``imaging_json`` → camera);
  same geometry as the ion-image pipeline. Click again to hide markers.

Recursively lists ``.npy`` and common rasters: ``.png .jpg .jpeg .tif .tiff .webp``.
``.npy`` must be 2D (or squeezable to 2D) grayscale, or (H, W, 3|4) for RGB/RGBA.
Equilibrium overlay works only for **2D** frames whose path matches ``Imgs/<stem>/…`` (see ``--pos-zx`` / ``--meta``).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ``python ImgSimulation/gallery.py`` is not a package import; add repo root so
# ``ImgSimulation.*`` resolves. ``python -m ImgSimulation.gallery`` sets __package__.
if __name__ == "__main__" and not __package__:
    _repo = Path(__file__).resolve().parent.parent
    _r = str(_repo)
    if _r not in sys.path:
        sys.path.insert(0, _r)

import numpy as np

from ImgSimulation.geometry import world_um_to_fractional_col_row
from ImgSimulation.types import CameraParams
from ImgSimulation.visualize import _select_interactive_backend

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
)


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def default_imgs_root() -> Path:
    return _package_dir() / "Imgs"


def default_pos_zx_root() -> Path:
    return _package_dir() / "pos_zx"


def default_meta_root() -> Path:
    return _package_dir() / "meta"


def _traj_stem_for_image(path: Path, imgs_root: Path) -> str | None:
    """
    Map ``Imgs/<stem>/file`` → ``stem`` (matches ``pos_zx/<stem>.npy`` / traj NPZ).
    If the file lies directly under ``imgs_root``, use ``name_0001`` → ``name`` or full stem.
    """
    try:
        rel = path.resolve().relative_to(imgs_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    if len(parts) == 1:
        m = re.match(r"^(.+)_\d{4}$", path.stem)
        if m:
            return m.group(1)
        return path.stem
    return None


def _find_meta_json(path: Path, imgs_root: Path, meta_root: Path) -> Path | None:
    stem = _traj_stem_for_image(path, imgs_root)
    if not stem:
        return None
    exact = (meta_root / stem / f"{path.stem}.json").resolve()
    if exact.is_file():
        return exact
    sub = meta_root / stem
    if not sub.is_dir():
        return None
    cands = sorted(sub.glob("*.json"))
    return cands[0] if cands else None


def _camera_from_meta_json(meta_path: Path) -> CameraParams:
    from ImgSimulation.json_config import load_imaging_json

    raw = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    ij = raw.get("imaging_json")
    if not isinstance(ij, str) or not ij.strip():
        raise ValueError("meta has no imaging_json string")
    bundle = load_imaging_json(ij, project_root=None)
    return bundle.camera


def _list_image_paths(root: str | Path) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in _IMAGE_EXTENSIONS:
            out.append(p)
    out.sort(key=lambda x: str(x).lower())
    return out


def _prepare_array(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 0:
        raise ValueError("not an image (0-d)")
    if a.ndim == 1:
        raise ValueError("not an image (1-d)")
    if a.size == 0:
        raise ValueError("empty array")
    if a.ndim > 2:
        shp = a.shape
        if a.ndim == 3 and shp[2] in (3, 4):
            return a
        a = np.squeeze(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return a
    raise ValueError(f"expected 2D or HxWx3/4, got shape {a.shape!r}")


def load_image_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        import matplotlib.image as mpimg

        arr = mpimg.imread(str(path))
    return _prepare_array(np.asarray(arr))


def _rel_display(root: Path, path: Path) -> str:
    try:
        r = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.name
    s = str(r)
    if len(s) > 80:
        return "…" + s[-77:]
    return s


class ImageGallery:
    def __init__(
        self,
        paths: list[Path],
        *,
        root: Path,
        pos_zx_root: Path | None = None,
        meta_root: Path | None = None,
    ) -> None:
        self._paths = paths
        self._root = root.resolve()
        self._pos_zx_root = (pos_zx_root or default_pos_zx_root()).resolve()
        self._meta_root = (meta_root or default_meta_root()).resolve()
        self._n = len(paths)
        self._index = 0
        self._updating = False
        self._load_error: str | None = None
        self._show_eq: bool = False
        self._eq_error: str | None = None

        _select_interactive_backend()
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, TextBox

        self._plt = plt
        self.fig = plt.figure(figsize=(10, 7.0))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.ax = self.fig.add_axes((0.07, 0.24, 0.86, 0.66))
        self.ax_slider = self.fig.add_axes((0.1, 0.12, 0.48, 0.03))
        self.ax_eq = self.fig.add_axes((0.6, 0.102, 0.12, 0.04))
        self.ax_text = self.fig.add_axes((0.74, 0.11, 0.18, 0.035))

        self._slider = Slider(
            self.ax_slider,
            "index",
            0,
            max(0, self._n - 1),
            valinit=0,
            valstep=1,
        )
        self._slider.on_changed(self._on_slider)

        self._btn_eq = Button(self.ax_eq, "Eq pos", color="0.9", hovercolor="0.8")
        self._btn_eq.on_clicked(self._on_eq_clicked)

        self._textbox = TextBox(self.ax_text, "0-based index ", initial="0")
        self._textbox.on_submit(self._on_text_submit)

        self._draw_frame()

    def _on_key(self, event) -> None:
        if not event.key:
            return
        key = event.key
        if key == "left":
            self._set_index(self._index - 1)
        elif key == "right":
            self._set_index(self._index + 1)
        elif key == "home":
            self._set_index(0)
        elif key == "end":
            self._set_index(self._n - 1)

    def _on_eq_clicked(self, _event) -> None:
        self._show_eq = not self._show_eq
        if not self._show_eq:
            self._eq_error = None
        self._set_eq_button_label()
        self._draw_frame()

    def _set_eq_button_label(self) -> None:
        self._btn_eq.label.set_text("Hide eq" if self._show_eq else "Eq pos")

    def _on_slider(self, val) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            i = int(round(float(val)))
            i = max(0, min(self._n - 1, i))
            if i != self._index:
                self._index = i
                self._draw_frame()
            elif self._slider.val != i:
                self._slider.set_val(i)
        finally:
            self._updating = False

    def _on_text_submit(self, text: str) -> None:
        if self._updating:
            return
        s = text.strip()
        try:
            i = int(s, 10)
        except ValueError:
            self._load_error = f"invalid index: {text!r}"
            self._sync_widgets_only()
            self._set_suptitle()
            self.fig.canvas.draw_idle()
            return
        if i < 0 or i >= self._n:
            self._load_error = f"out of range: need 0..{self._n - 1}"
            self._sync_widgets_only()
            self._set_suptitle()
            self.fig.canvas.draw_idle()
            return
        self._load_error = None
        self._set_index(i)

    def _sync_widgets_only(self) -> None:
        self._updating = True
        try:
            self._textbox.set_val(str(self._index))
            if self._n > 0:
                self._slider.set_val(self._index)
        finally:
            self._updating = False

    def _set_index(self, i: int) -> None:
        if self._n == 0:
            return
        i = max(0, min(self._n - 1, i))
        if i == self._index and self._load_error is None:
            return
        self._load_error = None
        self._index = i
        self._draw_frame()

    def _set_suptitle(self) -> None:
        if self._n == 0:
            self.fig.suptitle("No images", fontsize=11)
            return
        p = self._paths[self._index]
        rel = _rel_display(self._root, p)
        one_based = self._index + 1
        err = f" — {self._load_error}" if self._load_error else ""
        eeq = f"  |  eq: {self._eq_error}" if self._eq_error else ""
        self.fig.suptitle(
            f"{one_based} / {self._n}  (0-based index {self._index})  —  {rel}{err}{eeq}",
            fontsize=10,
        )

    def _imshow_data(self, arr: np.ndarray) -> None:
        self.ax.clear()
        a = arr
        if a.ndim == 2:
            vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                self.ax.imshow(
                    a,
                    origin="lower",
                    cmap="gray",
                    aspect="equal",
                )
            else:
                self.ax.imshow(
                    a,
                    origin="lower",
                    cmap="gray",
                    aspect="equal",
                    vmin=vmin,
                    vmax=vmax,
                )
        else:
            if a.shape[2] == 4:
                a = a[..., :3]
            if a.dtype == np.floating and a.size and np.nanmax(a) <= 1.0 + 1e-6:
                a = np.clip(a, 0.0, 1.0)
            self.ax.imshow(a, origin="lower", aspect="equal")
        self.ax.set_xticks(())
        self.ax.set_yticks(())

    def _apply_eq_overlay(self, path: Path, arr: np.ndarray) -> None:
        self._eq_error = None
        if not self._show_eq:
            return
        if arr.ndim != 2:
            self._eq_error = "eq marks need 2D image"
            return
        stem = _traj_stem_for_image(path, self._root)
        if not stem:
            self._eq_error = "cannot infer traj stem (use Imgs/<stem>/file)"
            return
        pos_path = self._pos_zx_root / f"{stem}.npy"
        if not pos_path.is_file():
            self._eq_error = f"missing pos_zx: {pos_path.name}"
            return
        meta_path = _find_meta_json(path, self._root, self._meta_root)
        if not meta_path:
            self._eq_error = "no meta JSON for this frame"
            return
        try:
            cam = _camera_from_meta_json(meta_path)
        except (OSError, ValueError, KeyError, TypeError, FileNotFoundError) as e:
            self._eq_error = f"camera: {e}"
            return
        try:
            pos = np.load(pos_path, allow_pickle=False)
        except OSError as e:
            self._eq_error = str(e)
            return
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            self._eq_error = f"pos_zx must be (N, 2), got {pos.shape}"
            return
        if pos.shape[0] == 0:
            return
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if int(cam.h) != h or int(cam.l) != w:
            self._eq_error = f"image {h}x{w} != camera {cam.h}x{cam.l}"
            return
        col, row = world_um_to_fractional_col_row(
            pos[:, 0],
            pos[:, 1],
            x0_um=float(cam.x0_um),
            y0_um=float(cam.y0_um),
            pixel_um=float(cam.pixel_um),
            l=cam.l,
            h=cam.h,
        )
        self.ax.scatter(
            col,
            row,
            marker="+",
            s=120.0,
            c="cyan",
            linewidths=1.4,
            zorder=20,
        )

    def _draw_frame(self) -> None:
        if self._n == 0:
            return
        path = self._paths[self._index]
        self._updating = True
        try:
            try:
                arr = load_image_array(path)
                self._load_error = None
            except (OSError, ValueError) as e:
                self._load_error = str(e)
                self._eq_error = None
                self.ax.clear()
                self.ax.set_xticks(())
                self.ax.set_yticks(())
            else:
                self._imshow_data(arr)
                if self._show_eq:
                    self._apply_eq_overlay(path, arr)
            self._textbox.set_val(str(self._index))
            self._slider.set_val(self._index)
        finally:
            self._updating = False
        self._set_suptitle()
        self.fig.canvas.draw_idle()

    def show(self, *, block: bool = True) -> None:
        self._plt.show(block=block)


def run_gallery(
    root: str | Path | None = None,
    *,
    pos_zx_root: str | Path | None = None,
    meta_root: str | Path | None = None,
    block: bool = True,
) -> int:
    imgs = Path(root) if root is not None else default_imgs_root()
    paths = _list_image_paths(imgs)
    if not paths:
        print(
            f"No image files (extensions {sorted(_IMAGE_EXTENSIONS)}) under {imgs!s}",
            file=sys.stderr,
        )
        return 1
    pz = Path(pos_zx_root) if pos_zx_root is not None else None
    mr = Path(meta_root) if meta_root is not None else None
    g = ImageGallery(paths, root=imgs, pos_zx_root=pz, meta_root=mr)
    g.show(block=block)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preview images under Imgs/ recursively (npy, png, jpg, …).",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="image root (default: ImgSimulation/Imgs next to the package)",
    )
    p.add_argument(
        "--pos-zx",
        type=Path,
        default=None,
        dest="pos_zx",
        help="directory with pos_zx/<stem>.npy (default: ImgSimulation/pos_zx)",
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="meta root with <stem>/<frame>.json (default: ImgSimulation/meta)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_gallery(
        args.root,
        pos_zx_root=args.pos_zx,
        meta_root=args.meta,
        block=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
