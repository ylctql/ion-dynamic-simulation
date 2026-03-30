"""
离子晶格平衡构型求解入口

流程：
1) 从电场数据拟合 3D 四次外势 V(x,y,z)
2) 构建总势能 U_total = U_trap + U_coulomb
3) 通过最小化 U_total 求平衡位置
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_OUT_SENTINEL = "__default__"


def _parse_range_2(s: str, arg_name: str, parser: argparse.ArgumentParser) -> tuple[float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        parser.error(f"--{arg_name} 须为两个数: min,max")
    return (parts[0], parts[1])


def _parse_center(s: str, parser: argparse.ArgumentParser) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        parser.error("--center 须为 x,y,z 三个数")
    return (parts[0], parts[1], parts[2])


def _parse_hessian_slice_indices(
    expr: str | None,
    size: int,
    parser: argparse.ArgumentParser,
) -> np.ndarray:
    """
    将 1D numpy 风格切片字符串（支持逗号并集）解析为索引数组。

    支持示例：
    - ":"（全选）
    - "0:10"
    - "::3"
    - "5"（单个索引）
    - "0::3,2::3"（多个子集并集）
    """
    if size <= 0:
        parser.error("Hessian 维度必须为正")
    s = (expr or ":").strip()
    arr = np.arange(size, dtype=int)

    def _parse_one(token: str) -> np.ndarray:
        t = token.strip()
        if t == "":
            parser.error("--hessian-slice 包含空项；请检查逗号分隔格式")
        if t == ":":
            return arr
        if ":" in t:
            parts = t.split(":")
            if len(parts) > 3:
                parser.error(
                    "--hessian-slice 每一项仅支持 1D 切片，格式如 start:stop:step（可省略项）"
                )
            parts = parts + [""] * (3 - len(parts))

            def _maybe_int(v: str) -> int | None:
                return None if v.strip() == "" else int(v.strip())

            start = _maybe_int(parts[0])
            stop = _maybe_int(parts[1])
            step = _maybe_int(parts[2])
            if step == 0:
                parser.error("--hessian-slice 的 step 不能为 0")
            return arr[slice(start, stop, step)]
        return np.asarray([arr[int(t)]], dtype=int)

    try:
        tokens = [tok.strip() for tok in (":" if s == "" else s).split(",")]
        parts = [_parse_one(tok) for tok in tokens]
    except (ValueError, IndexError):
        parser.error(
            f'--hessian-slice="{s}" 解析失败；请使用如 ":"、"0:10"、"::3"、"5"、"0::3,2::3" 的格式'
        )

    seen = np.zeros(size, dtype=bool)
    ordered_unique: list[int] = []
    for part in parts:
        for i in np.asarray(part, dtype=int).ravel():
            ii = int(i)
            if not seen[ii]:
                seen[ii] = True
                ordered_unique.append(ii)

    idx = np.asarray(ordered_unique, dtype=int)
    if idx.size == 0:
        parser.error(f'--hessian-slice="{s}" 结果为空，请检查范围')
    return idx


def _slice_to_filename_part(s: str) -> str:
    """将 slice 字符串转换为文件名可用片段。"""
    t = (s or ":").strip().replace(" ", "")
    t = t.replace("/", "_").replace("\\", "_")
    return t


def _build_initial_positions(
    n_ions: int,
    x_range_um: tuple[float, float],
    y_range_um: tuple[float, float],
    z_range_um: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """默认初值：在给定 x/y/z 范围内均匀随机排布"""
    r = np.zeros((n_ions, 3), dtype=float)
    r[:, 0] = rng.uniform(x_range_um[0], x_range_um[1], size=n_ions)
    r[:, 1] = rng.uniform(y_range_um[0], y_range_um[1], size=n_ions)
    r[:, 2] = rng.uniform(z_range_um[0], z_range_um[1], size=n_ions)
    return r


def _plot_equilibrium_layout(
    r_um: np.ndarray,
    x_range_um: tuple[float, float],
    y_range_um: tuple[float, float],
    z_range_um: tuple[float, float],
    color_mode: str | None = None,
    charge_ec: np.ndarray | None = None,
    plot_out: str | None = None,
    point_size: float | None = None,
) -> None:
    """
    绘制平衡位置离子的空间分布：
    - 上子图：zoy（x 轴 z，y 轴 y）
    - 下子图：zox（x 轴 z，y 轴 x）
    """
    import matplotlib.pyplot as plt

    z = r_um[:, 2]
    y = r_um[:, 1]
    x = r_um[:, 0]

    mode = (color_mode or "none").lower()
    scatter_kwargs: dict = {"s": 15 if point_size is None else float(point_size), "alpha": 0.9}
    # 尽量对齐 Plotter 约定：none / y_pos / v2 / isotope
    if mode in ("none", ""):
        scatter_kwargs["c"] = "tab:red"
    elif mode == "y_pos":
        scatter_kwargs["c"] = y
        scatter_kwargs["cmap"] = "RdBu"
    elif mode == "v2":
        # 平衡位置图仅有坐标无速度，v2 模式不可用，降级为单色
        print("[plot] color_mode=v2 需速度信息，当前降级为 none")
        scatter_kwargs["c"] = "tab:red"
    elif mode == "isotope":
        # 当前流程无同位素质量数组，若提供 charge 则按电荷离散上色
        if charge_ec is not None and len(np.unique(charge_ec)) > 1:
            uq = np.unique(charge_ec)
            idx = np.searchsorted(uq, charge_ec)
            scatter_kwargs["c"] = idx
            scatter_kwargs["cmap"] = "tab10"
        else:
            print("[plot] color_mode=isotope 需同位素/多电荷信息，当前降级为 none")
            scatter_kwargs["c"] = "tab:red"
    else:
        print(f"[plot] 未识别 color_mode={color_mode}，当前降级为 none")
        scatter_kwargs["c"] = "tab:red"

    # 参考 Plotter 的布局风格：两行子图使用更高画布，避免元素拥挤
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # 上图：zoy
    sc_top = ax_top.scatter(z, y, **scatter_kwargs)
    ax_top.set_xlim(z_range_um[0], z_range_um[1])
    ax_top.set_ylim(y_range_um[0], y_range_um[1])
    ax_top.set_xlabel("z (μm)")
    ax_top.set_ylabel("y (μm)")
    ax_top.set_title("Equilibrium Layout: zoy")
    ax_top.set_aspect("equal")
    ax_top.grid(True, alpha=0.3)

    # 下图：zox
    ax_bottom.scatter(z, x, **scatter_kwargs)
    ax_bottom.set_xlim(z_range_um[0], z_range_um[1])
    ax_bottom.set_ylim(x_range_um[0], x_range_um[1])
    ax_bottom.set_xlabel("z (μm)")
    ax_bottom.set_ylabel("x (μm)")
    ax_bottom.set_title("Equilibrium Layout: zox")
    ax_bottom.set_aspect("equal")
    ax_bottom.grid(True, alpha=0.3)

    # 不使用 tight_layout，改为手动控制间距（与 Plotter 思路一致）
    fig.subplots_adjust(hspace=0.28, bottom=0.08, top=0.95)

    if plot_out:
        out_path = Path(plot_out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200)
        plt.close(fig)
        print(f"平衡位置图已保存: {out_path}")
    else:
        plt.show()


def _plot_hessian_heatmap(
    hessian_eV_per_um2: np.ndarray,
    hessian_kind: str,
    plot_out: str | None = None,
    show_plot: bool = True,
) -> None:
    """将 Hessian 矩阵以热力图形式可视化。"""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    h = np.asarray(hessian_eV_per_um2, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"Hessian 应为方阵，当前形状 {h.shape}")

    vmax = float(np.max(np.abs(h)))
    if vmax <= 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    hmin = float(np.min(h))
    hmax = float(np.max(h))
    if hmin < 0.0 < hmax:
        norm = colors.TwoSlopeNorm(vmin=hmin, vcenter=0.0, vmax=hmax)
        im = ax.imshow(h, cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
    else:
        im = ax.imshow(h, cmap="viridis", origin="lower", aspect="auto", vmin=hmin, vmax=hmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Hessian (eV/μm²)")
    ax.set_xlabel("DOF index")
    ax.set_ylabel("DOF index")
    ax.set_title(f"Hessian Heatmap ({hessian_kind})")
    fig.tight_layout()

    if plot_out:
        out_path = Path(plot_out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=220)
        print(f"Hessian 热力图已保存: {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_phonon_spectrum(
    freq_mhz: np.ndarray,
    mode: str = "frequency",
    plot_out: str | None = None,
    show_plot: bool = True,
) -> None:
    """
    绘制声子频谱（线频率，MHz）。

    mode:
    - "frequency": 横轴频率（MHz），纵轴离散谱线强度
    - "index": 横轴模式 index，纵轴频率（MHz，按降序）
    """
    import matplotlib.pyplot as plt

    f = np.asarray(freq_mhz, dtype=float).ravel()
    if f.size == 0:
        raise ValueError("频率数组为空，无法绘制声子频谱")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    stable_mask = f >= 0.0
    unstable_mask = ~stable_mask
    if mode == "index":
        x = np.arange(f.size, dtype=int)
        ax.axhline(0.0, color="k", linewidth=0.8)
        if np.any(stable_mask):
            ax.vlines(
                x[stable_mask],
                0.0,
                f[stable_mask],
                colors="tab:blue",
                linewidth=1.2,
                label="stable",
            )
        if np.any(unstable_mask):
            ax.vlines(
                x[unstable_mask],
                0.0,
                f[unstable_mask],
                colors="tab:red",
                linewidth=1.2,
                label="unstable",
            )
        ax.set_xlabel("Mode index (frequency-desc order)")
        ax.set_ylabel("Phonon frequency (MHz)")
        ax.set_title("Phonon Spectrum (index mode)")
    else:
        x = f
        y0 = np.zeros_like(f)
        y1 = np.ones_like(f)
        if np.any(stable_mask):
            ax.vlines(
                x[stable_mask],
                y0[stable_mask],
                y1[stable_mask],
                colors="tab:blue",
                linewidth=1.0,
                label="stable",
            )
        if np.any(unstable_mask):
            ax.vlines(
                x[unstable_mask],
                y0[unstable_mask],
                y1[unstable_mask],
                colors="tab:red",
                linewidth=1.0,
                label="unstable",
            )
        ax.set_xlabel("Phonon frequency (MHz)")
        ax.set_ylabel("Spectral line")
        ax.set_yticks([])
        ax.set_title("Phonon Spectrum (frequency mode)")

    ax.grid(True, alpha=0.3)
    if np.any(unstable_mask):
        ax.legend(loc="best")
    fig.tight_layout()

    if plot_out:
        out_path = Path(plot_out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=220)
        print(f"声子频谱图已保存: {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_phonon_mode_vector_zox(
    r_um: np.ndarray,
    eigvec_cartesian: np.ndarray,
    dof_indices: np.ndarray,
    mode_index: int,
    freq_hz_signed_all: np.ndarray | None = None,
    arrow_scale_factor: float = 1.0,
    plot_out: str | None = None,
    show_plot: bool = True,
    point_size: float | None = None,
    z_range_um: tuple[float, float] | None = None,
    x_range_um: tuple[float, float] | None = None,
) -> None:
    """
    绘制指定声子模式在 zox 平面上的本征向量分布。

    - 散点位置：离子平衡位置投影到 zox 平面（横轴 z，纵轴 x）
    - 箭头方向/长度：该模式在 (z, x) 分量
    - 颜色：该模式在该离子上的三维分量模长
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox

    r = np.asarray(r_um, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"r_um 应为 (N,3)，当前 {r.shape}")
    n_ions = r.shape[0]
    n_dof = 3 * n_ions

    eig = np.asarray(eigvec_cartesian, dtype=float)
    if eig.ndim != 2:
        raise ValueError(f"eigvec_cartesian 应为二维数组，当前形状 {eig.shape}")
    n_modes = int(eig.shape[1])
    if mode_index < 0 or mode_index >= n_modes:
        raise ValueError(
            f"mode_index={mode_index} 超出范围 [0, {n_modes - 1}]"
        )

    idx = np.asarray(dof_indices, dtype=int).ravel()
    if idx.size != eig.shape[0]:
        raise ValueError(
            f"dof_indices 长度 {idx.size} 与 eigvec_cartesian 行数 {eig.shape[0]} 不一致"
        )

    z = r[:, 2]
    x = r[:, 0]
    if arrow_scale_factor <= 0.0:
        raise ValueError("arrow_scale_factor 必须为正数")

    if z_range_um is not None and x_range_um is not None:
        z_lim = (float(z_range_um[0]), float(z_range_um[1]))
        x_lim = (float(x_range_um[0]), float(x_range_um[1]))
    else:
        _z_pad = max(1e-9, 0.08 * (float(np.max(z)) - float(np.min(z)) + 1e-12))
        _x_pad = max(1e-9, 0.08 * (float(np.max(x)) - float(np.min(x)) + 1e-12))
        z_lim = (float(np.min(z)) - _z_pad, float(np.max(z)) + _z_pad)
        x_lim = (float(np.min(x)) - _x_pad, float(np.max(x)) + _x_pad)
    z_span = max(z_lim[1] - z_lim[0], 1e-12)
    x_span = max(x_lim[1] - x_lim[0], 1e-12)
    ref_span = max(z_span, x_span, 1e-12)
    target_max_len = 0.08 * ref_span * float(arrow_scale_factor)

    freq_all = None
    if freq_hz_signed_all is not None:
        freq_all = np.asarray(freq_hz_signed_all, dtype=float).ravel()
        if freq_all.size != n_modes:
            raise ValueError(
                f"freq_hz_signed_all 长度 {freq_all.size} 与 mode 数 {n_modes} 不一致"
            )

    def _compute_mode_data(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, bool]:
        # 子空间本征向量还原到完整 3N 自由度；未纳入子空间的分量置零
        mode_full = np.zeros(n_dof, dtype=float)
        mode_full[idx] = eig[:, k]
        mode_xyz = mode_full.reshape(n_ions, 3)
        vz_local = mode_xyz[:, 2]
        vx_local = mode_xyz[:, 0]
        amp = np.linalg.norm(mode_xyz, axis=1)
        amp_max = float(np.max(amp))
        amp_norm_local = (amp / amp_max) if amp_max > 0.0 else amp
        plane_amp = np.sqrt(vz_local * vz_local + vx_local * vx_local)
        plane_max = float(np.max(plane_amp))
        has_plane_vector = bool(np.isfinite(plane_max) and (plane_max > 0.0))
        if has_plane_vector and np.isfinite(target_max_len) and target_max_len > 0.0:
            quiver_scale_local = plane_max / target_max_len
            if not np.isfinite(quiver_scale_local) or quiver_scale_local <= 0.0:
                quiver_scale_local = None
        else:
            quiver_scale_local = None
        return vz_local, vx_local, amp_norm_local, quiver_scale_local, has_plane_vector

    def _mode_title(k: int) -> str:
        if freq_all is not None:
            f_mhz = float(freq_all[k]) / 1e6
            return f"Phonon Mode Vector (zox), mode={k}, f={f_mhz:+.6f} MHz"
        return f"Phonon Mode Vector (zox), mode={k}"

    current_mode = int(mode_index)
    vz, vx, amp_norm, quiver_scale, has_plane_vector = _compute_mode_data(current_mode)
    _ratio = z_span / x_span
    _max_dim = 9.0
    _figw = _max_dim if _ratio >= 1.0 else max(3.0, _max_dim * _ratio)
    _figh = _max_dim if _ratio < 1.0 else max(3.0, _max_dim / _ratio)
    fig, ax = plt.subplots(1, 1, figsize=(_figw, _figh))
    scatter_size = 15 if point_size is None else float(point_size)
    sc = ax.scatter(z, x, c=amp_norm, cmap="viridis", s=scatter_size, alpha=0.95)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mode amplitude (normalized)")
    quiver_artist = None
    if has_plane_vector:
        quiver_artist = ax.quiver(
            z,
            x,
            vz,
            vx,
            color="red",
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            width=0.003,
            alpha=0.9,
            zorder=3,
        )
    ax.set_xlim(*z_lim)
    ax.set_ylim(*x_lim)
    ax.set_xlabel("z (μm)")
    ax.set_ylabel("x (μm)")
    ax.set_title(_mode_title(current_mode))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if plot_out:
        out_path = Path(plot_out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=220)
        print(f"声子模式向量图已保存: {out_path}")
    if not show_plot:
        plt.close(fig)
        return

    fig.subplots_adjust(bottom=0.26)

    slider_ax = fig.add_axes([0.14, 0.145, 0.74, 0.045])
    mode_slider = Slider(
        ax=slider_ax,
        label="mode index",
        valmin=0,
        valmax=n_modes - 1,
        valinit=current_mode,
        valstep=1,
    )
    box_ax = fig.add_axes([0.14, 0.055, 0.26, 0.06])
    mode_text = TextBox(box_ax, "mode index", initial=str(current_mode))
    status_ax = fig.add_axes([0.43, 0.055, 0.45, 0.06])
    status_ax.axis("off")
    status_text = status_ax.text(
        0.0,
        0.5,
        "Hint: drag slider / press Enter after index input / use left-right keys",
        va="center",
        ha="left",
        fontsize=10,
        color="dimgray",
    )
    sync_guard = False

    def _set_status(msg: str) -> None:
        color = "dimgray" if msg.startswith("Hint:") else "crimson"
        status_text.set_color(color)
        status_text.set_text(msg)

    def _refresh_mode(k: int, *, sync_widgets: bool = True) -> None:
        nonlocal current_mode, quiver_artist, sync_guard
        vz_local, vx_local, amp_norm_local, quiver_scale_local, has_plane_vector_local = _compute_mode_data(k)
        sc.set_array(amp_norm_local)
        if quiver_artist is not None:
            quiver_artist.remove()
            quiver_artist = None
        if has_plane_vector_local:
            quiver_artist = ax.quiver(
                z,
                x,
                vz_local,
                vx_local,
                color="red",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale_local,
                width=0.003,
                alpha=0.9,
                zorder=3,
            )
        ax.set_title(_mode_title(k))
        current_mode = k
        if sync_widgets and not sync_guard:
            sync_guard = True
            try:
                if int(round(mode_slider.val)) != k:
                    mode_slider.set_val(k)
                if mode_text.text.strip() != str(k):
                    mode_text.set_val(str(k))
            finally:
                sync_guard = False
        if has_plane_vector_local:
            _set_status("Hint: drag slider / press Enter after index input / use left-right keys")
        else:
            _set_status("Hint: this mode has zero in-plane (z,x) vector; no arrows to draw")
        fig.canvas.draw_idle()

    def _on_submit(text: str) -> None:
        if sync_guard:
            return
        raw = text.strip()
        try:
            k = int(raw)
        except ValueError:
            _set_status(f"Invalid input: {raw!r} (must be an integer)")
            return
        if not (0 <= k < n_modes):
            _set_status(f"Mode out of range [0, {n_modes - 1}]")
            return
        _refresh_mode(k)

    def _on_slider(val: float) -> None:
        if sync_guard:
            return
        k = int(round(float(val)))
        if 0 <= k < n_modes and k != current_mode:
            _refresh_mode(k)

    def _on_key(event) -> None:
        if event.inaxes == box_ax:
            return
        if event.key == "left":
            _refresh_mode((current_mode - 1) % n_modes)
        elif event.key == "right":
            _refresh_mode((current_mode + 1) % n_modes)

    mode_slider.on_changed(_on_slider)
    mode_text.on_submit(_on_submit)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="最小化总势能（外势+库伦）以求离子晶格平衡位置"
    )
    parser.add_argument("--csv", type=str, default="", help="电场 CSV 路径")
    parser.add_argument("--config", type=str, default="", help="电压配置 JSON 路径")
    parser.add_argument("--N", type=int, default=20, help="离子数，默认 20")
    parser.add_argument("--charge", type=float, default=1.0, help="每个离子的电荷（单位 e），默认 +1")
    parser.add_argument(
        "--center",
        type=str,
        default="0,0,0",
        help="参考中心 x,y,z (μm)",
    )
    parser.add_argument("--x_range", type=str, default="-50,50", help="x 轴拟合/优化范围 (μm)")
    parser.add_argument("--y_range", type=str, default="-20,20", help="y 轴拟合/优化范围 (μm)")
    parser.add_argument("--z_range", type=str, default="-150,150", help="z 轴拟合/优化范围 (μm)")
    parser.add_argument("--fit-n-pts-x", type=int, default=100, help="3D 拟合 x 轴采样点数，默认 100")
    parser.add_argument("--fit-n-pts-y", type=int, default=40, help="3D 拟合 y 轴采样点数，默认 40")
    parser.add_argument("--fit-n-pts-z", type=int, default=300, help="3D 拟合 z 轴采样点数，默认 300")
    parser.add_argument(
        "--fit-mode",
        type=str,
        default="none",
        choices=["none", "even", "quartic", "quartic_even", "quadratic"],
        help=(
            "3D 势拟合：none=各变量≤4 张量积 125 项；even=其上删奇次指数 27 项；"
            "quartic=总次数≤4 共 35 项；quartic_even=quartic 全偶 10 项；quadratic=4 项"
        ),
    )
    parser.add_argument("--softening-um", type=float, default=0.001, help="库伦软化长度 (μm)，默认 0.001")
    parser.add_argument("--phonon", action="store_true", help="在平衡位置处计算 Hessian 并提取声子模式")
    parser.add_argument(
        "--mass-amu",
        type=float,
        default=135.0,
        help="--phonon 时离子质量（amu，默认 135.0，即 Ba135）",
    )
    parser.add_argument(
        "--phonon-print-modes",
        type=int,
        default=10,
        help="--phonon 时打印前若干个本征频率（按频率降序），默认 10",
    )
    parser.add_argument(
        "--plot-hessian",
        nargs="?",
        const="total",
        default=None,
        choices=["total", "trap", "coulomb"],
        help='绘制 Hessian 热力图；不带值默认 total，也可指定 "trap" 或 "coulomb"',
    )
    parser.add_argument(
        "--hessian-slice",
        type=str,
        default=":",
        help='Hessian 子矩阵索引（支持逗号并集），如 ":"、"0:10"、"::3"、"5"、"0::3,2::3"',
    )
    parser.add_argument(
        "--plot-hessian-out",
        nargs="?",
        const=_DEFAULT_OUT_SENTINEL,
        default=None,
        help="Hessian 热力图输出图片路径；仅指定该选项但不传路径时，默认保存到 equilibrium/results/hessian_plot/{N}_{slice}.png",
    )
    parser.add_argument(
        "--plot-phonon-spectrum",
        nargs="?",
        const="frequency",
        default=None,
        choices=["frequency", "index"],
        help='绘制声子频谱；默认 frequency（横轴频率，离散谱线），可选 "index"',
    )
    parser.add_argument(
        "--plot-phonon-spectrum-out",
        nargs="?",
        const=_DEFAULT_OUT_SENTINEL,
        default=None,
        help="声子频谱输出图片路径；仅指定该选项但不传路径时，默认保存到 equilibrium/results/spectra/{N}_{slice}.png",
    )
    parser.add_argument(
        "--plot-mode-vector",
        nargs="?",
        const=0,
        type=int,
        default=None,
        help="绘制指定声子模式的本征向量（zox 平面）；不带值默认 0（按频率降序）",
    )
    parser.add_argument(
        "--plot-mode-vector-out",
        nargs="?",
        const=_DEFAULT_OUT_SENTINEL,
        default=None,
        help="模式向量图输出图片路径；仅指定该选项但不传路径时，默认保存到 equilibrium/results/mode_vector/{N}_{slice}_mode{k}.png",
    )
    parser.add_argument(
        "--plot-mode-vector-arrow-scale",
        type=float,
        default=1.0,
        help="--plot-mode-vector 箭头长度倍率（>0，默认 1.0）",
    )
    parser.add_argument(
        "--plot-point-size",
        type=float,
        default=None,
        help="绘图散点大小（可同时作用于 --plot 与 --plot-mode-vector；不传则二者均默认 15）",
    )
    parser.add_argument(
        "--save-hessian-data",
        action="store_true",
        help="保存 Hessian 数据（total/trap/coulomb）到 npz",
    )
    parser.add_argument(
        "--hessian-data-out",
        type=str,
        default="",
        help="Hessian 数据输出 npz 路径；不传则默认保存到 equilibrium/results/hessian_data/{N}_{slice}.npz",
    )
    parser.add_argument("--maxiter", type=int, default=500, help="优化最大迭代步数，默认 500")
    parser.add_argument("--tol", type=float, default=1e-10, help="优化收敛阈值，默认 1e-10")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--init_file", type=str, default="", help="初值 npz 路径（含 r，shape=(N,3)，单位 μm）")
    parser.add_argument("--plot", action="store_true", help="绘制平衡位置空间分布（上 zoy，下 zox）")
    parser.add_argument(
        "--color_mode",
        type=str,
        default="none",
        choices=["none", "y_pos", "v2", "isotope"],
        help="绘图颜色模式：none / y_pos / v2 / isotope，默认 none",
    )
    parser.add_argument("--plot-out", type=str, default="", help="--plot 时输出图片路径；不传则弹窗显示")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 npz 路径（保存平衡位置与能量）；不传则保存到 equilibrium/results/equi_pos/{N}.npz",
    )
    parser.add_argument("--smooth-axes", type=str, default="z", help="势场平滑方向，默认 z；none 关闭")
    parser.add_argument("--smooth-sg", type=str, default="11,3", help="Savitzky-Golay 参数 window,poly，默认 11,3")
    args = parser.parse_args()
    if args.plot_point_size is not None and args.plot_point_size <= 0.0:
        parser.error("--plot-point-size 必须为正数")

    from Interface.cli import (
        DEFAULT_CONFIG_DIR,
        DEFAULT_CONFIG_PATH,
        DEFAULT_CSV_DIR,
        DEFAULT_CSV_PATH,
    )
    from FieldConfiguration.constants import init_from_config
    from FieldConfiguration.loader import build_voltage_list, field_settings_from_config
    from FieldParser.calc_field import calc_field, calc_potential
    from FieldParser.csv_reader import read as read_csv
    from field_visualize.core import apply_savgol_smooth, compute_potentials, um_to_norm

    from equilibrium.energy import total_energy_and_grad
    from equilibrium.phonon import solve_phonon_modes, total_hessian
    from equilibrium.potential_fit_3d import (
        fit_potential_3d_quartic,
        write_potential_fit_coeff_json,
    )

    def _resolve_path(arg: str, default_full: str, default_dir: str) -> str:
        if not arg:
            return str(_ROOT / default_full)
        p = Path(arg)
        if not p.is_absolute() and "/" not in arg and "\\" not in arg:
            return str(_ROOT / default_dir / arg)
        return str(_ROOT / arg) if not p.is_absolute() else arg

    config_path = _resolve_path(args.config, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_DIR)
    csv_path = _resolve_path(args.csv, DEFAULT_CSV_PATH, DEFAULT_CSV_DIR)

    cfg, config = init_from_config(config_path)
    grid_coord, grid_voltage = read_csv(csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV)
    n_voltage = grid_voltage.shape[1]
    if config:
        field_settings = field_settings_from_config(csv_path, config_path, n_voltage, cfg)
    else:
        from FieldConfiguration.field_settings import FieldSettings

        field_settings = FieldSettings(csv_filename=csv_path, voltage_list=[])
        field_settings.voltage_list = build_voltage_list({"voltage_list": []}, n_voltage, cfg)

    if args.smooth_axes.strip().lower() != "none":
        axes = tuple(a.strip().lower() for a in args.smooth_axes.split(",") if a.strip() in "xyz")
        if axes:
            parts = [p.strip() for p in args.smooth_sg.split(",")]
            wl = int(parts[0]) if parts else 11
            poly = int(parts[1]) if len(parts) >= 2 else 3
            grid_voltage = apply_savgol_smooth(grid_coord, grid_voltage, axes, window_length=wl, polyorder=poly)

    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)
    voltage_list = field_settings.voltage_list

    def compute_V_total(r_norm: np.ndarray) -> np.ndarray:
        _, _, _, v_total = compute_potentials(potential_interps, field_interps, voltage_list, cfg, r_norm)
        return v_total

    v_grid_all = np.asarray(compute_V_total(grid_coord), dtype=float).ravel()
    v_grid_valid = v_grid_all[np.isfinite(v_grid_all)]
    if v_grid_valid.size == 0:
        parser.error("格点总势场全为非有限值，无法确定统一势能零点")
    v_min_grid = float(np.min(v_grid_valid))

    center_um = _parse_center(args.center, parser)
    x_range = _parse_range_2(args.x_range, "x_range", parser)
    y_range = _parse_range_2(args.y_range, "y_range", parser)
    z_range = _parse_range_2(args.z_range, "z_range", parser)
    range_um = (x_range, y_range, z_range)

    fit = fit_potential_3d_quartic(
        compute_V_total=compute_V_total,
        um_to_norm=lambda v: um_to_norm(v, cfg.dl),
        center_um=center_um,
        range_um=range_um,
        n_pts_per_axis=(args.fit_n_pts_x, args.fit_n_pts_y, args.fit_n_pts_z),
        potential_offset_V=v_min_grid,
        fit_mode=args.fit_mode,
    )
    write_potential_fit_coeff_json(
        fit,
        _ROOT / "equilibrium" / "results" / "potential_fit_coeff.json",
        csv=csv_path,
        config=config_path,
    )

    n_ions = int(args.N)
    if n_ions <= 0:
        parser.error("--N 必须为正整数")
    n_dof = 3 * n_ions
    hessian_dof_indices = _parse_hessian_slice_indices(args.hessian_slice, n_dof, parser)
    slice_part = _slice_to_filename_part(args.hessian_slice)
    default_stem = f"{n_ions}_{slice_part}"
    print(
        f'Hessian 子空间: slice="{args.hessian_slice}" -> '
        f"{hessian_dof_indices.size}/{n_dof} 个自由度"
    )
    charge_ec = np.full(n_ions, float(args.charge), dtype=float)

    if args.init_file:
        init_path = Path(args.init_file)
        if not init_path.is_absolute():
            init_path = _ROOT / init_path
        arr = np.load(str(init_path))
        if "r" not in arr:
            parser.error("--init_file 文件需包含键 'r'，shape=(N,3)，单位 μm")
        r0 = np.asarray(arr["r"], dtype=float)
        if r0.shape != (n_ions, 3):
            parser.error(f"init r 形状应为 ({n_ions},3)，当前 {r0.shape}")
    else:
        rng = np.random.default_rng(args.seed)
        r0 = _build_initial_positions(
            n_ions=n_ions,
            x_range_um=x_range,
            y_range_um=y_range,
            z_range_um=z_range,
            rng=rng,
        )

    bounds = [x_range, y_range, z_range] * n_ions

    def _reshape(x_flat: np.ndarray) -> np.ndarray:
        return x_flat.reshape(n_ions, 3)

    def objective(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
        r = _reshape(x_flat)
        breakdown, grad = total_energy_and_grad(
            fit=fit,
            r_um=r,
            charge_ec=charge_ec,
            softening_um=float(args.softening_um),
        )
        return breakdown.total_eV, grad.ravel()

    e0, g0 = objective(r0.ravel())
    print("=" * 68)
    print("离子晶格平衡构型求解")
    print("=" * 68)
    print(f"N = {n_ions}, q = {args.charge:+.3f} e")
    fit_mode_disp = fit.fit_mode if fit.fit_mode else "none"
    print(
        f"拟合 R² = {fit.r_squared:.6f}, scale L = {fit.scale_um:.1f} μm, "
        f"fit_mode={fit_mode_disp}（{len(fit.basis_exps)} 项）"
    )
    print(f"势能零点平移: V_shifted = V_true - V_min_grid = V_true - ({fit.potential_offset_V:.6e} V)")
    print(f"初始总能量: {e0:.6e} eV")

    res = minimize(
        fun=lambda x: objective(x)[0],
        x0=r0.ravel(),
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(args.maxiter), "ftol": float(args.tol)},
    )

    r_eq = _reshape(res.x)
    e_eq, g_eq = objective(res.x)
    breakdown_eq, _ = total_energy_and_grad(
        fit=fit,
        r_um=r_eq,
        charge_ec=charge_ec,
        softening_um=float(args.softening_um),
    )
    grad_norm = float(np.linalg.norm(g_eq))
    y_min = float(np.min(r_eq[:, 1]))
    y_max = float(np.max(r_eq[:, 1]))
    y_span = y_max - y_min

    print(f"优化状态: success={res.success}, status={res.status}")
    print(f"message: {res.message}")
    print(f"迭代: nit={res.nit}, nfev={res.nfev}, njev={getattr(res, 'njev', -1)}")
    print(f"终态总能量: {e_eq:.6e} eV")
    print(f"终态梯度范数: {grad_norm:.6e} eV/μm")
    print(f"  其中 U_trap={breakdown_eq.trap_eV:.6e} eV")
    print(f"      U_coul={breakdown_eq.coulomb_eV:.6e} eV")
    print(f"y 范围: [{y_min:.3f}, {y_max:.3f}] μm (span={y_span:.3f} μm)")
    print("=" * 68)

    phonon = None
    need_phonon = bool(
        args.phonon
        or (args.plot_phonon_spectrum is not None)
        or (args.plot_phonon_spectrum_out is not None)
        or (args.plot_mode_vector is not None)
        or (args.plot_mode_vector_out is not None)
    )
    if need_phonon:
        phonon = solve_phonon_modes(
            fit=fit,
            r_um=r_eq,
            charge_ec=charge_ec,
            mass_amu=float(args.mass_amu),
            softening_um=float(args.softening_um),
            dof_indices=hessian_dof_indices,
        )
        if args.phonon:
            freq_mhz = phonon.freq_hz_signed / 1e6
            unstable = int(np.sum(phonon.omega2_s2 < 0.0))
            print("声子模式分析:")
            print(f"  mass = {float(args.mass_amu):.6f} amu")
            print(f"  模式数 = {freq_mhz.size}, 不稳定模数(omega^2<0) = {unstable}")
            n_print = max(0, min(int(args.phonon_print_modes), freq_mhz.size))
            for i in range(n_print):
                print(
                    f"  mode {i:03d}: "
                    f"f = {freq_mhz[i]:+.6f} MHz, "
                    f"omega^2 = {phonon.omega2_s2[i]:+.6e} s^-2"
                )
            print("=" * 68)

    if args.out.strip():
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
    else:
        out_path = _ROOT / "equilibrium" / "results" / "equi_pos" / f"{n_ions}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "r": r_eq,
        "r0": r0,
        "total_energy_eV": e_eq,
        "trap_energy_eV": float(breakdown_eq.trap_eV),
        "coulomb_energy_eV": float(breakdown_eq.coulomb_eV),
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "fit_r2": float(fit.r_squared),
        "center_um": np.array(center_um, dtype=float),
        "x_range_um": np.array(x_range, dtype=float),
        "y_range_um": np.array(y_range, dtype=float),
        "z_range_um": np.array(z_range, dtype=float),
        "hessian_slice_expr": np.array(args.hessian_slice),
        "hessian_dof_indices": hessian_dof_indices,
    }
    if phonon is not None:
        save_data.update(
            {
                "phonon_mass_amu": float(args.mass_amu),
                "hessian_total_eV_per_um2": phonon.hessian_total_eV_per_um2,
                "hessian_trap_eV_per_um2": phonon.hessian_trap_eV_per_um2,
                "hessian_coulomb_eV_per_um2": phonon.hessian_coulomb_eV_per_um2,
                "dynamical_matrix_s2": phonon.dynamical_matrix_s2,
                "omega2_s2": phonon.omega2_s2,
                "freq_hz_signed": phonon.freq_hz_signed,
                "eigvec_mass_weighted": phonon.eigvec_mass_weighted,
                "eigvec_cartesian": phonon.eigvec_cartesian,
                "phonon_dof_indices": phonon.dof_indices,
            }
        )
    np.savez(str(out_path), **save_data)
    print(f"结果已保存: {out_path}")

    if args.plot:
        plot_out = args.plot_out.strip() if args.plot_out else ""
        _plot_equilibrium_layout(
            r_um=r_eq,
            x_range_um=x_range,
            y_range_um=y_range,
            z_range_um=z_range,
            color_mode=args.color_mode,
            charge_ec=charge_ec,
            plot_out=plot_out or None,
            point_size=args.plot_point_size,
        )

    want_plot_hessian = args.plot_hessian is not None
    want_save_hessian = args.plot_hessian_out is not None
    if want_plot_hessian or want_save_hessian:
        if phonon is not None:
            h_total = phonon.hessian_total_eV_per_um2
            h_trap = phonon.hessian_trap_eV_per_um2
            h_coul = phonon.hessian_coulomb_eV_per_um2
        else:
            h_total, h_trap, h_coul = total_hessian(
                fit=fit,
                r_um=r_eq,
                charge_ec=charge_ec,
                softening_um=float(args.softening_um),
            )
            h_total = h_total[np.ix_(hessian_dof_indices, hessian_dof_indices)]
            h_trap = h_trap[np.ix_(hessian_dof_indices, hessian_dof_indices)]
            h_coul = h_coul[np.ix_(hessian_dof_indices, hessian_dof_indices)]

        kind = args.plot_hessian if args.plot_hessian is not None else "total"
        if kind == "trap":
            h_plot = h_trap
        elif kind == "coulomb":
            h_plot = h_coul
        else:
            h_plot = h_total

        hessian_plot_out = None
        if want_save_hessian:
            if args.plot_hessian_out == _DEFAULT_OUT_SENTINEL:
                hessian_plot_out = str(_ROOT / "equilibrium" / "results" / "hessian_plot" / f"{default_stem}.png")
            else:
                hessian_plot_out = args.plot_hessian_out.strip()
                if hessian_plot_out == "":
                    hessian_plot_out = str(_ROOT / "equilibrium" / "results" / "hessian_plot" / f"{default_stem}.png")
        _plot_hessian_heatmap(
            hessian_eV_per_um2=h_plot,
            hessian_kind=kind,
            plot_out=hessian_plot_out,
            show_plot=want_plot_hessian,
        )

    want_plot_spectrum = args.plot_phonon_spectrum is not None
    want_save_spectrum = args.plot_phonon_spectrum_out is not None
    if want_plot_spectrum or want_save_spectrum:
        if phonon is None:
            parser.error("--plot-phonon-spectrum/--plot-phonon-spectrum-out 需要可用的声子模式结果")
        spectrum_mode = args.plot_phonon_spectrum if args.plot_phonon_spectrum is not None else "frequency"
        spectrum_out = None
        if want_save_spectrum:
            if args.plot_phonon_spectrum_out == _DEFAULT_OUT_SENTINEL:
                spectrum_out = str(_ROOT / "equilibrium" / "results" / "spectra" / f"{default_stem}.png")
            else:
                spectrum_out = args.plot_phonon_spectrum_out.strip()
                if spectrum_out == "":
                    spectrum_out = str(_ROOT / "equilibrium" / "results" / "spectra" / f"{default_stem}.png")
        _plot_phonon_spectrum(
            freq_mhz=phonon.freq_hz_signed / 1e6,
            mode=spectrum_mode,
            plot_out=spectrum_out,
            show_plot=want_plot_spectrum,
        )

    want_plot_mode_vector = args.plot_mode_vector is not None
    want_save_mode_vector = args.plot_mode_vector_out is not None
    if want_plot_mode_vector or want_save_mode_vector:
        if phonon is None:
            parser.error("--plot-mode-vector/--plot-mode-vector-out 需要可用的声子模式结果")
        mode_index = int(args.plot_mode_vector) if args.plot_mode_vector is not None else 0
        n_modes = int(phonon.eigvec_cartesian.shape[1])
        if mode_index < 0 or mode_index >= n_modes:
            parser.error(f"--plot-mode-vector 超出范围 [0, {n_modes - 1}]")

        mode_plot_out = None
        if want_save_mode_vector:
            if args.plot_mode_vector_out == _DEFAULT_OUT_SENTINEL:
                mode_plot_out = str(
                    _ROOT
                    / "equilibrium"
                    / "results"
                    / "mode_vector"
                    / f"{default_stem}_mode{mode_index}.png"
                )
            else:
                mode_plot_out = args.plot_mode_vector_out.strip()
                if mode_plot_out == "":
                    mode_plot_out = str(
                        _ROOT
                        / "equilibrium"
                        / "results"
                        / "mode_vector"
                        / f"{default_stem}_mode{mode_index}.png"
                    )

        _plot_phonon_mode_vector_zox(
            r_um=r_eq,
            eigvec_cartesian=phonon.eigvec_cartesian,
            dof_indices=phonon.dof_indices,
            mode_index=mode_index,
            freq_hz_signed_all=phonon.freq_hz_signed,
            arrow_scale_factor=float(args.plot_mode_vector_arrow_scale),
            plot_out=mode_plot_out,
            show_plot=want_plot_mode_vector,
            point_size=args.plot_point_size,
            z_range_um=z_range,
            x_range_um=x_range,
        )

    if args.save_hessian_data:
        if phonon is not None:
            h_total = phonon.hessian_total_eV_per_um2
            h_trap = phonon.hessian_trap_eV_per_um2
            h_coul = phonon.hessian_coulomb_eV_per_um2
        else:
            h_total, h_trap, h_coul = total_hessian(
                fit=fit,
                r_um=r_eq,
                charge_ec=charge_ec,
                softening_um=float(args.softening_um),
            )
            h_total = h_total[np.ix_(hessian_dof_indices, hessian_dof_indices)]
            h_trap = h_trap[np.ix_(hessian_dof_indices, hessian_dof_indices)]
            h_coul = h_coul[np.ix_(hessian_dof_indices, hessian_dof_indices)]

        hessian_data_out = args.hessian_data_out.strip() if args.hessian_data_out else ""
        if not hessian_data_out:
            hessian_data_out = str(_ROOT / "equilibrium" / "results" / "hessian_data" / f"{default_stem}.npz")
        hessian_data_path = Path(hessian_data_out)
        if not hessian_data_path.is_absolute():
            hessian_data_path = _ROOT / hessian_data_path
        hessian_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(hessian_data_path),
            hessian_total_eV_per_um2=h_total,
            hessian_trap_eV_per_um2=h_trap,
            hessian_coulomb_eV_per_um2=h_coul,
            hessian_slice_expr=np.array(args.hessian_slice),
            hessian_dof_indices=hessian_dof_indices,
            N=int(n_ions),
        )
        print(f"Hessian 数据已保存: {hessian_data_path}")


if __name__ == "__main__":
    main()

