"""radial_trap 交互 UI：滑块 + 可编辑文本框调 A/B/D/E/F + 势场实时面板 + 阱频反算。

v1.3 性能与布局（本版）：
- **重解防抖**：拖动/键盘步进后的热启动重解不再逐事件执行，交互停止
  ~0.35 s 才跑一次（GUI 后端 canvas.new_timer 定时器回主线程；无头后端
  立即执行保持可测）——修复方向键自动重复/滑块连点触发的事件风暴卡死
- **绘制节流**：交互期间全图重绘至多 ~30 fps（拖动/按键收尾帧强制
  刷新不丢末帧），默认网格 129×89、画幅 13.8×8.4，降低每帧栅格化开销
- **水平 colorbar**：置于主面板下方而非右侧，不占面板宽度——x 域拉宽
  时等比画面尽量少缩小；主面板加宽至 0.585

v1.2 增强：
- 每系数一行：滑块 + **数值文本框**（直接输入精确值，Enter/失焦提交，越界
  自动扩展滑块范围）+ **范围文本框**（"lo,hi" 提交后按新界重建滑块）
- **N / x-range / y-range 文本框**：离子数（热启动增删离子后重解）与势场
  绘制区域（重建网格 + pcolormesh/colorbar）在面板上动态可调
- **键盘方向键**：聚焦数值框 ↑↓ 步进（Shift ×5 步长）；鼠标悬停滑块轴
  ←→↑↓ 步进、Home/End 跳端点；N 框 ↑↓ ±1（Shift ±5）；区域/范围框
  ↑↓ 以中心为锚放宽/收窄（Shift 细调）
- **DB+F=0 约束开关**（CheckButtons）：勾选后 F = −D·B 随 B/D 联动，
  F 滑块/文本框隐藏禁调，行内显示联动值；取消勾选 F 冻结在当前值

布局（matplotlib.widgets，无额外依赖；WSLg/X11/macOS 桌面环境）：
- 左主面板：总势 pcolormesh（原位 set_array/set_clim/set_cmap 更新，无
  collection 重建；仅区域改变时重建）+ 离子散点 set_offsets + micromotion
  peak-to-peak 线段（LineCollection.set_segments）+ 面板下方水平 colorbar
- 右读出：系数、二次系数 c_x2/c_y2、阱频、q、4 条件 PASS/FAIL、链跨度、
  |a_mm|、约束状态
- 底部 6 行控件：5 系数行（滑块 + 数值框 + 范围框）+ 第 6 行 N/x/y；
  右下 f_x/f_y 文本框（随当前阱频刷新，供 Solve A,E 改目标）+ 约束开关
  + Solve A,E / Save JSON 按钮

更新流（防事件风暴，两级）：
- 逐事件（拖动滑块/键盘步进）→ 仅廉价路径（evaluate_conditions + 势网格
  重算 + 原位刷新 + 节流 draw_idle ≤30fps），绝不重解平衡
- 重解本身再防抖：release 事件只置 _needs_resolve 并启动/重启 350ms
  单发定时器，静止后才热启动重解一次（Slider 无 on_release API，以
  release 事件落在滑块轴上判定；无定时器后端 flush 立即执行）
- 文本框提交（Enter 或点击外部失焦）→ 立即应用：数值/N 连带重解，
  范围/区域只重建对应控件/网格
- Solve A,E：invert_trap_freqs_to_params → 越界按**当前**滑块范围裁剪回写
- Save JSON：--json 严格键集格式（可被 --json 读回；约束开关状态不存档，
  F 的联动结果以数值形式保存）

无头可测：构造与 refresh(compute_eq=...) / set_coeff_value / set_coeff_range /
set_n_ions / set_region / set_constraint / save_json() 均不调用 show()。
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from radial_trap.types import RadialTrapParams

# 默认 Save JSON 输出目录（与 CLI DEFAULT_OUT_DIR 一致）
_RESULTS_DIR = Path(__file__).resolve().parent / "results"

# (params 字段, 滑块标签, 初始 valmin, valmax)
_SLIDER_SPECS = (
    ("A_V_per_um2", "A (V/µm²)", 1e-4, 2e-2),
    ("B_V_per_um4", "B (V/µm⁴)", -5e-7, 5e-7),
    ("D_dimless", "D", -1.0, 1.0),
    ("E_dc_V_per_um2", "E (V/µm²)", -4e-4, 4e-4),
    ("F_dc_V_per_um4", "F (V/µm⁴)", -2e-7, 2e-7),
)
_F_FIELD = "F_dc_V_per_um4"
_CONSTRAINT_LABEL = "F = −D·B (DB+F=0)"

# 数值/范围文本框格式
_VALFMT = "{:.3e}"
_RANGEFMT = "{:.3g},{:.3g}"
_REGIONFMT = "{:g},{:g}"

# N 上下限与区域跨度守卫
_N_MIN, _N_MAX = 1, 200
_REGION_MIN_SPAN_UM = 1e-3
_REGION_MAX_SPAN_UM = 1e4

# 键盘步进：普通 1% 跨度，Shift ×5
_STEP_FRAC = 0.01
_STEP_FRAC_FINE = 0.05
# 区域/范围框 ↑↓ 放缩因子（Shift 细调）
_ZOOM_FACTOR = 1.1
_ZOOM_FACTOR_FINE = 1.02

# v1.3 性能参数：重解防抖静默期与交互期绘制节流上限
_RESOLVE_DELAY_MS = 350
_DRAW_MIN_DT = 1.0 / 30.0  # 全图重绘最小间隔（~30 fps）
# 有事件循环、new_timer 定时器可用的 GUI 后端（其余后端重解立即执行）
_TIMER_BACKENDS = frozenset(
    {"tkagg", "qtagg", "qt5agg", "gtk3agg", "gtk4agg", "wxagg", "macosx"}
)


def _format_pass(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _panel_cmap_clim(v: np.ndarray, clip_pct: tuple[float, float] = (0.5, 99.5)):
    """按数据符号结构选 (cmap, vmin, vmax)：跨零 → RdBu_r 对称；单符号 → viridis。"""
    finite = v[np.isfinite(v)]
    if finite.size == 0 or float(np.max(np.abs(finite))) == 0.0:
        return "viridis", 0.0, 1.0  # 平面板占位
    lo, hi = np.nanpercentile(v, clip_pct)
    if lo < 0.0 < hi:
        m = max(abs(lo), abs(hi))
        return "RdBu_r", -m, m
    return "viridis", lo, hi


def _parse_pair(text: str) -> tuple[float, float]:
    """解析 "lo,hi"（逗号/分号/空白分隔）→ (lo, hi)；非法抛 ValueError。"""
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"need two numbers 'lo,hi', got {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    return lo, hi


class RadialTrapUI:
    """滑块 + 文本框交互窗：构造后 show() 弹窗；refresh()/set_*()/save_json() 无头可测。"""

    def __init__(self, params: RadialTrapParams, *, n_pts: tuple[int, int] = (129, 89)):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

        from radial_trap.lattice import (
            find_radial_equilibrium,
            micromotion_amplitude,
        )
        from radial_trap.potential import (
            build_total_potential_fit,
            evaluate_conditions,
            total_potential_V,
        )

        self.params = replace(params)
        self._solve = find_radial_equilibrium
        self._micromotion = micromotion_amplitude
        self._build_fit = build_total_potential_fit
        self._evaluate = evaluate_conditions
        self._total_potential = total_potential_V
        self._sync_guard = False
        self._needs_resolve = False
        self._f_constrained = False
        self._constraint_syncing = False
        self._n_pts = tuple(n_pts)
        # v1.3：重解防抖定时器 + 绘制节流记账
        self._resolve_timer = None
        self._draw_pending = False
        self._last_draw = 0.0
        self._timer_ok = matplotlib.get_backend().lower() in _TIMER_BACKENDS

        fig = plt.figure(figsize=(13.8, 8.4))
        self.fig = fig

        # 左主面板：总势 + 离子 + micromotion 线段（colorbar 水平置于下方）
        ax = fig.add_axes([0.03, 0.435, 0.585, 0.515])
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Total potential + ions")
        self.ax = ax

        x = np.linspace(*self.params.x_range_um, self._n_pts[0])
        y = np.linspace(*self.params.y_range_um, self._n_pts[1])
        self._grid_x, self._grid_y = x, y
        self._grid_xx, self._grid_yy = np.meshgrid(x, y, indexing="ij")

        # 初始冷启动解（后续重解走热启动）
        fit = build_total_potential_fit(self.params)
        self.eq = find_radial_equilibrium(fit, self.params)
        self.mm = micromotion_amplitude(self.eq.r_eq_um, self.params)
        self.conditions = evaluate_conditions(self.params)

        v0 = total_potential_V(self._grid_xx, self._grid_yy, self.params)
        cmap, vmin, vmax = _panel_cmap_clim(v0)
        self._quad = ax.pcolormesh(x, y, v0.T, shading="auto", cmap=cmap)
        self._quad.set_clim(vmin, vmax)
        self._attach_cbar()

        self._ion_scatter = ax.scatter(
            self.eq.r_eq_um[:, 0], self.eq.r_eq_um[:, 1],
            s=14, color="black", zorder=3, label="r_eq",
        )
        ax.legend(loc="upper right", fontsize=8)
        self._mm_lines = LineCollection([], colors="red", lw=1.5, alpha=0.8, zorder=2)
        self._mm_lines.set_segments(self._mm_segments())
        ax.add_collection(self._mm_lines)

        # 右侧读出
        readout_ax = fig.add_axes([0.635, 0.40, 0.34, 0.55])
        readout_ax.axis("off")
        self.readout_text = readout_ax.text(
            0.0, 1.0, "", va="top", ha="left", fontsize=8.5, family="monospace",
        )

        # ------------------------------------------------------------------
        # 底部 5 系数行：滑块 + 数值框 + 范围框
        # ------------------------------------------------------------------
        self.sliders: dict[str, Slider] = {}
        self.value_boxes: dict[str, TextBox] = {}
        self.range_boxes: dict[str, TextBox] = {}
        self._slider_ax_map: dict[str, object] = {}
        self._slider_labels = {f: lbl for f, lbl, _, _ in _SLIDER_SPECS}

        y0, pitch = 0.322, 0.047
        for i, (field, label, lo, hi) in enumerate(_SLIDER_SPECS):
            y = y0 - i * pitch
            sax = fig.add_axes([0.075, y, 0.30, 0.022])
            vax = fig.add_axes([0.400, y - 0.005, 0.082, 0.030])
            rax = fig.add_axes([0.498, y - 0.005, 0.100, 0.030])
            self._slider_ax_map[field] = sax
            init = float(np.clip(getattr(self.params, field), lo, hi))
            self._make_slider(field, lo, hi, init)
            vb = TextBox(
                vax, "", initial=_VALFMT.format(init), textalignment="center",
            )
            vb.on_submit(lambda txt, f=field: self._on_value_submit(f, txt))
            self.value_boxes[field] = vb
            rb = TextBox(
                rax, "rng", initial=_RANGEFMT.format(lo, hi), textalignment="center",
            )
            rb.on_submit(lambda txt, f=field: self._on_range_submit(f, txt))
            self.range_boxes[field] = rb

        # F 行约束提示（勾选 DB+F=0 后显示，替代隐藏的 F 控件）
        y_f = y0 - 4 * pitch
        self._f_note = fig.text(
            0.075, y_f + 0.010, "", fontsize=7.5, color="dimgray",
            va="center", ha="left", visible=False,
        )

        # 第 6 行：N / x-range / y-range
        y6 = y0 - 5 * pitch
        n_ax = fig.add_axes([0.400, y6 - 0.005, 0.082, 0.030])
        self.tb_n = TextBox(
            n_ax, "N", initial=str(self.params.n_ions), textalignment="center",
        )
        self.tb_n.on_submit(self._on_n_submit)
        xr_ax = fig.add_axes([0.498, y6 - 0.005, 0.100, 0.030])
        self.tb_xrange = TextBox(
            xr_ax, "x µm", initial=_REGIONFMT.format(*self.params.x_range_um),
            textalignment="center",
        )
        self.tb_xrange.on_submit(lambda txt: self._on_region_submit("x", txt))
        yr_ax = fig.add_axes([0.612, y6 - 0.005, 0.100, 0.030])
        self.tb_yrange = TextBox(
            yr_ax, "y µm", initial=_REGIONFMT.format(*self.params.y_range_um),
            textalignment="center",
        )
        self.tb_yrange.on_submit(lambda txt: self._on_region_submit("y", txt))

        # ------------------------------------------------------------------
        # 右下：约束开关 + 阱频反算 + 存档
        # ------------------------------------------------------------------
        cb_ax = fig.add_axes([0.80, 0.322, 0.165, 0.045])
        cb_ax.axis("off")
        self.cb_constraint = CheckButtons(cb_ax, [_CONSTRAINT_LABEL], actives=[False])
        self.cb_constraint.on_clicked(lambda _label: self._on_constraint_toggle())

        tb_fx_ax = fig.add_axes([0.655, 0.205, 0.095, 0.038])
        self.tb_fx = TextBox(
            tb_fx_ax, "f_x (MHz)", initial=f"{self.conditions.f_trap_x_mhz:.4f}"
        )
        tb_fy_ax = fig.add_axes([0.655, 0.130, 0.095, 0.038])
        self.tb_fy = TextBox(
            tb_fy_ax, "f_y (MHz)", initial=f"{self.conditions.f_trap_y_mhz:.4f}"
        )
        solve_ax = fig.add_axes([0.815, 0.205, 0.105, 0.038])
        self._solve_btn = Button(solve_ax, "Solve A,E")
        self._solve_btn.on_clicked(self._on_solve)
        save_ax = fig.add_axes([0.815, 0.130, 0.105, 0.038])
        self._save_btn = Button(save_ax, "Save JSON")
        self._save_btn.on_clicked(self._on_save)

        status_ax = fig.add_axes([0.635, 0.030, 0.34, 0.035])
        status_ax.axis("off")
        self._status_text = status_ax.text(
            0.0, 0.5, "", va="center", ha="left", fontsize=7.5, color="dimgray",
        )

        # 事件：拖动结束（鼠标松开在滑块轴上）与键盘步进后松开 → 防抖重解
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("key_release_event", self._on_key_release)

        self._update_all()
        self._set_status(
            "drag sliders / type values (Enter); arrows: step (Shift x5); "
            "arrows in N/x/y/rng boxes; release -> re-solve (debounced)"
        )

    # ------------------------------------------------------------------
    # 控件构建
    # ------------------------------------------------------------------

    def _make_slider(self, field: str, lo: float, hi: float, val: float) -> None:
        """在既有滑块轴上（重）建 Slider：范围改变时断开旧件、原轴重建。"""
        from matplotlib.widgets import Slider

        old = self.sliders.get(field)
        if old is not None:
            old.disconnect_events()
        sax = self._slider_ax_map[field]
        sax.clear()
        s = Slider(
            sax, self._slider_labels[field], lo, hi,
            valinit=float(np.clip(val, lo, hi)),
        )
        s.label.set_fontsize(8)
        s.valtext.set_visible(False)  # 数值框已显示精确值，滑块自带 valtext 冗余
        if field == _F_FIELD and self._f_constrained:
            s.eventson = False
        s.on_changed(self._make_slider_cb(field))
        self.sliders[field] = s

    def _make_slider_cb(self, field: str):
        def _cb(_val: float) -> None:
            if self._sync_guard:
                return
            self._sync_params_from_widgets()
            self._update_all()  # 廉价路径：不重解平衡
            self._needs_resolve = True  # 重解推迟到 release（再经防抖定时器）

        return _cb

    # ------------------------------------------------------------------
    # 参数同步与刷新
    # ------------------------------------------------------------------

    def _sync_params_from_widgets(self) -> None:
        values = {f: s.val for f, s in self.sliders.items()}
        if self._f_constrained:
            values[_F_FIELD] = -values["D_dimless"] * values["B_V_per_um4"]
        self.params = replace(self.params, **values)

    def refresh(self, compute_eq: bool = False) -> None:
        """重算并刷新面板/读出；compute_eq=True 时热启动重解平衡构型。"""
        self._sync_params_from_widgets()
        if compute_eq:
            self._resolve()
        self._update_all()

    def _resolve(self, r_init_um: np.ndarray | None = None) -> None:
        """重解平衡（默认以上一构型热启动）并刷新散点/micromotion 线段。"""
        if r_init_um is None and self.eq is not None:
            r_init_um = self.eq.r_eq_um
        fit = self._build_fit(self.params)
        self.eq = self._solve(fit, self.params, r_init_um=r_init_um)
        self.mm = self._micromotion(self.eq.r_eq_um, self.params)
        self._ion_scatter.set_offsets(
            np.column_stack([self.eq.r_eq_um[:, 0], self.eq.r_eq_um[:, 1]])
        )
        self._mm_lines.set_segments(self._mm_segments())
        self._needs_resolve = False  # 已重解：清除挂起标记（含防抖定时器路径）

    def _update_all(self) -> None:
        """廉价路径：条件 + 势网格 + 面板/读出/控件文本原位更新 + 节流重绘。"""
        self.conditions = self._evaluate(self.params)
        v = self._total_potential(self._grid_xx, self._grid_yy, self.params)
        cmap, vmin, vmax = _panel_cmap_clim(v)
        self._quad.set_array(np.asarray(v.T).ravel())
        self._quad.set_cmap(cmap)
        self._quad.set_clim(vmin, vmax)
        self._cbar.update_normal(self._quad)
        self.readout_text.set_text("\n".join(self._readout_lines()))
        self._update_widget_texts()
        self._request_draw()

    def _request_draw(self, force: bool = False) -> None:
        """绘制节流：交互期间全图重绘至多 ~30 fps；force 直通（收尾帧）。"""
        now = time.perf_counter()
        if force or (now - self._last_draw) >= _DRAW_MIN_DT:
            self._last_draw = now
            self._draw_pending = False
            self.fig.canvas.draw_idle()
        else:
            self._draw_pending = True  # 下一个事件 / 收尾 force 时补画

    def _schedule_resolve(self) -> None:
        """防抖重解：交互静默 _RESOLVE_DELAY_MS 后执行一次。

        GUI 后端用 canvas.new_timer 单发定时器（回调在 GUI 主线程跑，可安全
        动 artist）；无定时器后端（Agg 等无头）立即执行，保持可测性。"""
        if not self._timer_ok:
            self._flush_resolve()
            return
        if self._resolve_timer is None:
            t = self.fig.canvas.new_timer(interval=_RESOLVE_DELAY_MS)
            t.single_shot = True
            t.add_callback(self._flush_resolve)
            self._resolve_timer = t
        self._resolve_timer.stop()  # 重启计时：持续交互时不断顺延
        self._resolve_timer.start()

    def _flush_resolve(self) -> None:
        """执行挂起的热启动重解（防抖定时器回调 / 无头立即调用）。"""
        if not self._needs_resolve:
            return
        if self._resolve_timer is not None:
            try:
                self._resolve_timer.stop()
            except Exception:
                pass
        self.refresh(compute_eq=True)
        self._request_draw(force=True)

    def _update_widget_texts(self) -> None:
        """把控件文本回写到当前参数（用户正在输入的框不动）。"""
        for f, s in self.sliders.items():
            vb = self.value_boxes[f]
            vtxt = _VALFMT.format(s.val)
            if not vb.capturekeystrokes and vb.text != vtxt:
                vb.set_val(vtxt)
            rb = self.range_boxes[f]
            rtxt = _RANGEFMT.format(s.valmin, s.valmax)
            if not rb.capturekeystrokes and rb.text != rtxt:
                rb.set_val(rtxt)
        if self._f_constrained:
            # F 滑块静默跟随联动值（值相同则不触发，无递归）
            s_f = self.sliders[_F_FIELD]
            if s_f.val != self.params.F_dc_V_per_um4:
                s_f.set_val(self.params.F_dc_V_per_um4)
            self._f_note.set_text(
                f"F = −D·B = {self.params.F_dc_V_per_um4:.3e} V/µm⁴ "
                "(constrained; uncheck to free)"
            )
        for box, txt in (
            (self.tb_n, str(self.params.n_ions)),
            (self.tb_xrange, _REGIONFMT.format(*self.params.x_range_um)),
            (self.tb_yrange, _REGIONFMT.format(*self.params.y_range_um)),
        ):
            if not box.capturekeystrokes and box.text != txt:
                box.set_val(txt)
        # f_x/f_y 框随当前阱频刷新（Solve 的初值提示；聚焦编辑时不覆盖）
        for box, f in (
            (self.tb_fx, self.conditions.f_trap_x_mhz),
            (self.tb_fy, self.conditions.f_trap_y_mhz),
        ):
            ftxt = f"{f:.4f}"
            if not box.capturekeystrokes and box.text != ftxt:
                box.set_val(ftxt)

    # ------------------------------------------------------------------
    # 数值 / 范围 / N / 区域 / 约束 —— 提交入口（文本框 on_submit 与键盘共用）
    # ------------------------------------------------------------------

    def set_coeff_value(
        self, field: str, value: float, *, ressolve: bool = True
    ) -> bool:
        """数值框提交：设精确值；越界时自动扩展滑块范围；默认连带重解。"""
        if field == _F_FIELD and self._f_constrained:
            self._set_status("F constrained by F = −D·B; uncheck to free F", error=True)
            return False
        if not np.isfinite(value):
            self._set_status(f"invalid value: {value!r}", error=True)
            return False
        s = self.sliders[field]
        lo, hi = s.valmin, s.valmax
        note = ""
        if value < lo or value > hi:
            lo, hi = min(lo, value), max(hi, value)
            self._make_slider(field, lo, hi, value)
            self._sync_params_from_widgets()
            self._update_all()
            note = "; slider range expanded"
        else:
            s.set_val(value)  # 触发廉价路径
        if ressolve:
            self.refresh(compute_eq=True)
        short = field.split("_", 1)[0]
        self._set_status(f"{short} = {value:.4e}{note}")
        return True

    def set_coeff_range(self, field: str, lo: float, hi: float) -> bool:
        """范围框提交：按 [lo, hi] 重建滑块（当前值越界被裁剪），不重解。"""
        if field == _F_FIELD and self._f_constrained:
            self._set_status("F constrained by F = −D·B; uncheck to free F", error=True)
            return False
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            self._set_status(
                f"invalid range {lo:.3g},{hi:.3g} (need lo < hi)", error=True
            )
            return False
        self._make_slider(field, lo, hi, getattr(self.params, field))
        self._sync_params_from_widgets()  # 若当前值被裁剪则同步进 params
        self._update_all()
        self._set_status(
            f"{field.split('_', 1)[0]} range -> [{lo:.3g}, {hi:.3g}]"
        )
        return True

    def set_n_ions(self, n: int) -> bool:
        """N 框提交：热启动增删离子（小 N 等距取样 / 大 N 随机补点）后重解。"""
        if not (isinstance(n, int) and _N_MIN <= n <= _N_MAX):
            self._set_status(
                f"invalid N {n!r} (need {_N_MIN}..{_N_MAX})", error=True
            )
            return False
        if n == self.params.n_ions:
            return True
        warm = self._resize_positions(self.eq.r_eq_um, n)
        self.params = replace(self.params, n_ions=n)
        self._resolve(r_init_um=warm)
        self._update_all()
        self._set_status(f"N = {n} (warm-resized + re-solved)")
        return True

    def set_region(self, axis: str, lo: float, hi: float) -> bool:
        """区域框提交：设 x/y 绘制范围，重建网格与 pcolormesh/colorbar。"""
        attr = f"{axis}_range_um"
        span = hi - lo
        if (
            axis not in ("x", "y")
            or not (np.isfinite(lo) and np.isfinite(hi))
            or span < _REGION_MIN_SPAN_UM
            or span > _REGION_MAX_SPAN_UM
        ):
            self._set_status(
                f"invalid {axis} range {lo:.3g},{hi:.3g} "
                f"(span {_REGION_MIN_SPAN_UM:g}..{_REGION_MAX_SPAN_UM:g} µm)",
                error=True,
            )
            return False
        self.params = replace(self.params, **{attr: (lo, hi)})
        self._rebuild_mesh()
        self._update_all()
        self._set_status(f"{axis} range -> [{lo:g}, {hi:g}] µm")
        return True

    def set_constraint(self, enabled: bool, *, ressolve: bool = True) -> None:
        """DB+F=0 约束开关：开启后 F = −D·B 联动并隐藏 F 控件；关闭时冻结 F。

        程序化调用时同步复选框视觉（set_active 会再触发 on_clicked，
        以 _constraint_syncing 守卫防重入）。"""
        if enabled == self._f_constrained:
            return
        if self.cb_constraint.get_status() != [enabled]:
            self._constraint_syncing = True
            try:
                self.cb_constraint.set_active(0)  # 翻转到目标态（触发守卫下的回调）
            finally:
                self._constraint_syncing = False
        self._f_constrained = enabled
        s_f = self.sliders[_F_FIELD]
        show = not enabled
        self._slider_ax_map[_F_FIELD].set_visible(show)
        self.value_boxes[_F_FIELD].ax.set_visible(show)
        self.range_boxes[_F_FIELD].ax.set_visible(show)
        self._f_note.set_visible(enabled)
        if enabled:
            s_f.eventson = False
            self._sync_params_from_widgets()  # F := −D·B
            if ressolve:
                self.refresh(compute_eq=True)
            self._set_status("constraint on: F = −D·B (F controls disabled)")
        else:
            s_f.eventson = True
            s_f.set_val(self.params.F_dc_V_per_um4)  # 回填滑块（廉价路径）
            self._set_status("constraint off: F frozen at current value")
        self._update_all()

    # -- 文本框 on_submit 薄壳 ------------------------------------------

    def _on_value_submit(self, field: str, text: str) -> None:
        try:
            v = float(text.strip())
        except ValueError:
            self._set_status(f"not a number: {text!r}", error=True)
            return
        self.set_coeff_value(field, v)

    def _on_range_submit(self, field: str, text: str) -> None:
        try:
            lo, hi = _parse_pair(text)
        except ValueError as exc:
            self._set_status(f"bad range: {exc}", error=True)
            return
        self.set_coeff_range(field, lo, hi)

    def _on_n_submit(self, text: str) -> None:
        try:
            n = int(text.strip())
        except ValueError:
            self._set_status(f"not an integer: {text!r}", error=True)
            return
        self.set_n_ions(n)

    def _on_region_submit(self, axis: str, text: str) -> None:
        try:
            lo, hi = _parse_pair(text)
        except ValueError as exc:
            self._set_status(f"bad range: {exc}", error=True)
            return
        self.set_region(axis, lo, hi)

    def _on_constraint_toggle(self) -> None:
        # set_active(0) 翻转可视状态并触发本回调；以复选框实际状态为准。
        # 程序化 set_constraint 内部翻转复选框时以守卫短路，防二次进入
        if self._constraint_syncing:
            return
        self.set_constraint(bool(self.cb_constraint.get_status()[0]))

    # ------------------------------------------------------------------
    # 键盘：方向键步进（聚焦文本框 ↑↓ / 悬停滑块 ←→↑↓ + Home/End）
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        key = event.key
        if key not in (
            "up", "down", "left", "right",
            "shift+up", "shift+down", "shift+left", "shift+right",
            "home", "end",
        ):
            return
        base = key.split("+")[-1]
        fine = key.startswith("shift+")

        # 1) 聚焦文本框（capturekeystrokes）：↑↓ 步进（←→ 留给光标）
        for f, b in self.value_boxes.items():
            if b.capturekeystrokes:
                if base in ("up", "down"):
                    self._step_value_box(f, +1 if base == "up" else -1, fine)
                return
        if self.tb_n.capturekeystrokes:
            if base in ("up", "down"):
                self._step_n(+1 if base == "up" else -1, fine)
            return
        for axis, b in (("x", self.tb_xrange), ("y", self.tb_yrange)):
            if b.capturekeystrokes:
                if base in ("up", "down"):
                    self._step_region(axis, +1 if base == "up" else -1, fine)
                return
        for f, b in self.range_boxes.items():
            if b.capturekeystrokes:
                if base in ("up", "down"):
                    self._step_range_box(f, +1 if base == "up" else -1, fine)
                return

        # 2) 悬停滑块轴：←→↑↓ 步进（1%/5% 跨度），Home/End 跳端点
        if event.inaxes is None:
            return
        for f, sax in self._slider_ax_map.items():
            if event.inaxes is not sax or not sax.get_visible():
                continue
            s = self.sliders[f]
            if key == "home":
                s.set_val(s.valmin)
            elif key == "end":
                s.set_val(s.valmax)
            else:
                step = (s.valmax - s.valmin) * (_STEP_FRAC_FINE if fine else _STEP_FRAC)
                dv = -step if base in ("left", "down") else +step
                s.set_val(float(np.clip(s.val + dv, s.valmin, s.valmax)))
            self._needs_resolve = True  # 重解推迟到 key_release（防抖）
            return

    def _on_key_release(self, event) -> None:
        base = (event.key or "").split("+")[-1]
        if base not in ("up", "down", "left", "right"):
            return
        self._request_draw(force=True)  # 步进收尾帧不丢
        if self._needs_resolve:
            self._schedule_resolve()  # 防抖：方向键自动重复只触发末次重解

    def _step_value_box(self, field: str, sign: int, fine: bool) -> None:
        try:
            v = float(self.value_boxes[field].text.strip())
        except ValueError:
            v = getattr(self.params, field)
        s = self.sliders[field]
        step = (s.valmax - s.valmin) * (_STEP_FRAC_FINE if fine else _STEP_FRAC)
        new = float(np.clip(v + sign * step, s.valmin, s.valmax))
        if self.set_coeff_value(field, new, ressolve=False):
            self._needs_resolve = True

    def _step_n(self, sign: int, fine: bool) -> None:
        delta = 5 if fine else 1
        self.set_n_ions(int(np.clip(self.params.n_ions + sign * delta, _N_MIN, _N_MAX)))

    def _step_region(self, axis: str, sign: int, fine: bool) -> None:
        lo0, hi0 = getattr(self.params, f"{axis}_range_um")
        center, span = (lo0 + hi0) / 2.0, hi0 - lo0
        span *= (_ZOOM_FACTOR_FINE if fine else _ZOOM_FACTOR) ** sign
        self.set_region(axis, center - span / 2.0, center + span / 2.0)

    def _step_range_box(self, field: str, sign: int, fine: bool) -> None:
        s = self.sliders[field]
        center, span = (s.valmin + s.valmax) / 2.0, s.valmax - s.valmin
        span *= (_ZOOM_FACTOR_FINE if fine else _ZOOM_FACTOR) ** sign
        if span <= 1e-15:
            return
        self.set_coeff_range(field, center - span / 2.0, center + span / 2.0)

    # ------------------------------------------------------------------
    # 网格/构型辅助
    # ------------------------------------------------------------------

    def _attach_cbar(self) -> None:
        """主面板下方挂水平 colorbar（不占面板宽度，x 域拉宽时画面少缩小）。

        pad 留足主面板 x 刻度 + xlabel 的竖直空间，避免与 colorbar 叠字。"""
        self._cbar = self.fig.colorbar(
            self._quad, ax=self.ax, orientation="horizontal",
            fraction=0.075, pad=0.10, shrink=0.9, label="V",
        )

    def _rebuild_mesh(self) -> None:
        """区域改变：重建格点网格与 pcolormesh/colorbar（散点/线段保留）。"""
        x = np.linspace(*self.params.x_range_um, self._n_pts[0])
        y = np.linspace(*self.params.y_range_um, self._n_pts[1])
        self._grid_x, self._grid_y = x, y
        self._grid_xx, self._grid_yy = np.meshgrid(x, y, indexing="ij")
        # 先删 colorbar 再删 QuadMesh（后者 remove 后 .axes=None 会让前者崩）
        self._cbar.remove()
        self._quad.remove()
        v = self._total_potential(self._grid_xx, self._grid_yy, self.params)
        cmap, vmin, vmax = _panel_cmap_clim(v)
        self._quad = self.ax.pcolormesh(x, y, v.T, shading="auto", cmap=cmap)
        self._quad.set_clim(vmin, vmax)
        self._attach_cbar()
        self.ax.set_xlim(x[0], x[-1])
        self.ax.set_ylim(y[0], y[-1])

    def _resize_positions(self, r: np.ndarray, n: int) -> np.ndarray:
        """热启动增删离子：减 N 等距取样（保跨度），增 N 在范围内随机补点。"""
        m = r.shape[0]
        if n == m:
            return r.copy()
        if n < m:
            idx = np.linspace(0, m - 1, n).round().astype(int)
            return r[idx].copy()
        rng = np.random.default_rng(self.params.seed)
        x0, x1 = self.params.x_range_um
        y0, y1 = self.params.y_range_um
        add = np.column_stack([
            rng.uniform(x0, x1, n - m),
            rng.uniform(y0, y1, n - m),
            np.zeros(n - m),
        ])
        return np.vstack([r, add])

    def _mm_segments(self) -> list[list[tuple[float, float]]]:
        r = self.eq.r_eq_um
        a = self.mm.a_mm_um[:, :2]
        return [
            [
                (r[i, 0] - a[i, 0], r[i, 1] - a[i, 1]),
                (r[i, 0] + a[i, 0], r[i, 1] + a[i, 1]),
            ]
            for i in range(r.shape[0])
        ]

    def _readout_lines(self) -> list[str]:
        p, c = self.params, self.conditions
        lines = [
            f"A = {p.A_V_per_um2:.4e} V/µm²   B = {p.B_V_per_um4:.2e} V/µm⁴",
            f"D = {p.D_dimless:.3f}   E = {p.E_dc_V_per_um2:.4e}   "
            f"F = {p.F_dc_V_per_um4:.2e}",
        ]
        if self._f_constrained:
            lines.append(f"F = −D·B = {p.F_dc_V_per_um4:.3e} (constrained)")
        lines += [
            f"c_x2 = {c.c_x2:.4e}   c_y2 = {c.c_y2:.4e} V/µm²",
            f"f_x = {c.f_trap_x_mhz:.4f} MHz   f_y = {c.f_trap_y_mhz:.4f} MHz",
            f"q_x = {c.q_mathieu_x:.4f}",
            f"(1) cross term : {_format_pass(c.pass_cross)}",
            f"(2) y quartic  : {_format_pass(c.pass_y_high)}",
            f"(3) y confine  : {_format_pass(c.pass_y_confine)}",
            f"(4) x confine  : {_format_pass(c.pass_x_confine)}",
        ]
        if c.xy_degenerate:
            lines.append("x/y quadratic degenerate: orientation set by initial state")
        if self.eq is not None:
            r = self.eq.r_eq_um
            span_x = float(r[:, 0].max() - r[:, 0].min())
            lines.append(
                f"N = {r.shape[0]}, x-span = {span_x:.3f} µm, "
                f"nit = {self.eq.n_iter}, converged = {self.eq.converged}"
            )
        if self.mm is not None:
            lines.append(
                f"|a_mm| max = {self.mm.mag_um.max():.4f} µm, "
                f"mean = {self.mm.mag_um.mean():.4f} µm"
            )
        return lines

    # ------------------------------------------------------------------
    # 事件：松开重解（防抖）/ 阱频反算 / 存档
    # ------------------------------------------------------------------

    def _on_release(self, event) -> None:
        # Slider 无 on_release API：以 release 事件落在滑块轴上判定拖动结束；
        # 重解经 _schedule_resolve 防抖（无头后端立即执行）
        if event.inaxes not in self._slider_ax_map.values():
            return
        self._request_draw(force=True)  # 拖动收尾帧不被节流丢弃
        if self._needs_resolve:
            self._schedule_resolve()

    def _on_solve(self, _event=None) -> None:
        from radial_trap.potential import invert_trap_freqs_to_params

        try:
            f_x = float(self.tb_fx.text)
            f_y = float(self.tb_fy.text)
            a_new, e_new, q_x = invert_trap_freqs_to_params(
                f_x, f_y, self.params.freq_rf_mhz, self.params.species_name,
                D=self.params.D_dimless,
            )
        except ValueError as exc:
            self._set_status(f"Solve failed: {exc}", error=True)
            return
        notes = []
        s_a = self.sliders["A_V_per_um2"]
        s_e = self.sliders["E_dc_V_per_um2"]
        a_c = float(np.clip(a_new, s_a.valmin, s_a.valmax))
        e_c = float(np.clip(e_new, s_e.valmin, s_e.valmax))
        if a_c != a_new:
            notes.append("A clipped to slider range")
        if e_c != e_new:
            notes.append("E clipped to slider range")
        self._sync_guard = True
        try:
            s_a.set_val(a_c)  # 触发守卫下的廉价路径
            s_e.set_val(e_c)
        finally:
            self._sync_guard = False
        self.refresh(compute_eq=True)  # 手动一次重解
        self._set_status(
            f"Solved A={a_new:.4e}, E={e_new:.4e} (q_x={q_x:.4f})"
            + ("; " + "; ".join(notes) if notes else "")
        )

    def save_json(self, out_path: str | Path | None = None) -> Path:
        """按 --json 严格键集格式保存当前参数（可被 --json 读回）。"""
        p = self.params
        data = {
            "A": p.A_V_per_um2,
            "B": p.B_V_per_um4,
            "D": p.D_dimless,
            "E": p.E_dc_V_per_um2,
            "F": p.F_dc_V_per_um4,
            "freq_rf_mhz": p.freq_rf_mhz,
            "species": p.species_name,
            "N": p.n_ions,
            "x_range": list(p.x_range_um),
            "y_range": list(p.y_range_um),
            "seed": p.seed,
            "softening_um": p.softening_um,
        }
        if out_path is None:
            out = _RESULTS_DIR / "ui_params.json"
        else:
            out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"UI 参数存档: {out}")
        return out

    def _on_save(self, _event=None) -> None:
        out = self.save_json()
        self._set_status(f"saved {out.name}")

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_text.set_color("crimson" if error else "dimgray")
        self._status_text.set_text(msg)
        self._request_draw()

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        if self._resolve_timer is not None:
            try:
                self._resolve_timer.stop()
            except Exception:
                pass
            self._resolve_timer = None
        import matplotlib.pyplot as plt

        plt.close(self.fig)
