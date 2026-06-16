"""
Micromotion 数值测量核心库

从 --continuous-sampling 产出的 frame*.npz 轨迹数据，数值测量每个离子的
RF micromotion 幅度（随时间变化的 β(t)）与有效 Mathieu 调制深度 q_eff，
并与 trap_stability 从场几何算出的理论 q 交叉验证。

物理基础
--------
Paul 阱绝热近似下离子的运动是乘性调制（非加性）：

    x(t) ≈ X_sec(t) · [ 1 + (q/2)·cos(Ω_RF t) ]

micromotion 幅度正比于瞬时 secular 位移。本模块用两条独立路径测量：

1. Phase-folding（主方法）：按 RF 相位滑窗分 bin，拟合一阶谐波，得随时间
   变化的 micromotion 幅度 β(t)（诊断用，天然处理幅度调制）。
2. 乘性模型最小二乘（验证方法 C）：对整段轨迹回归调制深度，得全局标量 q_eff，
   直接对标 trap_stability 的理论 q。

全程使用物理单位（µm、µs）。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_DEFAULT_SECULAR_FREQ_MHZ = 0.5  # secular 频率估计（采样校验用），可由调用方覆盖


# ============================== 数据加载 ==============================

@dataclass
class TrajectoryData:
    """连续采样轨迹，物理单位。"""

    t_us: np.ndarray              # (T,)
    r_um: np.ndarray              # (T, N, 3)
    v_um_per_s: np.ndarray        # (T, N, 3)
    Omega_rf: float               # rad/µs
    freq_rf_MHz: float
    dt_us: float                  # 中位帧间隔
    n_ions: int
    run_dir: Path


def _load_rf_frequency(config_path: str | Path, mass_amu: float | None) -> tuple[float, float]:
    """从 config JSON 提取 RF 频率，返回 (freq_RF_MHz, Omega_rf_rad_per_us)。

    Omega_rf = 2π·freq_RF（rad/µs），因 1 MHz = 1 cycle/µs。
    """
    from FieldConfiguration.constants import init_from_config

    cfg, _ = init_from_config(str(config_path), mass_amu=mass_amu)
    freq_rf = float(cfg.freq_RF)
    return freq_rf, 2.0 * np.pi * freq_rf


def load_continuous_sampling(
    run_dir: str | Path,
    *,
    csv_path: str | Path | None = None,
    config_path: str | Path | None = None,
    species: str = "Ba135+",
    freq_rf_MHz: float | None = None,
    check_sampling: bool = True,
    secular_freq_MHz: float = _DEFAULT_SECULAR_FREQ_MHZ,
) -> TrajectoryData:
    """
    加载 continuous_sampling 目录下的 frame*.npz，构建 TrajectoryData。

    RF 频率优先级：freq_rf_MHz（显式）> config_path 解析。
    （按规划要求，正常使用应通过 csv/config 传入，freq_rf_MHz 仅作回退/测试钩子。）

    Parameters
    ----------
    run_dir : 连续采样输出目录（含 frame*.npz）
    csv_path, config_path : 电场 CSV 与电压 JSON；config 用于解析 RF 频率
    species : 离子种类（决定 config 无量纲化的质量）
    freq_rf_MHz : 显式 RF 频率覆盖（测试用）
    check_sampling : 是否执行采样质量校验
    secular_freq_MHz : secular 频率估计，用于总时长校验

    Raises
    ------
    FileNotFoundError / ValueError : 目录无帧 / 采样率或时长不达标
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"连续采样目录不存在: {run_dir}")

    paths = sorted(
        run_dir.glob("frame*.npz"),
        key=lambda p: int(p.stem.replace("frame", "")),
    )
    if not paths:
        raise FileNotFoundError(f"{run_dir} 下未找到 frame*.npz")

    # RF 频率
    mass_amu: float | None = None
    if freq_rf_MHz is None:
        if config_path is None:
            raise ValueError(
                "需通过 config_path 解析 RF 频率，或显式传 freq_rf_MHz"
            )
        if species != "Ba135+" or True:
            from FieldConfiguration.ion_species import ION_SPECIES

            sp = ION_SPECIES.get(species)
            mass_amu = sp.mass_amu if sp is not None else None
        freq_rf_MHz, Omega_rf = _load_rf_frequency(config_path, mass_amu)
    else:
        Omega_rf = 2.0 * np.pi * float(freq_rf_MHz)

    # 逐帧堆叠
    r_list, v_list, t_list = [], [], []
    n_ions: int | None = None
    for p in paths:
        with np.load(p) as data:
            r = np.asarray(data["r"], dtype=np.float64)
            v = np.asarray(data["v"], dtype=np.float64)
            if "t_us" in data:
                t = float(np.asarray(data["t_us"]).ravel()[0])
            else:
                t = np.nan
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"{p}: r 须为 (N,3)，实际 {r.shape}")
        if n_ions is None:
            n_ions = r.shape[0]
        elif r.shape[0] != n_ions:
            raise ValueError(
                f"{p}: 离子数 {r.shape[0]} 与首帧 {n_ions} 不一致"
            )
        r_list.append(r)
        v_list.append(v)
        t_list.append(t)

    assert n_ions is not None
    r_um = np.stack(r_list, axis=0)            # (T, N, 3)
    v_um_per_s = np.stack(v_list, axis=0)
    t_us = np.asarray(t_list, dtype=np.float64)

    if not np.all(np.isfinite(t_us)):
        raise ValueError("部分帧缺 t_us，无法构建时间轴")

    dt_us = float(np.median(np.diff(t_us)))

    if check_sampling:
        _check_sampling_quality(
            dt_us, t_us, Omega_rf, freq_rf_MHz, secular_freq_MHz
        )

    return TrajectoryData(
        t_us=t_us,
        r_um=r_um,
        v_um_per_s=v_um_per_s,
        Omega_rf=Omega_rf,
        freq_rf_MHz=float(freq_rf_MHz),
        dt_us=dt_us,
        n_ions=n_ions,
        run_dir=run_dir,
    )


def _check_sampling_quality(
    dt_us: float,
    t_us: np.ndarray,
    Omega_rf: float,
    freq_rf_MHz: float,
    secular_freq_MHz: float,
) -> None:
    """采样质量断言：每 RF 周期点数与总时长。"""
    T_rf_us = 2.0 * np.pi / Omega_rf          # = 1/freq_rf_MHz
    pts_per_rf = T_rf_us / dt_us if dt_us > 0 else 0.0

    # 每 RF 周期至少 8 点，方可稳健分辨一阶谐波
    min_pts_per_rf = 8.0
    if pts_per_rf < min_pts_per_rf:
        raise ValueError(
            f"采样率不足：每 RF 周期仅 {pts_per_rf:.1f} 点（需 ≥ {min_pts_per_rf:.0f}）。"
            f" 帧间隔 dt={dt_us*1e3:.2f} ns，RF 周期 T_RF={T_rf_us*1e3:.2f} ns。"
            " 请减小 --interval 或增大 --step 以提高 RF 周期内的采样密度"
            "（如 --interval 0.08 --step 10）。"
        )

    total_us = float(t_us[-1] - t_us[0])
    T_secular_us = 1.0 / secular_freq_MHz if secular_freq_MHz > 0 else np.inf
    min_total = max(3.0 * T_secular_us, 5.0)
    if total_us < min_total:
        raise ValueError(
            f"总时长不足：{total_us:.3f} µs（需 ≥ {min_total:.2f} µs ≈ 3 个 secular 周期）。"
            f" 请增加 --continuous-sampling-frames，或通过 secular_freq 调整估计。"
        )

    logger.info(
        "采样校验通过：dt=%.2f ns，%.1f 点/RF周期，总时长 %.2f µs",
        dt_us * 1e3, pts_per_rf, total_us,
    )


# ============================== Phase-folding + 方法 C ==============================

@dataclass
class WindowMM:
    """单个滑窗的 micromotion 一阶谐波结果。"""

    t_center_us: float
    r_secular_um: float       # 窗内 secular 中心 r̄（拟合常数项）
    beta_um: float            # 一阶谐波幅度 |β|
    phase_rad: float          # atan2(β_s, β_c)
    ptp_um: float             # 2|β|，peak-to-peak


@dataclass
class IonAxisResult:
    """单离子单轴的 micromotion 分析结果。"""

    t_windows: np.ndarray                 # (W,) 窗中心时刻
    beta_t: np.ndarray                    # (W,) β(t) µm
    r_secular_t: np.ndarray               # (W,) 窗内 secular 中心
    phase_t: np.ndarray                   # (W,) 相位
    windows: list[WindowMM] = field(default_factory=list)
    q_eff: float = float("nan")           # 方法 C 全局调制深度
    q_eff_stderr: float = float("nan")
    secular_envelope_um: np.ndarray | None = None   # 低通 X_sec(t)，全段
    residual_rms_um: float = float("nan")  # 方法 C 残差 RMS
    t_star_us: float = float("nan")        # warmup 收敛起点（裁剪阈值），nan=未设
    dropped_frames: int = 0                # 丢弃的初始帧数（= 裁剪起点帧索引）
    warmup_reason: str = "none"            # {none, static, unconverged-tail, too-short, auto, manual-abs, manual-rel}[+clamped]


def _fft_lowpass(x: np.ndarray, dt: float, cutoff_per_us: float) -> np.ndarray:
    """FFT 理想低通：截断高于 cutoff (cycle/µs) 的频率分量。

    锐截止可干净分离 RF 基波（≈ freq_RF）与 secular 运动（≪ freq_RF），
    避免 savgol 类低通的传递函数泄漏对 q_eff 回归的系统性偏差。
    """
    n = x.size
    freqs = np.fft.rfftfreq(n, d=dt)        # cycle/µs
    spectrum = np.fft.rfft(x)
    spectrum[freqs > cutoff_per_us] = 0.0
    return np.fft.irfft(spectrum, n=n)


# ============================== Warmup / 收敛检测 ==============================


def _mad(x: np.ndarray) -> float:
    """归一化中位绝对偏差（×1.4826，σ 的稳健估计）。"""
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * 1.4826)


@dataclass
class WarmupInfo:
    """单离子单轴的瞬态收敛检测结果。

    ``i_start`` 与 ``dropped_frames`` 恒相等（裁掉的初始帧数）。
    三条 no-op 路径（too-short / static / unconverged-tail）返回 ``i_start=0``，
    调用方据此不对轨迹裁剪。
    """

    t_star_us: float          # 稳态起点时刻
    i_start: int              # 对应帧索引（= dropped_frames）
    dropped_frames: int       # 丢弃的初始帧数
    secular_freq_MHz: float   # 检测所用 secular 频率（估计或回退先验）
    reason: str               # {none, static, unconverged-tail, too-short, auto}


def detect_warmup(
    r_axis_um: np.ndarray,
    t_us: np.ndarray,
    Omega_rf: float,
    *,
    secular_freq_MHz: float = _DEFAULT_SECULAR_FREQ_MHZ,
    freq_rf_MHz: float,
    n_periods: int = 3,
    tol_rel: float = 0.1,
    n_overlap: int = 2,
) -> WarmupInfo:
    """
    检测单离子单轴轨迹的瞬态收敛起点 t*。

    用 FFT 低通提取 secular 包络 X_sec，按滑窗跟踪其中心/幅度统计量相对末段
    参考的稳定性，找出最早 ``t*`` 使其后（经 3 窗中值滤波去孤点的）窗都收敛。
    仅当存在长度 ≥3 的连续未收敛窗段时才裁剪；孤立噪声窗被中值滤波抑制。

    三条 no-op 路径返回 ``t*=t[0]``、``dropped_frames=0``：
      * ``too-short`` —— 样本不足以填满 ≥4 个检测窗；
      * ``static``    —— 末段 secular 包络近常数（如常位移轴），无瞬态可言；
      * ``unconverged-tail`` —— 末段自身未稳，无法定义收敛参考（保守不裁）。

    Parameters
    ----------
    r_axis_um : (T,) 单轴位置序列（µm）
    t_us : (T,) 时间（µs）
    Omega_rf : RF 角频率（rad/µs）
    secular_freq_MHz : secular 频率先验（MHz），rfft 估计失败时回退
    freq_rf_MHz : RF 频率（MHz），定低通截止与谱搜索上界
    n_periods : 检测窗覆盖的 secular 周期数
    tol_rel : 相对收敛容差
    n_overlap : 窗重叠因子（step = W_n / n_overlap），默认 2（50%）
    """
    r = np.asarray(r_axis_um, dtype=np.float64)
    t = np.asarray(t_us, dtype=np.float64)
    n = r.size
    t0 = float(t[0]) if n else 0.0

    def _no_op(reason: str, f_sec: float) -> WarmupInfo:
        return WarmupInfo(t_star_us=t0, i_start=0, dropped_frames=0,
                          secular_freq_MHz=f_sec, reason=reason)

    if n < 16:
        return _no_op("too-short", secular_freq_MHz)

    dt_med = float(np.median(np.diff(t)))
    freq_rf = float(freq_rf_MHz) if freq_rf_MHz else Omega_rf / (2.0 * np.pi)
    if dt_med <= 0 or freq_rf <= 0:
        return _no_op("too-short", secular_freq_MHz)

    # ---- R1 低通 + 边缘裁剪（复用 _fft_lowpass 与 n//20 Gibbs 约定）----
    X_sec = _fft_lowpass(r, dt_med, cutoff_per_us=0.5 * freq_rf)
    e = max(n // 20, 1)
    if n - 2 * e < 16:
        return _no_op("too-short", secular_freq_MHz)
    X_e = X_sec[e:n - e]
    t_e = t[e:n - e]
    n_e = X_e.size

    # ---- R2 去趋势 + secular 频率估计 ----
    f_sec = secular_freq_MHz
    resid_sec = X_e - np.polyval(np.polyfit(t_e - t_e[0], X_e, 2), t_e - t_e[0])
    df = 1.0 / (n_e * dt_med)
    band_lo = 2.0 * df
    band_hi = min(0.45 * freq_rf, 2.5 * secular_freq_MHz)
    if band_hi > band_lo and np.any(np.abs(resid_sec) > 0):
        freqs = np.fft.rfftfreq(n_e, d=dt_med)
        spec = np.abs(np.fft.rfft(resid_sec))
        m = (freqs >= band_lo) & (freqs <= band_hi)
        if m.any():
            local_med = float(np.median(spec[m]))
            peak_h = float(spec[m].max())
            if local_med > 0 and peak_h / local_med >= 3.0:
                f_sec = float(freqs[m][np.argmax(spec[m])])
            else:
                logger.warning(
                    "detect_warmup: secular 谱峰不显著(峰/中位=%.1f)，"
                    "回退先验 %.3f MHz", peak_h / max(local_med, 1e-30),
                    secular_freq_MHz,
                )

    # ---- R3 窗几何（50% 重叠，匹配 phase-folding step_us=W/2 约定）----
    if f_sec <= 0:
        f_sec = secular_freq_MHz
    W_n = int(round(n_periods / (f_sec * dt_med)))
    step_n = max(1, W_n // n_overlap)
    n_win = (n_e - W_n) // step_n + 1 if (W_n >= 2 and n_e >= W_n) else 0
    if n_win < 4:
        return _no_op("too-short", f_sec)

    # ---- R4 鲁棒参考（末 25%；P95-P05 抗尖刺）----
    ref = X_e[int(np.floor(0.75 * n_e)):]
    if ref.size < 4:
        return _no_op("too-short", f_sec)
    ref_mean = float(np.median(ref))
    ref_scale = float(np.percentile(ref, 95) - np.percentile(ref, 5))
    ref_mad = _mad(ref)

    # ---- R5 static 守卫（常位移/静止轴 → 无瞬态）----
    sig_scale = float(np.median(np.abs(r))) if np.any(r != 0) else 0.0
    if ref_scale < max(1e-6, 1e-3 * sig_scale):
        return _no_op("static", f_sec)

    # ---- 混合阈：相对项 + 噪声驱动的绝对项（防冷离子过裁剪）----
    noise = _mad(r - X_sec)
    tol_abs = 4.0 * noise * np.sqrt(W_n / max(n, 1))

    # ---- R6 末段稳态守卫 ----
    half = ref.size // 2
    if half >= 2:
        ra, rb = ref[:half], ref[half:]
        if (abs(float(np.median(ra)) - float(np.median(rb))) > tol_rel * ref_scale + tol_abs
                or (ref_mad > 0 and abs(_mad(ra) - _mad(rb)) > tol_rel * ref_mad + tol_abs)):
            return _no_op("unconverged-tail", f_sec)

    # ---- R7 每窗统计 + GOOD 标记 ----
    starts = e + np.arange(n_win) * step_n
    good = np.ones(n_win, dtype=bool)
    thresh = tol_rel * ref_scale + tol_abs
    for k, s in enumerate(starts):
        w = X_sec[s:s + W_n]
        w_mean = float(np.median(w))
        w_scale = float(np.percentile(w, 95) - np.percentile(w, 5))
        if abs(w_mean - ref_mean) > thresh or abs(w_scale - ref_scale) > thresh:
            good[k] = False

    # ---- R8 3 窗中值滤波（杀孤立噪声窗）+ ≥3 连续 bad 段判定 ----
    if n_win >= 3:
        g = good.astype(float)
        pad = np.pad(g, 1, mode="edge")
        good = np.array([np.median(pad[i:i + 3]) for i in range(n_win)]) >= 0.5

    bad_runs: list[tuple[int, int]] = []
    k = 0
    while k < n_win:
        if not good[k]:
            j = k
            while j < n_win and not good[j]:
                j += 1
            if j - k >= 3:
                bad_runs.append((k, j - 1))
            k = j
        else:
            k += 1

    if not bad_runs:
        return WarmupInfo(t_star_us=t0, i_start=0, dropped_frames=0,
                          secular_freq_MHz=f_sec, reason="auto")

    last_end = bad_runs[-1][1]
    if last_end >= n_win - 1:
        return _no_op("unconverged-tail", f_sec)

    # ---- R9 t* = 末个 bad 段之后首个 good 窗的起点 ----
    after = last_end + 1
    s_keep = int(starts[after])
    s_keep = max(0, min(s_keep, n - 1))
    t_star = float(t[s_keep])
    i_start = int(np.searchsorted(t, t_star))
    if i_start >= n:
        i_start = n - 1
    return WarmupInfo(t_star_us=t_star, i_start=i_start, dropped_frames=i_start,
                      secular_freq_MHz=f_sec, reason="auto")


def compute_micromotion(
    r_axis_um: np.ndarray,
    t_us: np.ndarray,
    Omega_rf: float,
    *,
    window_us: float | None = None,
    n_phase_bins: int = 32,
) -> IonAxisResult:
    """
    单离子单轴 micromotion 分析：phase-folding（β(t)）+ 方法 C（q_eff）。

    Parameters
    ----------
    r_axis_um : (T,) 位置序列（µm）
    t_us : (T,) 时间（µs）
    Omega_rf : RF 角频率（rad/µs）
    window_us : 滑窗长度（µs），默认 max(3·T_RF, 0.3)
    n_phase_bins : 相位 bin 数
    """
    r = np.asarray(r_axis_um, dtype=np.float64)
    t = np.asarray(t_us, dtype=np.float64)
    n = r.size
    if n < 16:
        raise ValueError(f"样本数 {n} 不足，phase-folding 需 ≥ 16")

    T_rf_us = 2.0 * np.pi / Omega_rf
    if window_us is None:
        window_us = max(3.0 * T_rf_us, 0.3)
    step_us = window_us / 2.0

    # ---- 方法 C：FFT 低通提取 secular 包络 + 全局 q_eff 回归 ----
    dt_med = float(np.median(np.diff(t)))
    freq_rf = Omega_rf / (2.0 * np.pi)      # cycle/µs
    # 截止取 RF 频率一半：secular（≪ RF）保留，RF 基波（≈ RF）干净去除
    X_sec = _fft_lowpass(r, dt_med, cutoff_per_us=0.5 * freq_rf)
    delta = r - X_sec
    cos_t = np.cos(Omega_rf * t)
    sin_t = np.sin(Omega_rf * t)
    # 裁掉边界区域缓解 FFT 周期延拓引入的 Gibbs 振荡
    trim = max(n // 20, 1)
    sl = slice(trim, n - trim) if (n - 2 * trim) > 4 else slice(None)
    # δr = a·X_sec·cos + b·X_sec·sin （乘性模型，无常数项）
    A = np.column_stack([X_sec[sl] * cos_t[sl], X_sec[sl] * sin_t[sl]])
    coef, *_ = np.linalg.lstsq(A, delta[sl], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    q_eff = 2.0 * np.sqrt(a * a + b * b)
    # 参数 stderr
    resid = delta[sl] - A @ coef
    n_eff = A.shape[0]
    dof = max(n_eff - 2, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        sa, sb = np.sqrt(np.diag(cov))
        # 误差传播 q_eff = 2√(a²+b²)
        denom = a * a + b * b
        q_eff_stderr = (
            2.0 * np.sqrt((a * sa) ** 2 + (b * sb) ** 2) / np.sqrt(denom)
            if denom > 0 else float("nan")
        )
    except np.linalg.LinAlgError:
        q_eff_stderr = float("nan")
    resid_rms = float(np.sqrt(np.mean(resid ** 2)))

    # ---- Phase-folding：滑窗 β(t) ----
    t0 = t[0]
    t_end = t[-1]
    centers = np.arange(t0 + window_us / 2, t_end - window_us / 2 + 1e-12, step_us)
    bins_edge = np.linspace(0.0, 2.0 * np.pi, n_phase_bins + 1)
    bin_phi_center = 0.5 * (bins_edge[:-1] + bins_edge[1:])

    win_list: list[WindowMM] = []
    for tc in centers:
        mask = (t >= tc - window_us / 2) & (t <= tc + window_us / 2)
        if mask.sum() < 3 * n_phase_bins:
            continue
        tw = t[mask]
        rw = r[mask]
        phi = np.mod(Omega_rf * tw, 2.0 * np.pi)
        idx = np.clip(np.digitize(phi, bins_edge) - 1, 0, n_phase_bins - 1)
        r_bin = np.array([rw[idx == k].mean() if np.any(idx == k) else np.nan
                          for k in range(n_phase_bins)])
        good = np.isfinite(r_bin)
        if good.sum() < 3:
            continue
        # 拟合 r(φ) = r̄ + β_c·cos φ + β_s·sin φ
        M = np.column_stack([
            np.ones(good.sum()),
            np.cos(bin_phi_center[good]),
            np.sin(bin_phi_center[good]),
        ])
        c, *_ = np.linalg.lstsq(M, r_bin[good], rcond=None)
        r_bar, bc, bs = float(c[0]), float(c[1]), float(c[2])
        beta = np.hypot(bc, bs)
        win_list.append(WindowMM(
            t_center_us=float(tc),
            r_secular_um=r_bar,
            beta_um=float(beta),
            phase_rad=float(np.arctan2(bs, bc)),
            ptp_um=float(2.0 * beta),
        ))

    if not win_list:
        raise ValueError(
            "phase-folding 无有效窗：窗长可能过小或样本不足，请增大 --window-us"
        )

    t_w = np.array([w.t_center_us for w in win_list])
    beta_t = np.array([w.beta_um for w in win_list])
    r_sec_t = np.array([w.r_secular_um for w in win_list])
    phase_t = np.array([w.phase_rad for w in win_list])

    return IonAxisResult(
        t_windows=t_w,
        beta_t=beta_t,
        r_secular_t=r_sec_t,
        phase_t=phase_t,
        windows=win_list,
        q_eff=q_eff,
        q_eff_stderr=q_eff_stderr,
        secular_envelope_um=X_sec,
        residual_rms_um=resid_rms,
    )


# ============================== 多离子批处理 ==============================

@dataclass
class MicromotionReport:
    """整次连续采样的多离子 micromotion 报告。"""

    trajectory: TrajectoryData
    results: dict[tuple[int, str], IonAxisResult]   # (ion_idx, axis)
    q_eff: np.ndarray                               # (N, 3)
    q_eff_stderr: np.ndarray                        # (N, 3)
    axes: tuple[str, ...]
    ions: tuple[int, ...]
    window_us: float
    n_phase_bins: int


def analyze_run(
    run_dir: str | Path,
    *,
    csv_path: str | Path | None = None,
    config_path: str | Path | None = None,
    species: str = "Ba135+",
    axes: tuple[str, ...] = ("x", "y", "z"),
    ions: tuple[int, ...] | list[int] | None = None,
    window_us: float | None = None,
    n_phase_bins: int = 32,
    secular_freq_MHz: float = _DEFAULT_SECULAR_FREQ_MHZ,
    freq_rf_MHz: float | None = None,
    check_sampling: bool = True,
    warmup_us: float | None = None,
    trim_start_us: float | None = None,
    no_auto_trim: bool = False,
    warmup_tol: float = 0.1,
    warmup_periods: int = 3,
) -> MicromotionReport:
    """
    加载一次连续采样并对每个指定离子的每个轴计算 micromotion。

    Parameters
    ----------
    csv_path, config_path : 用于解析 RF 频率（正常用法必传 config）
    ions : 待分析离子索引；None 表示全部 N 个离子
    freq_rf_MHz : 显式 RF 频率覆盖（测试钩子，优先于 config）
    check_sampling : 是否执行采样质量校验（测试可关闭）
    warmup_us : 手动丢弃前 N µs（相对 t[0]）；优先级低于 trim_start_us
    trim_start_us : 手动指定稳态起点（绝对时刻 µs）；优先级最高
    no_auto_trim : 关闭自动瞬态检测（不裁剪）；默认 False（自动检测开）
    warmup_tol : 自动检测相对收敛容差（默认 0.1）
    warmup_periods : 自动检测窗覆盖的 secular 周期数（默认 3）

    warmup 起点优先级：``trim_start_us`` > ``warmup_us`` > ``no_auto_trim`` > 自动检测。
    默认（无任何 warmup 参数）逐 (ion,axis) 自动检测瞬态收敛点并裁剪；检测器对干净
    稳态/静止轴返回 ``i_start=0``（no-op），结果与不裁剪一致。
    """
    traj = load_continuous_sampling(
        run_dir, csv_path=csv_path, config_path=config_path,
        species=species, secular_freq_MHz=secular_freq_MHz,
        freq_rf_MHz=freq_rf_MHz, check_sampling=check_sampling,
    )
    N = traj.n_ions
    ion_idx = tuple(range(N)) if ions is None else tuple(int(i) for i in ions)
    for i in ion_idx:
        if not (0 <= i < N):
            raise ValueError(f"离子索引 {i} 超出 [0, {N})")

    results: dict[tuple[int, str], IonAxisResult] = {}
    q_eff = np.full((N, 3), np.nan)
    q_err = np.full((N, 3), np.nan)

    # 裁后须保留足够帧让 compute_micromotion 产出 ≥1 个 phase-folding 窗
    eff_window_us = window_us if window_us is not None else max(3.0 * 2 * np.pi / traj.Omega_rf, 0.3)
    min_remain = max(16, int(np.ceil(2.0 * eff_window_us / traj.dt_us))) if traj.dt_us > 0 else 16

    for i in ion_idx:
        for ax in axes:
            ai = _AXIS_INDEX[ax]
            r_full = traj.r_um[:, i, ai]
            t_full = traj.t_us
            n_full = t_full.size

            # ---- warmup 起点解析 ----
            if trim_start_us is not None:
                i_start = int(np.searchsorted(t_full, float(trim_start_us)))
                reason = "manual-abs"
            elif warmup_us is not None:
                i_start = int(np.searchsorted(t_full, float(t_full[0]) + float(warmup_us)))
                reason = "manual-rel"
            elif no_auto_trim:
                i_start = 0
                reason = "none"
            else:
                info = detect_warmup(
                    r_full, t_full, traj.Omega_rf,
                    secular_freq_MHz=secular_freq_MHz,
                    freq_rf_MHz=traj.freq_rf_MHz,
                    n_periods=warmup_periods, tol_rel=warmup_tol,
                )
                i_start = info.i_start
                reason = info.reason
                logger.debug(
                    "warmup ion %d axis %s: t*=%.3fµs drop=%d (%s)",
                    i, ax, info.t_star_us, info.dropped_frames, info.reason,
                )

            # ---- 钳制：裁后剩余帧不足时回退 ----
            if n_full - i_start < min_remain:
                i_start = max(0, n_full - min_remain)
                reason = reason + "+clamped"

            res = compute_micromotion(
                r_full[i_start:], t_full[i_start:], traj.Omega_rf,
                window_us=window_us, n_phase_bins=n_phase_bins,
            )
            res.t_star_us = float(t_full[i_start])
            res.dropped_frames = int(i_start)
            res.warmup_reason = reason
            results[(i, ax)] = res
            q_eff[i, ai] = res.q_eff
            q_err[i, ai] = res.q_eff_stderr

    w = window_us if window_us is not None else max(3.0 * 2 * np.pi / traj.Omega_rf, 0.3)
    return MicromotionReport(
        trajectory=traj,
        results=results,
        q_eff=q_eff,
        q_eff_stderr=q_err,
        axes=tuple(axes),
        ions=ion_idx,
        window_us=float(w),
        n_phase_bins=n_phase_bins,
    )


# ============================== 交叉验证 ==============================

@dataclass
class CrossCheck:
    """测量 q_eff 与理论 q（trap_stability）对比。"""

    q_theory: dict[str, float]                 # {"x":..,"y":..,"z":..}
    q_measured_median: dict[str, float]        # 各离子 q_eff 中位数
    ratio: dict[str, float]                    # measured_median / theory
    center_um: tuple[float, float, float]
    is_stable: bool
    freq_rf_MHz: float
    species: str


def _build_field_inputs(
    csv_path: str, config_path: str, species: str,
    smooth_axes: tuple[str, ...] | None = ("z",),
    smooth_sg: tuple[int, int] = (11, 3),
):
    """复用 trap_stability.cli 的插值器构建逻辑。"""
    from FieldConfiguration.constants import init_from_config
    from FieldConfiguration.ion_species import ION_SPECIES
    from FieldConfiguration.loader import build_voltage_list
    from FieldParser.csv_reader import read as read_csv
    from FieldParser.calc_field import calc_potential, calc_field
    from field_visualize.core import apply_savgol_smooth

    sp = ION_SPECIES[species]
    cfg, config_dict = init_from_config(config_path, mass_amu=sp.mass_amu)
    grid_coord, grid_voltage = read_csv(
        csv_path, None, normalize=True, dl=cfg.dl, dV=cfg.dV
    )
    if smooth_axes:
        grid_voltage = apply_savgol_smooth(
            grid_coord, grid_voltage, smooth_axes,
            window_length=smooth_sg[0], polyorder=smooth_sg[1],
        )
    potential_interps = calc_potential(grid_coord, grid_voltage)
    field_interps = calc_field(grid_coord, grid_voltage)
    if config_dict:
        n_voltage = grid_voltage.shape[1]
        voltage_list = build_voltage_list(config_dict, n_voltage, cfg)
    else:
        from utils import Voltage, constant
        voltage_list = [Voltage(f"U{i+1}", 0.0, constant(1.0), 0.0)
                        for i in range(grid_voltage.shape[1])]
    return cfg, sp, potential_interps, field_interps, voltage_list


def cross_check_q(
    report: MicromotionReport,
    *,
    csv_path: str | Path,
    config_path: str | Path,
    species: str = "Ba135+",
    center_um: tuple[float, float, float] | None = None,
    fit_range_um: tuple = ((-50, 50), (-20, 20), (-150, 150)),
    smooth_axes: tuple[str, ...] | None = ("z",),
    smooth_sg: tuple[int, int] = (11, 3),
) -> CrossCheck:
    """
    用 trap_stability 从场几何算理论 q，与 report 中各离子 q_eff 中位数对比。

    冷单粒子在阱中心时 ratio ≈ 1；晶格库仑排斥致离子偏离 RF 零场点时
    ratio > 1（excess micromotion）。

    注：若 ``report`` 由 warmup 裁剪生成，各离子 q_eff 可能来自不同的稳态时间窗
    （per-ion 各自裁掉瞬态），中位数仍定义良好——这正是 excess micromotion 诊断
    所需（每个离子取其各自收敛后的 q_eff）。

    smooth_axes / smooth_sg 透传给场插值器构建（与 trap_stability CLI 约定一致）。
    """
    from trap_stability.stability import (
        compute_stability_from_field, find_trap_center,
    )

    cfg, sp, potential_interps, field_interps, voltage_list = _build_field_inputs(
        str(csv_path), str(config_path), species,
        smooth_axes=smooth_axes, smooth_sg=smooth_sg,
    )
    if center_um is None:
        center_um = find_trap_center(
            potential_interps, field_interps, voltage_list, cfg,
        )
        logger.info("自动检测阱中心: (%.2f, %.2f, %.2f) µm", *center_um)

    res = compute_stability_from_field(
        potential_interps=potential_interps,
        field_interps=field_interps,
        voltage_list=voltage_list,
        cfg=cfg,
        species=sp,
        center_um=center_um,
        fit_range_um=fit_range_um,
        n_pts=200,
        fit_degree=2,
    )
    q_theory = {"x": res.q_x, "y": res.q_y, "z": res.q_z}

    q_meas: dict[str, float] = {}
    ratio: dict[str, float] = {}
    for ax in report.axes:
        ai = _AXIS_INDEX[ax]
        col = report.q_eff[:, ai]
        col = col[np.isfinite(col)]
        med = float(np.median(col)) if col.size else float("nan")
        q_meas[ax] = med
        qt = q_theory[ax]
        ratio[ax] = med / qt if qt != 0 else float("nan")

    return CrossCheck(
        q_theory=q_theory,
        q_measured_median=q_meas,
        ratio=ratio,
        center_um=tuple(float(v) for v in center_um),
        is_stable=bool(res.is_stable),
        freq_rf_MHz=res.freq_rf_MHz,
        species=species,
    )


# ============================== 序列化 ==============================

def report_to_dict(report: MicromotionReport, cross: CrossCheck | None = None) -> dict:
    """报告转 JSON 可序列化字典（不含完整 window 序列以控制体积）。"""
    d = {
        "run_dir": str(report.trajectory.run_dir),
        "freq_rf_MHz": report.trajectory.freq_rf_MHz,
        "Omega_rf_rad_per_us": report.trajectory.Omega_rf,
        "dt_ns": report.trajectory.dt_us * 1e3,
        "n_ions": report.trajectory.n_ions,
        "axes": list(report.axes),
        "ions": list(report.ions),
        "window_us": report.window_us,
        "n_phase_bins": report.n_phase_bins,
        "q_eff": report.q_eff.tolist(),
        "q_eff_stderr": report.q_eff_stderr.tolist(),
    }
    # per-ion per-axis 汇总（中位 β、q_eff）
    summary = {}
    for (i, ax), res in report.results.items():
        summary.setdefault(str(i), {})[ax] = {
            "q_eff": float(res.q_eff) if np.isfinite(res.q_eff) else None,
            "q_eff_stderr": float(res.q_eff_stderr) if np.isfinite(res.q_eff_stderr) else None,
            "beta_median_um": float(np.median(res.beta_t)),
            "beta_max_um": float(np.max(res.beta_t)),
            "residual_rms_um": float(res.residual_rms_um)
            if np.isfinite(res.residual_rms_um) else None,
            "t_star_us": float(res.t_star_us)
            if np.isfinite(res.t_star_us) else None,
            "dropped_frames": int(res.dropped_frames),
            "warmup_reason": str(res.warmup_reason),
        }
    d["per_ion"] = summary
    if cross is not None:
        d["cross_check"] = {
            "q_theory": cross.q_theory,
            "q_measured_median": cross.q_measured_median,
            "ratio_measured_over_theory": cross.ratio,
            "center_um": list(cross.center_um),
            "is_stable": cross.is_stable,
            "species": cross.species,
        }
    return d
