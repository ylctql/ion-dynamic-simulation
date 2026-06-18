"""
tests for motion_analysis.micromotion — 数值测量 RF micromotion 与 q_eff。

核心验证：合成 r(t)=X_sec(t)·[1+(q/2)cos(Ωt)]，回收已知 q。
"""
import numpy as np
import pytest

from motion_analysis.micromotion import (
    IonAxisResult,
    MicromotionReport,
    TrajectoryData,
    WarmupInfo,
    analyze_run,
    compute_micromotion,
    detect_warmup,
    load_continuous_sampling,
    report_to_dict,
)
from motion_analysis.plots import plot_lattice_micromotion

F_RF_MHZ = 35.0
OMEGA_RF = 2.0 * np.pi * F_RF_MHZ   # rad/µs


def _signal_modulated(q_true, n=20000, dt_us=0.001, a_sec=50.0,
                      f_sec_mhz=0.5, seed=0):
    """r(t) = X_sec(t)·[1 + (q/2)cos(Ωt)]，X_sec 为 secular 调制包络。"""
    t = np.arange(n) * dt_us
    X_sec = a_sec * np.cos(2 * np.pi * f_sec_mhz * t)
    r = X_sec * (1.0 + (q_true / 2.0) * np.cos(OMEGA_RF * t))
    return t, r, X_sec


def _signal_constant_offset(q_true, R0=40.0, n=20000, dt_us=0.001):
    """r(t) = R0·[1 + (q/2)cos(Ωt)]，常位移（离子静止于 RF null 外）。"""
    t = np.arange(n) * dt_us
    r = R0 * (1.0 + (q_true / 2.0) * np.cos(OMEGA_RF * t))
    return t, r


def _signal_transient_then_steady(q_true=0.3, q_wrong=0.9, a_sec=50.0,
                                  f_sec_mhz=0.5, n_trans=16000, n_steady=24000,
                                  dt_us=0.001, amp_scale=2.0):
    """前段瞬态（amp_scale 倍 secular 振幅 + 错误 q，高杠杆污染）+ 后段稳态（正确 q）。

    瞬态段放大 secular 包络 → 方法 C 回归中 X_sec² 加权致瞬态点高杠杆，未裁时
    q_eff 被拉向 q_wrong；裁后回收 q_true。模拟绝热破坏/非谐-振幅耦合瞬态。
    """
    t1 = np.arange(n_trans) * dt_us
    t2 = n_trans * dt_us + np.arange(n_steady) * dt_us
    t = np.concatenate([t1, t2])
    X1 = amp_scale * a_sec * np.cos(2 * np.pi * f_sec_mhz * t1)
    X2 = a_sec * np.cos(2 * np.pi * f_sec_mhz * t2)
    r1 = X1 * (1.0 + (q_wrong / 2.0) * np.cos(OMEGA_RF * t1))
    r2 = X2 * (1.0 + (q_true / 2.0) * np.cos(OMEGA_RF * t2))
    return t, np.concatenate([r1, r2]), np.concatenate([X1, X2])


def _signal_growing_tail(q_true=0.3, n=24000, dt_us=0.001, a_sec=50.0,
                         f_sec_mhz=0.5):
    """secular 振幅在后 40% 线性增长（末段未冷却/未稳 → 不可定义收敛参考）。"""
    t = np.arange(n) * dt_us
    grow = np.clip((t - 0.6 * t[-1]) / (0.4 * t[-1]), 0.0, 1.0)
    env = a_sec * (1.0 + 1.5 * grow)
    X_sec = env * np.cos(2 * np.pi * f_sec_mhz * t)
    r = X_sec * (1.0 + (q_true / 2.0) * np.cos(OMEGA_RF * t))
    return t, r, X_sec


# ============== compute_micromotion: 合成数据回收 q ==============

class TestRecoverQ:
    def test_constant_offset_recovers_q(self):
        """常位移：q_eff 应精确回收 q。"""
        for q_true in (0.1, 0.3, 0.6):
            t, r = _signal_constant_offset(q_true)
            res = compute_micromotion(r, t, OMEGA_RF)
            assert res.q_eff == pytest.approx(q_true, rel=0.01), (
                f"q_true={q_true}, got q_eff={res.q_eff}"
            )

    def test_secular_modulated_recovers_q(self):
        """secular 调制包络下 q_eff 仍回收 q。"""
        q_true = 0.3
        t, r, _ = _signal_modulated(q_true)
        res = compute_micromotion(r, t, OMEGA_RF)
        assert res.q_eff == pytest.approx(q_true, rel=0.02)

    def test_zero_q(self):
        """无 micromotion (q=0) 时 q_eff ≈ 0。"""
        t, r = _signal_constant_offset(0.0)
        res = compute_micromotion(r, t, OMEGA_RF)
        assert abs(res.q_eff) < 1e-3

    def test_negative_q_amplitude(self):
        """q 为负（相位翻转）时 |q_eff| 回收 |q|。"""
        t, r = _signal_constant_offset(-0.4)
        res = compute_micromotion(r, t, OMEGA_RF)
        assert abs(res.q_eff) == pytest.approx(0.4, rel=0.01)

    def test_beta_tracks_secular(self):
        """phase-folding β(t) ≈ (q/2)·|X_sec(t)|。"""
        q_true = 0.3
        t, r, X_sec = _signal_modulated(q_true)
        res = compute_micromotion(r, t, OMEGA_RF)
        # 在窗中心插值 |X_sec|，仅在显著位移处比较（避开过零点数值噪声）
        Xw = np.interp(res.t_windows, t, np.abs(X_sec))
        mask = Xw > 5.0
        assert mask.sum() > 10
        ratio = res.beta_t[mask] / Xw[mask]
        assert np.median(ratio) == pytest.approx(q_true / 2.0, rel=0.05)

    def test_returns_expected_fields(self):
        t, r = _signal_constant_offset(0.2)
        res = compute_micromotion(r, t, OMEGA_RF)
        assert isinstance(res, IonAxisResult)
        assert res.t_windows.shape == res.beta_t.shape
        assert res.secular_envelope_um is not None
        assert res.beta_t.ndim == 1 and res.beta_t.size > 0


# ============== load_continuous_sampling: 异常路径与采样校验 ==============

class TestLoad:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="不存在"):
            load_continuous_sampling(tmp_path / "nope", config_path="x")

    def test_no_frames_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="frame"):
            load_continuous_sampling(tmp_path, freq_rf_MHz=F_RF_MHZ)

    def _write_frames(self, tmp_path, n_frames, dt_us, r_arr, t0=0.0):
        for k in range(n_frames):
            np.savez(
                tmp_path / f"frame{k}.npz",
                r=r_arr, v=np.zeros_like(r_arr),
                t_us=t0 + k * dt_us,
            )

    def test_sampling_rate_too_low_raises(self, tmp_path):
        """每 RF 周期点数 < 8 时 raise。"""
        N = 2
        r = np.zeros((N, 3))
        self._write_frames(tmp_path, 10, dt_us=0.01, r_arr=r)  # 10 ns/帧
        with pytest.raises(ValueError, match="采样率不足"):
            load_continuous_sampling(tmp_path, freq_rf_MHz=F_RF_MHZ)

    def test_duration_too_short_raises(self, tmp_path):
        """总时长不足时 raise（采样率本身足够）。"""
        N = 1
        r = np.zeros((N, 3))
        self._write_frames(tmp_path, 20, dt_us=0.001, r_arr=r)  # 0.02 µs 总
        with pytest.raises(ValueError, match="总时长不足"):
            load_continuous_sampling(tmp_path, freq_rf_MHz=F_RF_MHZ)

    def test_load_ok(self, tmp_path):
        """合规数据正常加载。"""
        N = 2
        r = np.tile([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (1, 1))
        # 6 µs 数据，dt=1ns，采样合规
        self._write_frames(tmp_path, 6000, dt_us=0.001, r_arr=r)
        traj = load_continuous_sampling(
            tmp_path, freq_rf_MHz=F_RF_MHZ, secular_freq_MHz=1.0,
        )
        assert isinstance(traj, TrajectoryData)
        assert traj.n_ions == N
        assert traj.r_um.shape == (6000, N, 3)
        assert traj.freq_rf_MHz == F_RF_MHZ
        assert traj.dt_us == pytest.approx(0.001)


# ============== analyze_run: 多离子批处理 ==============

class TestAnalyzeRun:
    def test_multi_ion_qeff(self, tmp_path):
        """2 离子不同 q，analyze_run 批量计算，q_eff 形状与值正确。"""
        N = 2
        n_frames = 4000
        dt_us = 0.001
        q_list = [0.25, 0.45]
        R0_list = [30.0, 50.0]
        for k in range(n_frames):
            t = k * dt_us
            r = np.zeros((N, 3))
            for i in range(N):
                r[i, 0] = R0_list[i] * (1.0 + (q_list[i] / 2.0) * np.cos(OMEGA_RF * t))
            np.savez(tmp_path / f"frame{k}.npz",
                     r=r, v=np.zeros_like(r), t_us=t)
        report = analyze_run(
            tmp_path, freq_rf_MHz=F_RF_MHZ, check_sampling=False,
            axes=("x", "y", "z"), window_us=0.1,
        )
        assert isinstance(report, MicromotionReport)
        assert report.q_eff.shape == (N, 3)
        for i in range(N):
            assert report.q_eff[i, 0] == pytest.approx(q_list[i], rel=0.02), (
                f"ion {i}: q={q_list[i]}, got {report.q_eff[i, 0]}"
            )
        # y/z 轴无 micromotion（合成仅 x 有），q_eff ≈ 0
        for i in range(N):
            assert abs(report.q_eff[i, 1]) < 1e-2
            assert abs(report.q_eff[i, 2]) < 1e-2

    def test_report_to_dict(self, tmp_path):
        N = 1
        n_frames = 4000
        dt_us = 0.001
        for k in range(n_frames):
            t = k * dt_us
            r = np.array([[40.0 * (1.0 + 0.15 * np.cos(OMEGA_RF * t)), 0.0, 0.0]])
            np.savez(tmp_path / f"frame{k}.npz",
                     r=r, v=np.zeros_like(r), t_us=t)
        report = analyze_run(
            tmp_path, freq_rf_MHz=F_RF_MHZ, check_sampling=False,
            axes=("x",), window_us=0.1,
        )
        d = report_to_dict(report)
        assert "q_eff" in d and "per_ion" in d
        assert d["n_ions"] == N
        assert len(d["q_eff"]) == N

    def test_report_to_dict_serializes_rf_null(self, tmp_path):
        """带 cross 的 report_to_dict 应序列化 rf_null_um 字段。"""
        from motion_analysis.micromotion import CrossCheck
        N = 1
        n_frames = 4000
        dt_us = 0.001
        for k in range(n_frames):
            t = k * dt_us
            r = np.array([[40.0 * (1.0 + 0.15 * np.cos(OMEGA_RF * t)), 0.0, 0.0]])
            np.savez(tmp_path / f"frame{k}.npz",
                     r=r, v=np.zeros_like(r), t_us=t)
        report = analyze_run(
            tmp_path, freq_rf_MHz=F_RF_MHZ, check_sampling=False,
            axes=("x",), window_us=0.1,
        )
        cross = CrossCheck(
            q_theory={"x": 0.3, "y": 0.0, "z": 0.0},
            q_measured_median={"x": 0.3, "y": 0.0, "z": 0.0},
            ratio={"x": 1.0, "y": 0.0, "z": 0.0},
            center_um=(-40.0, 0.0, 0.0), is_stable=True,
            freq_rf_MHz=F_RF_MHZ, species="Ba135+",
            rf_null_um=(0.0, 0.0, 0.0),
        )
        d = report_to_dict(report, cross)
        assert d["cross_check"]["rf_null_um"] == [0.0, 0.0, 0.0]
        assert d["cross_check"]["center_um"] == [-40.0, 0.0, 0.0]


# ============== detect_warmup: 瞬态收敛检测 ==============

class TestWarmup:
    def test_transient_trim_recovers_q(self):
        """(a) 瞬态高杠杆污染：自动检测裁剪后 q_eff 回收优于不裁。"""
        q_true = 0.3
        t, r, _X = _signal_transient_then_steady(q_true=q_true)
        # 不裁：瞬态高杠杆把 q_eff 拉向 q_wrong=0.9
        res_full = compute_micromotion(r, t, OMEGA_RF)
        assert abs(res_full.q_eff - q_true) > 0.05, (
            f"期望瞬态偏置 q_eff，got {res_full.q_eff}"
        )
        # 自动检测识别瞬态
        info = detect_warmup(r, t, OMEGA_RF, freq_rf_MHz=F_RF_MHZ)
        assert isinstance(info, WarmupInfo)
        assert info.reason == "auto"
        assert info.dropped_frames > 0
        # 裁后回收 q_true，且严格优于不裁
        res_trim = compute_micromotion(r[info.i_start:], t[info.i_start:], OMEGA_RF)
        assert res_trim.q_eff == pytest.approx(q_true, rel=0.05)
        assert abs(res_trim.q_eff - q_true) < abs(res_full.q_eff - q_true)

    def test_clean_steady_no_trim(self):
        """(b) 干净稳态信号从头收敛 → t*=t[0]，非破坏。"""
        t, r, _X = _signal_modulated(0.3)
        info = detect_warmup(r, t, OMEGA_RF, freq_rf_MHz=F_RF_MHZ)
        assert info.dropped_frames == 0
        assert info.i_start == 0
        assert info.reason == "auto"

    def test_static_axis(self):
        """(c) 常位移轴（无 secular 调制）→ static 守卫 no-op。"""
        t, r = _signal_constant_offset(0.3)
        info = detect_warmup(r, t, OMEGA_RF, freq_rf_MHz=F_RF_MHZ)
        assert info.reason == "static"
        assert info.dropped_frames == 0

    def test_unconverged_tail(self):
        """(d) 末段振幅仍增长 → unconverged-tail 保守不裁。"""
        t, r, _X = _signal_growing_tail()
        info = detect_warmup(r, t, OMEGA_RF, freq_rf_MHz=F_RF_MHZ)
        assert info.reason == "unconverged-tail"
        assert info.dropped_frames == 0

    def test_secular_freq_estimated(self):
        """(e) secular 频率从谱峰估计，回收真实值（比先验更接近真值）。"""
        f_true = 0.8
        t, r, _X = _signal_modulated(0.3, n=40000, f_sec_mhz=f_true)
        info = detect_warmup(r, t, OMEGA_RF, freq_rf_MHz=F_RF_MHZ,
                             secular_freq_MHz=0.5)
        assert info.secular_freq_MHz == pytest.approx(f_true, rel=0.05)

    def test_secular_freq_fallback(self):
        """(f) 平稳宽带信号无显著 secular 谱峰 → 回退先验。"""
        rng = np.random.default_rng(0)
        n = 24000
        t = np.arange(n) * 0.001
        r = rng.standard_normal(n) * 10.0  # 平稳宽带，无主导谱峰
        info = detect_warmup(r, t, OMEGA_RF, freq_rf_MHz=F_RF_MHZ,
                             secular_freq_MHz=0.5)
        assert info.secular_freq_MHz == pytest.approx(0.5, abs=1e-9)

    def test_trim_robust_to_scattered_noise(self):
        """(g) 瞬态+稳态信号加中等噪声（散落单窗涨落），裁剪点仍稳定锁定早期瞬态。

        H1 鲁棒性：中值滤波 + ≥3 连续 bad 规则使散落噪声不破坏瞬态判定。
        """
        rng = np.random.default_rng(0)
        t, r, _X = _signal_transient_then_steady(q_true=0.3)
        info0 = detect_warmup(r, t, OMEGA_RF, freq_rf_MHz=F_RF_MHZ)
        # 叠加 ~5% secular 振幅的噪声，制造散落的单窗统计涨落
        r_noisy = r + rng.standard_normal(r.size) * 2.5
        info1 = detect_warmup(r_noisy, t, OMEGA_RF, freq_rf_MHz=F_RF_MHZ)
        assert info0.reason == "auto" and info1.reason == "auto"
        assert info0.dropped_frames > 0 and info1.dropped_frames > 0
        # 噪声下裁剪点仍在同一早期区域（容差一个窗步 step≈3000 帧）
        assert abs(info1.i_start - info0.i_start) <= 3000

    def test_analyze_run_clamp_no_crash(self, tmp_path):
        """(h) 手动 warmup 过大裁到不足 → 钳制保护，不崩且 q_eff 可算。"""
        N = 1
        n_frames = 4000
        dt_us = 0.001
        q = 0.3
        for k in range(n_frames):
            tk = k * dt_us
            r = np.array([[40.0 * (1.0 + (q / 2.0) * np.cos(OMEGA_RF * tk)), 0.0, 0.0]])
            np.savez(tmp_path / f"frame{k}.npz",
                     r=r, v=np.zeros_like(r), t_us=tk)
        report = analyze_run(
            tmp_path, freq_rf_MHz=F_RF_MHZ, check_sampling=False,
            axes=("x",), window_us=0.1, warmup_us=3.9,
        )
        res = report.results[(0, "x")]
        assert res.warmup_reason.endswith("+clamped")
        assert np.isfinite(res.q_eff)
        assert res.dropped_frames < n_frames  # 钳制后保留了尾部

    def test_compute_micromotion_warmup_defaults(self):
        """(i) 直调 compute_micromotion 不设 warmup → 新字段为默认值。"""
        t, r = _signal_constant_offset(0.2)
        res = compute_micromotion(r, t, OMEGA_RF)
        assert np.isnan(res.t_star_us)
        assert res.dropped_frames == 0
        assert res.warmup_reason == "none"

    def test_report_to_dict_has_warmup_fields(self, tmp_path):
        """per_ion 序列化含 warmup 三字段。"""
        N = 1
        n_frames = 4000
        dt_us = 0.001
        for k in range(n_frames):
            tk = k * dt_us
            r = np.array([[40.0 * (1.0 + 0.15 * np.cos(OMEGA_RF * tk)), 0.0, 0.0]])
            np.savez(tmp_path / f"frame{k}.npz",
                     r=r, v=np.zeros_like(r), t_us=tk)
        report = analyze_run(
            tmp_path, freq_rf_MHz=F_RF_MHZ, check_sampling=False,
            axes=("x",), window_us=0.1,
        )
        d = report_to_dict(report)
        entry = d["per_ion"]["0"]["x"]
        assert "t_star_us" in entry
        assert "dropped_frames" in entry
        assert "warmup_reason" in entry


# ============== plot_lattice_micromotion: 晶格 x 方向 micromotion 可视化 ==============

class TestLatticePlot:
    def _build_report(self, tmp_path, q_list, R0_list, z_list):
        """N 离子沿 z 排列，x 方向常位移 micromotion。

        常位移（无 secular 调制）→ detect_warmup static no-op，平衡位置 x_eq=R0，
        β ≈ (q/2)·R0，竖线总长 2β = q·R0。
        """
        N = len(q_list)
        n_frames = 4000
        dt_us = 0.001
        for k in range(n_frames):
            t = k * dt_us
            r = np.zeros((N, 3))
            for i in range(N):
                r[i, 0] = R0_list[i] * (1.0 + (q_list[i] / 2.0) * np.cos(OMEGA_RF * t))
                r[i, 2] = z_list[i]
            np.savez(tmp_path / f"frame{k}.npz", r=r, v=np.zeros_like(r), t_us=t)
        return analyze_run(
            tmp_path, freq_rf_MHz=F_RF_MHZ, check_sampling=False,
            axes=("x",), window_us=0.1,
        )

    @staticmethod
    def _vertical_lines(ax):
        """提取竖线 (zc, y_lo, y_hi)：xdata 两点相等、ydata 不等（排除水平参考线）。"""
        out = []
        for ln in ax.get_lines():
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if (len(xd) == 2 and len(yd) == 2
                    and np.allclose(xd[0], xd[1])
                    and not np.allclose(yd[0], yd[1])):
                out.append((float(xd[0]), float(yd[0]), float(yd[1])))
        return out

    @staticmethod
    def _vlines_by_style(ax, style):
        """提取指定线型的竖线 (zc, y_lo, y_hi)：linestyle==style 且竖直（区分数值实线 / 理论虚线；RF null 水平虚线被排除）。"""
        out = []
        for ln in ax.get_lines():
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if (ln.get_linestyle() == style and len(xd) == 2 and len(yd) == 2
                    and np.allclose(xd[0], xd[1])
                    and not np.allclose(yd[0], yd[1])):
                out.append((float(xd[0]), float(yd[0]), float(yd[1])))
        return out

    @staticmethod
    def _mock_cross(q_x=0.3):
        from motion_analysis.micromotion import CrossCheck
        return CrossCheck(
            q_theory={"x": q_x, "y": 0.0, "z": 0.0},
            q_measured_median={"x": q_x, "y": 0.0, "z": 0.0},
            ratio={"x": 1.0, "y": 0.0, "z": 0.0},
            center_um=(0.0, 0.0, 0.0), is_stable=True,
            freq_rf_MHz=F_RF_MHZ, species="Ba135+",
        )

    def test_plot_runs_and_line_count(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.25, 0.45], R0_list=[30.0, 50.0], z_list=[0.0, 20.0],
        )
        fig = plot_lattice_micromotion(report)
        assert fig is not None
        assert len(self._vertical_lines(fig.axes[0])) == 2
        plt.close(fig)

    def test_line_length_tracks_amplitude(self, tmp_path):
        """R0 更大（更偏离 RF null）的离子竖线更长；竖线中心=末端帧瞬时位置（非均值）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        fig = plot_lattice_micromotion(report)   # 默认 amp_stat="last"
        vlines = sorted(self._vertical_lines(fig.axes[0]))  # 按 z 升序
        lengths = [abs(v[2] - v[1]) for v in vlines]
        # R0=60 离子竖线总长 ≈ q·R0=18，远大于 R0=20 的 ≈6
        assert lengths[1] > lengths[0] * 2
        # 竖线中心 = 末端帧瞬时 x 位置（含 micromotion 偏移，非时间均值 R0）
        x_last = report.trajectory.r_um[-1, :, 0]
        for (zc, ylo, yhi), x_end in zip(vlines, x_last):
            assert 0.5 * (ylo + yhi) == pytest.approx(x_end, abs=1e-6)
        plt.close(fig)

    def test_amp_stat_max(self, tmp_path):
        """amp_stat='max' 不崩，仍每离子一条竖线。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        fig = plot_lattice_micromotion(report, amp_stat="max")
        assert len(self._vertical_lines(fig.axes[0])) == 2
        plt.close(fig)

    def test_missing_rf_axis_no_crash(self, tmp_path):
        """report 未分析 rf_axis（--axes 不含 x）→ 不崩，无竖线，显示提示。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        N = 2
        n_frames = 4000
        dt_us = 0.001
        for k in range(n_frames):
            t = k * dt_us
            r = np.zeros((N, 3))
            r[0, 2] = 0.0
            r[1, 2] = 20.0
            np.savez(tmp_path / f"frame{k}.npz", r=r, v=np.zeros_like(r), t_us=t)
        report = analyze_run(
            tmp_path, freq_rf_MHz=F_RF_MHZ, check_sampling=False,
            axes=("z",), window_us=0.1,
        )
        fig = plot_lattice_micromotion(report)   # 默认 rf_axis=x，report 无 x 数据
        assert fig is not None
        assert len(self._vertical_lines(fig.axes[0])) == 0
        plt.close(fig)

    def test_invalid_args_raise(self, tmp_path):
        """非法 rf_axis / amp_stat 抛 ValueError。"""
        report = self._build_report(
            tmp_path, q_list=[0.3], R0_list=[30.0], z_list=[0.0],
        )
        with pytest.raises(ValueError, match="rf_axis"):
            plot_lattice_micromotion(report, rf_axis="w")
        with pytest.raises(ValueError, match="amp_stat"):
            plot_lattice_micromotion(report, amp_stat="mean")
        with pytest.raises(ValueError, match="不能相同"):
            plot_lattice_micromotion(report, rf_axis="z", axial_axis="z")

    def test_show_theory_lines(self, tmp_path):
        """show_theory=True 叠加理论虚线竖线，总长 = |q|·|x_last − x_null|（x_last=末端瞬时）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        fig = plot_lattice_micromotion(report, cross=self._mock_cross(0.3),
                                       show_theory=True)
        ax = fig.axes[0]
        assert len(self._vlines_by_style(ax, "-")) == 2    # 数值实线
        dashed = self._vlines_by_style(ax, "--")           # 理论虚线（RF null 水平被排除）
        assert len(dashed) == 2
        dashed.sort(key=lambda v: v[0])
        x_last = report.trajectory.r_um[-1, :, 0]
        for (zc, ylo, yhi), x_end in zip(dashed, x_last):
            assert abs(yhi - ylo) == pytest.approx(0.3 * abs(x_end), rel=0.01)
        plt.close(fig)

    def test_show_theory_z_offset_separates_lines(self, tmp_path):
        """理论虚线沿 z 错开与实测实线并排（不再被覆盖）；theory_z_offset=0 则重合。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        # 默认偏移：理论与实测 z 不重合
        fig = plot_lattice_micromotion(report, cross=self._mock_cross(0.3),
                                       show_theory=True)
        ax = fig.axes[0]
        solid_z = {round(zc, 6) for zc, _, _ in self._vlines_by_style(ax, "-")}
        dashed_z = {round(zc, 6) for zc, _, _ in self._vlines_by_style(ax, "--")}
        assert solid_z and dashed_z
        assert solid_z.isdisjoint(dashed_z), "理论线应 z 错开，不与实测线同位"
        plt.close(fig)
        # offset=0：理论与实测 z 重合
        fig0 = plot_lattice_micromotion(report, cross=self._mock_cross(0.3),
                                        show_theory=True, theory_z_offset=0.0)
        ax0 = fig0.axes[0]
        solid_z0 = {round(zc, 6) for zc, _, _ in self._vlines_by_style(ax0, "-")}
        dashed_z0 = {round(zc, 6) for zc, _, _ in self._vlines_by_style(ax0, "--")}
        assert solid_z0 == dashed_z0, "offset=0 时理论与实测应同 z"
        plt.close(fig0)

    def test_show_theory_default_off(self, tmp_path):
        """默认 show_theory=False：仅数值实线竖线，无理论虚线竖线。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        fig = plot_lattice_micromotion(report, cross=self._mock_cross(0.3))
        ax = fig.axes[0]
        assert len(self._vlines_by_style(ax, "-")) == 2
        assert len(self._vlines_by_style(ax, "--")) == 0
        plt.close(fig)

    def test_show_theory_without_cross_warns(self, tmp_path, caplog):
        """show_theory=True 但 cross=None → warning，数值线照画，无理论线。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        with caplog.at_level("WARNING", logger="motion_analysis.plots"):
            fig = plot_lattice_micromotion(report, cross=None, show_theory=True)
        ax = fig.axes[0]
        assert len(self._vlines_by_style(ax, "-")) == 2
        assert len(self._vlines_by_style(ax, "--")) == 0
        assert any("cross" in r.message for r in caplog.records)
        plt.close(fig)

    def test_rf_null_line_uses_rf_null_not_center(self, tmp_path):
        """RF null 灰线用 cross.rf_null_um（赝势最小），而非含 DC 偏置的 center_um。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from motion_analysis.micromotion import CrossCheck
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        cross = CrossCheck(
            q_theory={"x": 0.3, "y": 0.0, "z": 0.0},
            q_measured_median={"x": 0.3, "y": 0.0, "z": 0.0},
            ratio={"x": 1.0, "y": 0.0, "z": 0.0},
            center_um=(-40.0, 0.0, 0.0),   # DC 偏移后的平衡点（旧"RF null"误标处）
            is_stable=True, freq_rf_MHz=F_RF_MHZ, species="Ba135+",
            rf_null_um=(0.0, 0.0, 0.0),   # 真 RF null
        )
        fig = plot_lattice_micromotion(report, cross=cross)
        # RF null 灰线（axhline，水平虚线）y 应=0（rf_null），而非 -40（center）。
        # axhline 是 y 恒定的水平线；按"y 恒定 + 虚线"筛选其 y 值。
        ax = fig.axes[0]
        hys = []
        for ln in ax.get_lines():
            yd = np.asarray(ln.get_ydata(), dtype=float)
            if yd.size >= 2 and np.std(yd) < 1e-9 and ln.get_linestyle() == "--":
                hys.append(float(yd[0]))
        assert any(abs(y) < 1e-9 for y in hys), f"RF null 线应=0，得到 {hys}"
        assert not any(abs(y + 40) < 1e-9 for y in hys), "不应使用 center_um=-40"
        plt.close(fig)

    def test_equal_aspect_default_on(self, tmp_path):
        """默认 equal_aspect=True：aspect==1.0（等比），figsize 比例 ≈ 数据跨度比。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        fig = plot_lattice_micromotion(report, cross=self._mock_cross(0.3),
                                       show_theory=True)
        ax = fig.axes[0]
        assert ax.get_aspect() == 1.0          # 等比
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xspan, yspan = (xl[1] - xl[0]), (yl[1] - yl[0])
        w, h = fig.get_size_inches()
        if xspan >= yspan:
            assert w >= h                       # 横向更宽的晶格画框更宽
        # 画框比例与数据比例方向一致（figsize 跟随数据）
        assert (w >= h) == (xspan >= yspan)
        plt.close(fig)

    def test_no_equal_aspect(self, tmp_path):
        """equal_aspect=False：aspect=='auto'，figsize 为默认 9×5。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        fig = plot_lattice_micromotion(report, cross=self._mock_cross(0.3),
                                       equal_aspect=False)
        ax = fig.axes[0]
        assert ax.get_aspect() == "auto"
        w, h = fig.get_size_inches()
        assert (w, h) == (9.0, 5.0)
        plt.close(fig)

    def test_axis_ranges_clamp_lims(self, tmp_path):
        """axis_ranges 按物理轴指定 → xlim/ylim 被钳到该范围（含未涉及的第三轴被忽略）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        # 默认 zox 平面：横轴 z、纵轴 x。z-range→xlim，x-range→ylim；y-range 不相关被忽略。
        fig = plot_lattice_micromotion(
            report, cross=self._mock_cross(0.3),
            axis_ranges={"z": (-5.0, 50.0), "x": (-10.0, 10.0), "y": (0.0, 1.0)},
        )
        ax = fig.axes[0]
        assert ax.get_xlim() == (-5.0, 50.0)
        assert ax.get_ylim() == (-10.0, 10.0)
        plt.close(fig)

    def test_scatter_span_collected_dense_lattice(self):
        """密集晶格（散点跨度 ≫ 仅有的几条竖线）xlim 应跟随散点，非塌缩到竖线 z。

        回归 ``ax.relim()`` 不收集 scatter (PathCollection) 的坑：散点很多但只有
        少数离子有竖线时，xlim 必须覆盖全部散点的跨度。直接测 _apply_lattice_aspect。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from motion_analysis.plots import _apply_lattice_aspect
        fig, ax = plt.subplots()
        # 散点 z 散布在 [-100,100]（模拟密集晶格），ax.relim 不收集 scatter
        z_scatter = np.linspace(-100, 100, 200)
        x_scatter = np.linspace(-3, 3, 200)
        ax.scatter(z_scatter, x_scatter, s=5)
        ax.plot([0, 0], [-1, 1], "r-")          # 仅一条竖线（如只有 1 离子有 results）
        _apply_lattice_aspect(ax, fig, "z", "x", None, True, z_scatter, x_scatter)
        xl = ax.get_xlim()
        # xlim 必须覆盖散点 z 跨度（~200），不能塌缩到竖线 z=0 附近
        assert (xl[1] - xl[0]) > 150, f"xlim 跨度 {xl} 未覆盖密集散点（漏 scatter）"
        assert ax.get_aspect() == 1.0            # 等比生效
        plt.close(fig)


# ============== CLI 参数解析 ==============

class TestCLI:
    def test_show_flag(self):
        from motion_analysis.__main__ import create_parser
        parser = create_parser()
        a = parser.parse_args(["r", "--csv", "c.csv", "--config", "g.json", "--show"])
        assert a.show is True
        b = parser.parse_args(["r", "--csv", "c.csv", "--config", "g.json"])
        assert b.show is False
        assert b.plot_dir is None

    def test_lattice_show_theory_flag(self):
        from motion_analysis.__main__ import create_parser
        parser = create_parser()
        a = parser.parse_args(["r", "--csv", "c.csv", "--config", "g.json",
                               "--lattice-show-theory"])
        assert a.lattice_show_theory is True

    def test_axis_range_flags(self):
        """--x-range/--y-range/--z-range 解析为 (lo,hi) tuple；--no-equal-aspect 关等比。"""
        from motion_analysis.__main__ import create_parser, _collect_axis_ranges
        parser = create_parser()
        a = parser.parse_args([
            "r", "--csv", "c.csv", "--config", "g.json",
            "--x-range", "-5", "5", "--y-range", "-3", "3", "--z-range", "0", "60",
            "--no-equal-aspect",
        ])
        assert a.x_range == [-5.0, 5.0]
        assert a.y_range == [-3.0, 3.0]
        assert a.z_range == [0.0, 60.0]
        assert a.equal_aspect is False
        assert _collect_axis_ranges(a) == {"x": (-5.0, 5.0), "y": (-3.0, 3.0),
                                           "z": (0.0, 60.0)}
        # 默认：无 range，equal_aspect=True
        b = parser.parse_args(["r", "--csv", "c.csv", "--config", "g.json"])
        assert b.equal_aspect is True
        assert _collect_axis_ranges(b) is None

    def test_axis_range_rejects_bad(self):
        """非法 range（hi≤lo）报 ArgumentTypeError；非数值/缺数由 argparse type/nargs 拦。"""
        import argparse
        import pytest
        from motion_analysis.__main__ import _collect_axis_ranges, create_parser
        parser = create_parser()
        # nargs=2 负责拦截个数，type=float 拦截非数值
        a = parser.parse_args(["r", "--csv", "c.csv", "--config", "g.json",
                               "--x-range", "5", "-5"])   # hi<lo
        with pytest.raises(argparse.ArgumentTypeError):
            _collect_axis_ranges(a)
        a2 = parser.parse_args(["r", "--csv", "c.csv", "--config", "g.json",
                                "--x-range", "5", "5"])    # hi==lo
        with pytest.raises(argparse.ArgumentTypeError):
            _collect_axis_ranges(a2)
        with pytest.raises(SystemExit):
            parser.parse_args(["r", "--csv", "c.csv", "--config", "g.json",
                               "--x-range", "a", "b"])     # 非数值→argparse 报错退出
