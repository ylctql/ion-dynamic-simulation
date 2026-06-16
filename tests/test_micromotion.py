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
        """R0 更大（更偏离 RF null）的离子 micromotion 竖线更长（excess micromotion）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = self._build_report(
            tmp_path, q_list=[0.3, 0.3], R0_list=[20.0, 60.0], z_list=[0.0, 20.0],
        )
        fig = plot_lattice_micromotion(report)
        vlines = sorted(self._vertical_lines(fig.axes[0]))  # 按 z 升序
        lengths = [abs(v[2] - v[1]) for v in vlines]
        # z=20（R0=60）离子竖线总长 ≈ q·R0=18，远大于 z=0（R0=20）的 ≈6
        assert lengths[1] > lengths[0] * 2
        # 竖线居中于平衡位置 x_eq ≈ R0
        for (zc, ylo, yhi), R0 in zip(vlines, [20.0, 60.0]):
            assert 0.5 * (ylo + yhi) == pytest.approx(R0, abs=1.0)
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
