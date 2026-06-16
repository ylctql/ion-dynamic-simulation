# motion_analysis — Micromotion 幅度数值测量

从 `continuous_sampling/` 产出的 `frame*.npz` 离子轨迹，**逐离子、逐轴**地数值测量 RF micromotion 的瞬时幅度 $\beta(t)$ 与有效 Mathieu 调制深度 $q_\text{eff}$，并与 [`trap_stability`](trap_stability.md) 从场几何算出的理论 $q$ 交叉验证。

与 `trap_stability`（从**势场几何**解析地算 $a,q$）互补，本模块从**实际动力学轨迹**数值地"测量" micromotion——两者对照可诊断 excess micromotion（离子偏离 RF 零场点导致的额外微运动）。

---

## 物理背景

### 乘性调制模型（为什么不是加性）

Paul 阱中，绝热近似下离子某一轴的运动是 RF 驱动对 secular（慢变）运动的**乘性调制**：

$$x(t) \approx X_\text{sec}(t)\cdot\Big[\,1 + \tfrac{q}{2}\cos(\Omega_\text{RF} t)\,\Big]$$

其中 $X_\text{sec}(t)$ 为 secular 位移（在阱中缓慢振荡，频率 $\omega_\text{sec} \ll \Omega_\text{RF}$），$q$ 为该轴的 Mathieu 调制深度，$\Omega_\text{RF}$ 为 RF 驱动角频率。

**关键推论**：micromotion 的瞬时幅度正比于 secular 位移本身：

$$\beta(t) \;=\; \tfrac{|q|}{2}\,\big|X_\text{sec}(t)\big|$$

- 离子在阱中心（$X_\text{sec}=0$）时 micromotion 为零；
- 离子偏离 RF 零场点越远，micromotion 越大；
- 频域上，micromotion 表现为 $\Omega_\text{RF} \pm \omega_\text{sec}$ 的边带，而非单一 $\Omega_\text{RF}$ 峰。

> ⚠️ **常见错误**：把轨迹拟合为恒幅加性模型 $x(t) = x_0 + a_\text{sec}\cos\omega_\text{sec}t + a_\text{micro}\cos\Omega_\text{RF}t$。该模型假设 micromotion 幅度 $a_\text{micro}$ 与 secular 位移无关，物理上错误；且 $a_\text{micro}$ 与 $\omega_\text{sec}$ 在数据窗不够长时强耦合，拟合不稳。本模块严格采用乘性模型。

### 两条互补的测量路径

| 路径 | 输出 | 物理量 | 用途 |
|------|------|--------|------|
| **Phase-folding**（主方法） | 时变 $\beta(t)$ (µm) | micromotion 一阶谐波幅度 | 诊断哪些时刻/位置 micromotion 显著 |
| **方法 C 乘性回归** | 全局标量 $q_\text{eff}$ | 有效调制深度 | 直接对标理论 $q$，诊断 excess micromotion |

两者算法独立，互为交叉验证：phase-folding 给时变诊断，方法 C 给全局标量。

### Excess micromotion

线性阱近似下，冷单粒子位于阱中心时 $q_\text{eff} \approx q_\text{theory}$（来自 `trap_stability`）。但在多离子晶格中，库仑排斥使离子偏离 RF 零场点，导致 $q_\text{eff} > q_\text{theory}$——即 excess micromotion。比值

$$\text{ratio} = \frac{\text{median}_\text{ions}(q_\text{eff})}{q_\text{theory}}$$

冷单粒子 ≈ 1，晶格 excess > 1。这是本模块的核心物理输出之一。

---

## 测量方法

### 方法 1：Phase-folding（时变 $\beta(t)$）

实验上测量 micromotion 的标准稳健做法，天然处理幅度调制：

1. 对位置时间序列计算 RF 相位 $\varphi_i = (\Omega_\text{RF} t_i) \bmod 2\pi$；
2. 在时间轴上滑窗（窗长 $W$，满足 $T_\text{RF} \ll W \ll T_\text{secular}$），窗之间 50% 重叠；
3. 窗内按 $\varphi$ 分 $n_\text{bins}$ 个 bin，对每个 bin 取位置均值；
4. 对 bin 均值拟合一阶谐波：

$$\langle r\rangle(\varphi) = \bar r + \beta_c\cos\varphi + \beta_s\sin\varphi$$

5. $\beta = \sqrt{\beta_c^2 + \beta_s^2}$ 为该窗 micromotion **一阶幅度**（peak amplitude），$2\beta$ 为 peak-to-peak，$\bar r$ 为该窗 secular 中心。

输出随时间变化的 $\beta(t)$，每个窗给一个采样点。窗长默认 $W = \max(3\,T_\text{RF},\ 0.3\,\mu\text{s})$。

### 方法 2：方法 C —— 乘性模型最小二乘回归 $q_\text{eff}$

对（裁掉瞬态后的）整段轨迹直接回归调制深度。先用 FFT 理想低通从 $r(t)$ 提取 secular 包络 $X_\text{sec}(t)$（截止频率 $f_\text{RF}/2$，干净分离 RF 基波与 secular 运动），再对残差做乘性回归：

$$\delta r(t) \;=\; r(t) - X_\text{sec}(t) \;=\; a\,X_\text{sec}(t)\cos(\Omega_\text{RF} t) \;+\; b\,X_\text{sec}(t)\sin(\Omega_\text{RF} t)$$

最小二乘解 $(a, b)$，则

$$q_\text{eff} = 2\sqrt{a^2 + b^2}$$

> **为什么用 FFT 理想低通而非 Savitzky-Golay**：savgol 类低通对 RF 基波的传递函数有泄漏，导致 $q_\text{eff}$ 系统性偏差（实测 savgol 版本在 $q=0.1$ 时误差 1.4–3.2%）。FFT 理想低通（锐截止于 $f_\text{RF}/2$）干净去除 RF 基波，合成数据回收误差 < 0.001%。回归时裁掉 $n/20$ 边界以缓解 FFT 周期延拓引入的 Gibbs 振荡。

`q_eff` 的 stderr 由残差方差经误差传播 $q_\text{eff}=2\sqrt{a^2+b^2}$ 给出。

---

## 瞬态裁剪（warmup）

### 为什么需要裁瞬态

离子从初始条件弛豫到平衡（热化/冷却）的瞬态段会污染两条路径：

- **Phase-folding**：瞬态期 secular 时间尺度变短，窗条件 $W \ll T_\text{secular}$ 失效，窗内 bin 均值混叠不同 secular 位置，$\beta(t)$ 失真；
- **方法 C**：瞬态大振幅运动违反绝热近似（$X_\text{sec}$ 在一个 RF 周期内显著变化），且非谐-振幅耦合使 secular 频率偏移导致低通提取不干净，加之回归基 $X_\text{sec}\cos$ 被 $X_\text{sec}$ 加权使瞬态高位移点高杠杆——三者共同偏置 $q_\text{eff}$。

### 自动检测（默认开启）

`analyze_run` 默认逐 (ion, axis) 调用 `detect_warmup()`，自动检测瞬态收敛点 $t^*$ 并裁剪后再分析。检测算法：

1. **低通 + 边缘裁剪**：FFT 低通（截止 $f_\text{RF}/2$）提 secular 包络 $X_\text{sec}$，对称裁掉 $n/20$ 边界（复用 Gibbs 约定）；
2. **去趋势 + secular 频率估计**：对 $X_\text{sec}$ 减 2 阶多项式趋势，rfft 残差在频带 $[2\Delta f,\ \min(0.45 f_\text{RF},\ 2.5 f_\text{prior})]$ 内取峰；峰不显著（峰/中位 < 3）则回退先验 $f_\text{prior}$；
3. **滑窗稳态判定**：窗长 $W = n_\text{periods}/f_\text{sec}$，50% 重叠；每窗计算中位值与 P95–P05 幅度，与末段 25% 参考比较；
4. **混合容差**（防冷离子过裁剪）：窗"收敛"判据为 $|w - r_\text{ref}| \le \text{tol}_\text{rel}\cdot s_\text{ref} + 4\sigma_\text{noise}\sqrt{W_n/n}$，其中 $\sigma_\text{noise}$ 来自 micromotion 残差的 MAD；
5. **鲁棒收敛点**：对窗收敛标记做 3 窗中值滤波（杀散落噪声窗），再找长度 $\ge 3$ 的连续未收敛段；$t^*$ = 末个未收敛段之后首个收敛窗的起点。

### 三条 no-op 守卫

以下情形检测器返回 $t^* = t[0]$、**不裁剪**（整段原样进入分析），保证默认开启不破坏干净数据：

| reason | 触发条件 | 含义 |
|--------|----------|------|
| `too-short` | 样本不足以填满 ≥4 个检测窗 | 轨迹过短，无法可靠检测 |
| `static` | 末段 secular 包络近常数（如常位移轴） | 该轴无 secular 运动，无瞬态可言 |
| `unconverged-tail` | 末段 25% 自身未稳（仍在漂移/振荡增长） | 无法定义收敛参考，保守不裁 |

干净稳态信号从头收敛 → reason=`auto`、dropped=0（no-op）；存在真实瞬态 → reason=`auto`、dropped>0。

### 手动 override

| 优先级 | CLI flag | 语义 | reason |
|--------|----------|------|--------|
| 高 | `--trim-start-us T` | 绝对时刻 $t^*=T$ (µs) | `manual-abs` |
| 中 | `--warmup-us T` | 相对 $t^*=t[0]+T$ (µs) | `manual-rel` |
| 低 | `--no-auto-trim` | 关闭自动检测（不裁） | `none` |
| — | （无 warmup 参数） | 自动检测 | `auto`/守卫 |

裁后剩余帧不足以产出 ≥1 个 phase-folding 窗时，自动回退（钳制），reason 追加 `+clamped`。

---

## 交叉验证

`cross_check_q()` 调用 `trap_stability.compute_stability_from_field()` 从场几何算理论 $q$，与各离子 $q_\text{eff}$ 的中位数对比，输出每轴 ratio。注：若启用了 warmup 裁剪，各离子 $q_\text{eff}$ 可能来自**不同的稳态时间窗**（per-ion 各自裁瞬态），但中位数仍定义良好——这正是 excess micromotion 诊断所需（每个离子取其各自收敛后的 $q_\text{eff}$）。

---

## 快速使用

```bash
# 1. 采集连续采样轨迹（每 RF 周期 ≥ 8 点，总时长 ≥ 3 个 secular 周期）
python main.py --N 3 --time 20 \
    --continuous-sampling --continuous-sampling-frames 20000 \
    --interval 0.08 --step 10

# 2. 分析（默认自动裁瞬态）
python -m motion_analysis continuous_sampling/t030.00_interval0.08_step10 \
    --csv monolithic20241118.csv --config default.json \
    --out mm.json --plot-dir plots

# 3. 关闭自动裁剪对比（或用 --warmup-us 已知热化时间手动裁）
python -m motion_analysis <run_dir> --csv <csv> --config <json> --no-auto-trim --out mm_raw.json
```

> ⚠️ `--config` 必须是**采集时所用的同一份**配置——RF 频率需精确匹配，否则 $q_\text{eff}$ 失真（`resid_rms` 诊断会警告）。

---

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `run_dir` | （必填） | `continuous_sampling` 输出目录（含 `frame*.npz`） |
| `--csv` | （必填） | 电场 CSV；仅文件名时在 `data/` 下查找 |
| `--config` | （必填） | 电压 JSON（解析 RF 频率）；仅文件名时在 `configs/` 下查找 |
| `--species` | `Ba135+` | 离子种类（决定 config 无量纲化的质量） |
| `--axes` | `x,y,z` | 分析轴，逗号分隔 |
| `--ions` | 全部 | 指定离子索引（逗号分隔） |
| `--window-us` | `max(3·T_RF, 0.3)` | phase-folding 滑窗长度 (µs) |
| `--n-phase-bins` | `32` | 相位 bin 数 |
| `--secular-freq` | `0.5` | secular 频率估计 (MHz)，采样校验用 |
| `--trim-start-us` | （自动） | 手动指定稳态起点（绝对 µs）；优先级最高 |
| `--warmup-us` | （自动） | 手动丢弃前 N µs（相对 t[0]） |
| `--no-auto-trim` | `False` | 关闭自动瞬态检测 |
| `--warmup-tol` | `0.1` | 自动检测相对收敛容差 |
| `--warmup-periods` | `3` | 自动检测窗覆盖的 secular 周期数 |
| `--center` | 自动检测 | 阱中心 `X,Y,Z` (µm) |
| `--smooth-axes` | `z` | 交叉验证中场平滑方向；`none` 关闭 |
| `--smooth-sg` | `11,3` | Savitzky-Golay 窗口,阶数 |
| `--no-cross-check` | `False` | 跳过 trap_stability 交叉验证 |
| `--lattice-show-theory` | `False` | 晶格 micromotion 图叠加理论 β 竖线比对（绿色虚线，需 cross-check 未关） |
| `--out` | （可选） | JSON 输出路径 |
| `--plot-dir` | （可选） | 图输出目录 |

---

## 采样要求

`load_continuous_sampling()` 会强制校验，不满足时抛错并给出参数建议：

| 校验项 | 阈值 | 失败行为 |
|--------|------|----------|
| 每 RF 周期采样点数 | ≥ 8（方稳健分辨一阶谐波） | `raise`，建议减小 `--interval` 或增大 `--step` |
| 总时长 | ≥ max(3·T_secular, 5 µs) | `raise`，建议增加 `--continuous-sampling-frames` |

默认 `--interval 0.08 --step 10` 满足要求（实测帧间隔 ~0.7 ns，35 MHz RF 周期 ~28.6 ns → 每周期 ~39 点）。

---

## 输出

### 终端摘要

```
============================================================
  Micromotion Analysis
============================================================
  Run:        continuous_sampling/t030.00_interval0.08_step10
  RF:         35.280 MHz (dt=0.70 ns)
  Ions:       3  axes: x,y,z
  window:     0.3000 us, 32 phase bins
  axis x: q_eff median=0.12320 (min=0.12100, max=0.12540)
  axis y: q_eff median=0.12315 (min=0.12090, max=0.12510)
  axis z: q_eff median=0.00012 (min=0.00008, max=0.00018)
  warmup: 6/9 (ion,axis) trimmed, global max t*=2.841 us
  Cross-check (trap_stability q_theory):
    x: q_theory=0.12346, q_meas_median=0.12320, ratio=0.998
    y: q_theory=0.12346, q_meas_median=0.12315, ratio=0.998
    z: q_theory=0.00000, q_meas_median=0.00012, ratio=nan
    center=(0.00,0.00,0.00) um  stable=True
============================================================
```

### JSON 结构（`--out`）

```json
{
  "run_dir": "continuous_sampling/t030.00_interval0.08_step10",
  "freq_rf_MHz": 35.28,
  "Omega_rf_rad_per_us": 221.56,
  "dt_ns": 0.70,
  "n_ions": 3,
  "axes": ["x", "y", "z"],
  "ions": [0, 1, 2],
  "window_us": 0.3,
  "n_phase_bins": 32,
  "q_eff": [[0.1234, 0.1233, 0.0001], [...], [...]],
  "q_eff_stderr": [[0.0012, 0.0011, 0.0002], [...], [...]],
  "per_ion": {
    "0": {
      "x": {
        "q_eff": 0.1234,
        "q_eff_stderr": 0.0012,
        "beta_median_um": 3.05,
        "beta_max_um": 6.12,
        "residual_rms_um": 0.08,
        "t_star_us": 2.841,
        "dropped_frames": 4058,
        "warmup_reason": "auto"
      },
      "y": { "...": "..." },
      "z": {
        "q_eff": 0.0001, "q_eff_stderr": 0.0002,
        "beta_median_um": 0.01, "beta_max_um": 0.02,
        "residual_rms_um": 0.05,
        "t_star_us": 0.0, "dropped_frames": 0,
        "warmup_reason": "static"
      }
    }
  },
  "cross_check": {
    "q_theory": {"x": 0.12346, "y": 0.12346, "z": 0.0},
    "q_measured_median": {"x": 0.12320, "y": 0.12315, "z": 0.00012},
    "ratio_measured_over_theory": {"x": 0.998, "y": 0.998, "z": null},
    "center_um": [0.0, 0.0, 0.0],
    "is_stable": true,
    "species": "Ba135+"
  }
}
```

字段说明：
- `q_eff` / `q_eff_stderr`：$(N, 3)$ 数组，每离子每轴的有效调制深度及 stderr；
- `per_ion[ion][axis]`：含 $q_\text{eff}$、$\beta$ 中位/最大值、方法 C 残差 RMS、以及 warmup 三字段（`t_star_us` 裁剪时刻、`dropped_frames` 丢弃帧数、`warmup_reason` 原因）；
- `residual_rms_um`：方法 C 残差 RMS。若 `median(resid_rms)/median(β) > 0.3`，CLI 会警告 RF 频率可能不匹配（config 非采集所用）。

### 图（`--plot-dir`）

| 文件 | 内容 |
|------|------|
| `qeff_histogram.png` | 各轴 $q_\text{eff}$ 分布直方图（RF 径向轴 vs 轴向对比） |
| `qeff_vs_displacement.png` | $q_\text{eff}$ vs 离子平衡位置偏离阱中心的距离（$r_\text{eq}$ 用收敛段均值，标注 $q_\text{theory}$ 水平线） |
| `beta_vs_secular.png` | 所有窗 $\beta(t)$ vs $|X_\text{sec}-\text{center}|$ 散点 + 理论斜率 $q_\text{theory}/2$ |
| `lattice_micromotion_x.png` | zox 平面晶格平衡位置散点 + 每离子 $x$ 方向 micromotion 竖线（**数值 phase-folding 实测**，红色实线，居中、总长 $2\beta$=ptp），偏离 RF 零场的离子竖线变长 → excess micromotion 直接成像；提供 `cross` 时叠加 RF null 水平虚线；`--lattice-show-theory` 时另叠加**理论** $\beta_\text{theory}=|q_\text{theory}|/2\cdot|x_\text{eq}-x_\text{null}|$ 绿色虚线竖线（仅比对，excess 时数值线长于理论线） |

`plot_ion_timeseries`（notebook 用，非 CLI）按 `dropped_frames` 自动裁掉 warmup 段绘图，并在发生裁剪时画 $t^*$ 竖虚线。

---

## 算法流程

```
continuous_sampling/frame*.npz
  │
  ├─ load_continuous_sampling()
  │     ├─ RF 频率从 config（init_from_config → cfg.freq_RF）
  │     ├─ 逐帧堆叠 r/v/t_us（物理单位 µm, m/s, µs）
  │     └─ 采样质量校验（≥8 点/RF周期，总时长 ≥3 secular 周期）
  │
  └─ analyze_run()  逐 (ion, axis):
        ├─ [warmup] detect_warmup(r, t, Ω_RF)
        │     ├─ FFT 低通(cutoff f_RF/2) → X_sec，边缘裁 n/20
        │     ├─ 去趋势 + rfft → secular 频率（不显著回退先验）
        │     ├─ 滑窗稳态（中值 + P95-P05，混合 rel/abs 容差）
        │     ├─ 3 窗中值滤波 + ≥3 连续 bad 段 → t*
        │     └─ no-op 守卫: too-short / static / unconverged-tail → t*=t[0]
        ├─ 解析 i_start（trim_start_us > warmup_us > no_auto_trim > auto）
        ├─ 钳制剩余帧 ≥ max(16, ⌈2·W/dt⌉)
        │
        └─ compute_micromotion(r[i_start:], t[i_start:], Ω_RF)
              ├─ 方法 C：X_sec = FFT低通(r, cutoff f_RF/2)
              │     δr = a·X_sec·cos + b·X_sec·sin  (裁 n/20 边界)
              │     q_eff = 2√(a²+b²),  stderr 由残差方差传播
              └─ Phase-folding：滑窗(W, 50%重叠) → 按 φ 分 bin
                    ⟨r⟩(φ)=r̄+β_c·cosφ+β_s·sinφ → β(t)=√(β_c²+β_s²)

  └─ cross_check_q()  [可选]
        └─ trap_stability.compute_stability_from_field() → q_theory
              ratio = median(q_eff) / q_theory  per axis
```

---

## 模块结构

| 文件 | 内容 |
|------|------|
| `micromotion.py` | 核心库：`load_continuous_sampling()`、`detect_warmup()`、`compute_micromotion()`、`analyze_run()`、`cross_check_q()`、`report_to_dict()`；数据类 `TrajectoryData`/`WindowMM`/`IonAxisResult`/`MicromotionReport`/`CrossCheck`/`WarmupInfo` |
| `plots.py` | `plot_ion_timeseries`/`plot_qeff_histogram`/`plot_qeff_vs_displacement`/`plot_beta_vs_secular`/`plot_lattice_micromotion`（延迟 import matplotlib） |
| `__main__.py` | CLI 入口；`python -m motion_analysis` |
| `micromotion_analysis.ipynb` | 调用库的展示层 notebook |

---

## 单位与关键约定

- **物理单位全程使用** µm、µs、m/s（仅交叉验证时经 `init_from_config` 取 `cfg`）；`Ω_RF = 2π·freq_RF` (rad/µs，因 1 MHz = 1 cycle/µs)。
- **RF 径向轴识别**：对所有轴都算，由 $q_\text{eff}$ 显著非零者判定为 RF 驱动轴（轴向通常 $q_\text{eff}\approx 0$）。
- **`dropped_frames` 为整数帧索引**（非 $t^*/dt$），非均匀帧无歧义。
- **`reason` 为闭集**：`{none, static, unconverged-tail, too-short, auto, manual-abs, manual-rel}`，钳制后追加 `+clamped`。

---

## 测试

```bash
pytest tests/test_micromotion.py -v
```

覆盖（28 项）：
- **TestRecoverQ**：合成 $r(t)=X_\text{sec}(t)\cdot[1+(q/2)\cos\Omega t]$ 回收已知 $q$（常位移/secular 调制，$q\in\{0.1,0.3,0.6\}$，rel<1–2%）；$q=0$ 退化、负 $q$、$\beta(t)$ 跟踪 secular、字段完整性。
- **TestLoad**：目录/帧缺失、采样率/时长不足异常、合规加载。
- **TestAnalyzeRun**：多离子批量 $q_\text{eff}$、report_to_dict 序列化。
- **TestWarmup**：瞬态高杠杆裁剪回收 $q$（裁后优于不裁）、干净稳态/常位移/增长尾的 no-op、secular 频率估计与回退、散落噪声鲁棒、钳制不崩、直调默认值、JSON warmup 字段。
- **TestLatticePlot**：`plot_lattice_micromotion` 运行不崩、竖线数=离子数、竖线居中且总长随偏离 RF null 增大（excess micromotion）、`amp_stat='max'`/缺轴/非法参数路径。

---

## 依赖关系

- `FieldConfiguration/` — `init_from_config`（Config, RF 频率）、`ion_species`、`build_voltage_list`
- `FieldParser/` — CSV 读取、势场/电场插值
- `field_visualize/core.py` — `apply_savgol_smooth`、`compute_potentials`
- `trap_stability/stability.py` — `compute_stability_from_field`、`find_trap_center`（交叉验证）
- `numpy.fft` / `numpy.linalg` — FFT 低通、最小二乘回归
- 规划文档：`docs/plan/micromotion_analysis.md`（实现规划历史）
