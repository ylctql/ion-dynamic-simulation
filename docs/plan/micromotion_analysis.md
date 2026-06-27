# Micromotion 数值测量 — 实现规划

> 在动力学模拟产物（`continuous_sampling/` 的 `frame*.npz`）基础上，数值测量每个离子的 RF micromotion 幅度与有效 Mathieu 调制深度 `q_eff`，并与 `trap_stability` 从场几何算出的理论 `q` 交叉验证。

## 一、物理基础

### 1.1 为什么不能直接用恒幅加性拟合

既有 `micromotion_analysis.ipynb` 把离子位置拟合为

```
y(t) = y0 + a_sec·cos(ω_sec t) + a_micro·cos(Ω_RF t)        ← 错误（加性）
```

Paul 阱中绝热近似的正确解是**乘性调制**：

```
x(t) ≈ X_sec(t) · [ 1 + (q/2)·cos(Ω_RF t) ]
```

- micromotion 幅度 ∝ 瞬时 secular 位移 `X_sec(t)`，随时间变化；
- 频域表现为 `Ω_RF ± ω_sec` 边带，而非单一 `Ω_RF` 峰；
- 加性模型中 `a_micro` 与 `ω_sec` 强耦合，数据窗不够长时拟合不稳，且 `a_micro` 物理意义错误。

### 1.2 主方法：RF 相位折叠（phase-folding）

实验上测 micromotion 的标准稳健做法：

1. 对位置时间序列计算 RF 相位 `φ_i = (Ω_RF·t_i) mod 2π`；
2. 把时间轴滑窗（窗长 `W`，满足 `T_RF << W << T_secular`），每窗内按 `φ` 分 bin；
3. 对 bin 均值拟合一阶谐波：`⟨r⟩(φ) = r̄ + β_c·cos φ + β_s·sin φ`；
4. `β = √(β_c² + β_s²)` 为该窗 micromotion 一阶幅度（peak amplitude），`2β` 为 peak-to-peak，`r̄` 为该窗 secular 中心。

输出**随时间变化的 micromotion 幅度** `β(t)`，用于诊断哪些时刻/位置 micromotion 显著。

### 1.3 验证方法 C：乘性模型最小二乘拟合 q_eff

对整段轨迹直接回归调制深度：

```
δr(t) = r(t) − X_sec(t)  =  a·X_sec(t)·cos(Ωt) + b·X_sec(t)·sin(Ωt)
```

其中 `X_sec(t)` 由 Savitzky-Golay 低通包络提取。最小二乘解 `a, b`，则

```
q_eff = 2·√(a² + b²)
```

`q_eff` 直接对标 `trap_stability` 的理论 `q`：冷单粒子在阱中心时 `q_eff ≈ q_theory`；离子平衡位置偏离 RF 零场点（晶格中库仑排斥所致）时 `q_eff > q_theory`，即 excess micromotion —— 这是本功能的核心物理输出。

> phase-folding 给时变诊断 `β(t)`，方法 C 给全局标量 `q_eff`。两者独立算法，互为交叉验证。

## 二、模块结构

```
motion_analysis/
  __init__.py                      ← 导出（新增）
  micromotion.py                   ← 核心库（新增）
  __main__.py                      ← CLI 入口（新增）
  micromotion_analysis.ipynb       ← 重写为展示层（调用库）
  secularmotion_analysis.ipynb     ← 既有，不动
```

不改动 `ComputeKernel` / `main.py` 的采集逻辑 —— `--continuous-sampling` 已能产出满足 Nyquist 的数据（实测帧间隔 ~0.7 ns，RF 周期 ~28 ns，每周期 ~39 点）。

## 三、核心 API（`micromotion.py`）

### 3.1 数据加载与采样校验

```python
@dataclass
class TrajectoryData:
    t_us: np.ndarray          # (T,)
    r_um: np.ndarray          # (T, N, 3)
    v_um_per_s: np.ndarray    # (T, N, 3)
    Omega_rf: float           # rad/µs
    freq_rf_MHz: float
    dt_us: float              # 中位帧间隔
    run_dir: Path

def load_continuous_sampling(
    run_dir: str | Path,
    *, csv_path: str | Path, config_path: str | Path,
    species: str = "Ba135+",
) -> TrajectoryData
```

- RF 频率从 `config_path` 经 `init_from_config` 得到（`cfg.freq_RF`），`Ω_RF = 2π·freq_RF`（rad/µs，因 1 MHz = 1 cycle/µs）。
- `r_um` / `v_um_per_s` / `t_us` 直接来自 npz（`save_frame_rv_npz` 已转好物理单位）。

**采样质量校验（断言 + 提示，强制）：**


| 校验项            | 阈值                                                        | 失败行为                               |
| -------------- | --------------------------------------------------------- | ---------------------------------- |
| 帧间隔 `dt_us`    | ≤ `2 / (n_phase_bins·freq_RF)` ≈ 1.3 ns（保证每 RF 周期 ≥ 20 点） | `raise`，附 `--interval`/`--step` 建议 |
| 总时长 `T`        | ≥ `max(5·T_secular_est, 5 µs)`                            | `raise`                            |
| `T_secular` 估计 | 取 `freq_secular ≈ 0.5 MHz` 默认，可由 `--secular-freq` 覆盖      | warning                            |
| 每相位 bin 样本数    | `≥ 8`                                                     | warning（自动减窗）                      |


### 3.2 Phase-folding

```python
@dataclass
class WindowMM:
    t_center_us: float
    r_secular_um: float      # 窗内 secular 中心 r̄
    beta_um: float           # 一阶谐波幅度 |β|
    phase_rad: float         # atan2(β_s, β_c)
    ptp_um: float            # 2|β|

@dataclass
class IonAxisResult:
    windows: list[WindowMM]
    q_eff: float             # 方法 C 全局调制深度
    q_eff_stderr: float
    t_windows: np.ndarray    # 窗中心时刻
    beta_t: np.ndarray       # β(t)
    secular_envelope_um: np.ndarray   # 低通 X_sec(t)，供 β vs |X_sec| 图

def compute_micromotion(
    r_axis_um: np.ndarray,   # (T,) 单离子单轴位置
    t_us: np.ndarray,        # (T,)
    Omega_rf: float,
    *, window_us: float | None = None, n_phase_bins: int = 32,
    savgol_window: int | None = None, savgol_poly: int = 3,
) -> IonAxisResult
```

- 窗长默认 `max(3·T_RF, 0.3 µs)`，滑步 = `W/2`（50% 重叠）。
- 每窗按 `φ` 分 `n_phase_bins` 个 bin，bin 均值后线性最小二乘解 `(r̄, β_c, β_s)`（设计矩阵 `[1, cos φ, sin φ]`）。
- 方法 C 在整段上对 `δr = r − X_sec` 回归 `X_sec·cos(Ωt)`、`X_sec·sin(Ωt)`，得 `q_eff = 2√(a²+b²)`，stderr 由残差方差给出。

### 3.3 多离子批处理

```python
@dataclass
class MicromotionReport:
    trajectory: TrajectoryData
    results: dict[tuple[int, str], IonAxisResult]   # (ion_idx, axis)
    q_eff: np.ndarray            # (N, 3)
    q_eff_stderr: np.ndarray     # (N, 3)

def analyze_run(
    run_dir, *, csv_path, config_path, species="Ba135+",
    axes=("x","y","z"), ions=None, window_us=None, n_phase_bins=32,
) -> MicromotionReport
```

`ions=None` 表示全部 N 个离子逐个计算。

### 3.4 交叉验证

```python
@dataclass
class CrossCheck:
    q_theory: dict[str, float]   # {"x":..,"y":..,"z":..}
    q_measured_median: dict[str, float]   # 各离子 q_eff 中位数
    ratio: dict[str, float]      # q_measured_median / q_theory
    center_um: tuple[float, float, float]
    stable: bool

def cross_check_q(report, *, csv_path, config_path, species="Ba135+",
                  center_um=None) -> CrossCheck
```

调用 `trap_stability.compute_stability_from_field(...)`（复用其插值器构建逻辑），返回各轴理论 `q` 与测量中位数之比。冷单粒子 `ratio ≈ 1`；晶格 excess micromotion `ratio > 1`。

## 四、CLI（`__main__.py`）

```
python -m motion_analysis <run_dir> \
    --csv <csv> --config <json> [--species Ba135+] \
    [--axes x,y,z] [--ions 0,1,2] [--window-us 0.3] [--n-phase-bins 32] \
    [--secular-freq 0.5] [--out result.json] [--plot-dir plots/]
```

- `--csv --config` 必填（RF 频率 + 交叉验证均依赖 config）。
- 输出 JSON：per-ion `q_eff`（N×3）、window 序列、`q_theory` 对比。
- 输出图（`--plot-dir`）：每轴 `β vs |X_sec − r_null|` 散点 + 拟合斜率（理论 `q/2` 验证线）；多离子 `q_eff` 空间分布（按离子平衡位置着色）。

## 五、Notebook 重写

`micromotion_analysis.ipynb` 改为调用库的展示层：

1. 单离子时序：`r(t)` + secular 包络 `X_sec(t)` + micromotion 残差 `δr(t)`；
2. 多离子空间分布：各离子平衡位置处 `q_eff`，RF 径向轴 vs 轴向对比；
3. `q` 交叉验证三联图：`q_eff`（测量）vs `q_theory`（场几何）vs 散点斜率。

删除原有错误的加性 `curve_fit` 拟合。

## 六、验证锚点


| 锚点                                              | 预期                            | 实现          |
| ----------------------------------------------- | ----------------------------- | ----------- |
| `--trap-freq` 纯谐振势（无 RF micromotion）跑数据         | `q_eff ≈ 0`，`β ≈ 0`           | 退化检验        |
| 单离子冷态 vs `trap_stability` q                     | `ratio(q_eff / q_theory) ≈ 1` | cross_check |
| Phase-folding `β(t)` 峰值 vs 方法 C `q_eff·X_sec/2` | 一致                            | 内部互验        |
| 合成数据 `X·[1+(q/2)cos Ωt]` 回收已知 q                 | `q_eff` 误差 < 1%               | 单元测试        |
| `β vs                                           | X_sec − r_null                | ` 散点        |


## 七、实施顺序

1. `micromotion.py`：`load_continuous_sampling` + 采样校验 + 合成数据自测
2. `compute_micromotion`（phase-folding）+ `fit_modulation_depth`（方法 C）
3. `analyze_run` 多离子批处理 + `cross_check_q`
4. `__main__.py` CLI + JSON/图输出
5. 重写 `micromotion_analysis.ipynb`
6. `tests/test_micromotion.py`：合成数据回收 q、`--trap-freq` 退化、加载/校验异常
7. 更新 `CLAUDE.md`，运行 pytest

## 八、关键约定

- **单位**：分析全程使用物理单位（µm、µs、m/s），仅交叉验证时经 `init_from_config` 取 `cfg`。
- **RF 径向轴识别**：对所有轴都算，由 `q_eff` 显著非零者判定为 RF 驱动轴（轴向通常 `q_eff ≈ 0`）。
- `**r_null`**：取 `trap_stability` 的 `center_um`（阱中心 ≈ RF 零场点，线性阱近似），用于 `β vs |X_sec−r_null|` 诊断图。
- **窗长鲁棒性**：`window_us` 默认自适应；若样本不足以填满 bin 则自动增大窗长并 warning。

