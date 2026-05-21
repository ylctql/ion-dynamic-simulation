# collision_pressure 模块教程

## 目录

1. [物理背景](#1-物理背景)
2. [模块架构](#2-模块架构)
3. [快速上手](#3-快速上手)
4. [势场指定](#4-势场指定)
5. [初始状态](#5-初始状态)
6. [碰撞力学详解](#6-碰撞力学详解)
7. [重构检测](#7-重构检测)
8. [压强估算与局限](#8-压强估算与局限)
9. [可视化](#9-可视化)
10. [构型预扫描](#10-构型预扫描)
11. [Python API 用法](#11-python-api-用法)
12. [文件格式](#12-文件格式)

---

## 1. 物理背景

在低温离子阱中，背景气体分子（H₂、He 等）与俘获离子的弹性碰撞会扰动离子晶格，可能导致**结构重构**（reconfiguration）——离子晶格从一种稳定构型跳变到另一种。

本模块模拟这一过程：

1. 采样一个背景气体分子的速度和碰撞参数
2. 计算散射角和动量冲量
3. 将冲量施加到随机选中的离子上
4. 积分碰撞后的离子晶格轨迹
5. 判断晶格是否发生了结构重构
6. 统计重构概率 $P_\text{flip}$
7. 结合实验观测率估算背景气压

### 离子-分子相互作用

由极化势描述：

$$U(r) = -\frac{C_4}{r^4}, \quad C_4 = \frac{\alpha q^2}{2(4\pi\varepsilon_0)^2}$$

其中 $\alpha$ 是分子极化率，$q$ 是离子电荷。

### Langevin 碰撞

临界碰撞参数将碰撞分为两类：

- **$b < b_c$**：螺旋轨道（spiral），散射角 $\theta = \pi$（完全后向散射）
- **$b \geq b_c$**：偏转轨道，散射角由 Binet 方程和椭圆积分给出

Langevin 碰撞速率系数（仅依赖基本物理常数）：

$$k_L = e\sqrt{\frac{\alpha\pi}{\mu\varepsilon_0}} \quad [\text{m}^3/\text{s}]$$

---

## 2. 模块架构

```
collision_pressure/
├── species.py              离子/分子物理参数
├── collision.py            碰撞力学：散射角、动量冲量
├── sampling.py             蒙特卡洛采样
├── topology.py             Delaunay 拓扑表征
├── reconfiguration.py      重构检测器
├── config_scan.py          平衡构型求解 + 构型库
├── simulation.py           碰撞模拟主循环
├── pressure.py             压强估算
├── visualize_collision.py  碰撞可视化
├── visualize_configs.py    构型库可视化
├── build_config_library.py 构型扫描 CLI 入口
├── __main__.py             模块 CLI 入口
├── __init__.py             公共 API 导出
└── configs/                预计算构型库
```

**数据流**：

```
势场 (--csv 或 --trap-freq)
  → FitResult3D (3D 多项式)
  → find_equilibrium → r₀ (平衡位置)
  → [Monte Carlo 循环]
      → sampling (v₀, b, ion, direction)
      → collision (θ, Δv)
      → solve_ivp (轨迹积分)
      → reconfiguration (结构重构？)
  → P_flip 统计
  → pressure 估算
```

---

## 3. 快速上手

### 基本碰撞模拟

```bash
python -m collision_pressure simulate \
  --trap-freq 2.0 3.0 0.1 \
  --n-ions 10 \
  --n-simulations 100
```

输出：
```
=== H2 Collision Simulation ===
  Ions:       N=10, Ba135+
  Molecule:   H2, T=10.0 K
  Detector:   zigzag
  Simulations: 100

[1/3] Setting up potential...
  Harmonic trap: fx=2.0 fy=3.0 fz=0.1 MHz

[2/3] Finding equilibrium...
  E = 0.004899 eV, success = True
  Flip-axis range: [-0.0176, 0.0148] um

[3/3] Running 100 collision simulations...
  Done (1.8s)

  Results: 65/100 reconfigured
  P_flip = 0.6500
  Langevin rate coefficient k_L = 1.4900e-15 m^3/s
  Pressure coefficient (P / R_obs) = 1.4255e-07 Pa*s
  => P = 1.4255e-07 * R_obs  (need experimental R_obs to get Pa)
```

### 带可视化的模拟

```bash
python -m collision_pressure simulate \
  --trap-freq 2.0 3.0 0.1 \
  --n-ions 10 \
  --n-simulations 50 \
  --visualize all \
  --viz-output results/
```

将在 `results/` 下生成：
- `trajectory_000.png` ~ `trajectory_002.png` — 碰撞轨迹 8 帧快照
- `before_after_reconfig.png` — 重构案例的前后对比
- `before_after_unchanged.png` — 未重构案例的前后对比
- `statistics.png` — 4 面板统计汇总

---

## 4. 势场指定

势场通过 `--csv` 或 `--trap-freq` 二选一指定。

### 4.1 理想谐振势 (--trap-freq)

指定三个轴向的阱频（MHz）：

```bash
python -m collision_pressure simulate \
  --trap-freq 2.0 3.0 0.1    # fx fy fz, 单位 MHz
```

对应势能：

$$V = \frac{1}{2}m(\omega_x^2 x^2 + \omega_y^2 y^2 + \omega_z^2 z^2) / q$$

适合理论研究和 zigzag 晶格测试。`fz << fx, fy` 的参数产生沿 z 轴延伸的线性链/zigzag 结构。

### 4.2 实验势场 (--csv + --config)

从实验测量的电场数据拟合：

```bash
python -m collision_pressure simulate \
  --csv data/monolithic20241118.csv \
  --config FieldConfiguration/configs/collision.json \
  --fit-mode quartic
```

`--fit-mode` 选项：
| 模式 | 基函数数量 | 说明 |
|------|-----------|------|
| `quadratic` | 10 | 纯谐振近似，快速 |
| `quartic` | 35 | 默认，含 4 阶修正项 |
| `quartic_even` | 35 | 仅偶数阶（镜像对称势阱） |
| `even` | 125 | 高阶偶函数 |
| `none` | 0 | 仅库仑力，无外势 |

---

## 5. 初始状态

碰撞发生时，离子晶格并非静止——它有热运动（声子）。初始状态有两种来源。

### 5.1 自动求解平衡构型（默认）

不指定 `--init-file` 时，模块自动通过 L-BFGS-B 最小化势能，找到静态平衡构型。离子初始速度设为零。

```bash
python -m collision_pressure simulate \
  --trap-freq 2.0 3.0 0.1 \
  --n-ions 10 \
  --maxiter 1000
```

这对应 T=0 的理想情况，适合快速测试和冷离子近似。

### 5.2 从文件加载 (r, v)

通过 `--init-file` 指定一个 `.npz` 文件，包含离子位置和速度：

```bash
python -m collision_pressure simulate \
  --trap-freq 2.0 3.0 0.1 \
  --init-file snapshot.npz \
  --n-simulations 100
```

npz 文件要求：
- **`r`**：shape `(N, 3)`，单位 μm（必需）
- **`v`**：shape `(N, 3)`，单位 m/s（可选，缺失则离子从静止开始）

**如何获取初始状态文件？**

可以从 `main.py` 动力学模拟的输出中提取快照：

```python
import numpy as np

# 假设已有动力学轨迹数据
data = np.load("trajectory.npz")
# 取某一帧
r = data["r"]   # (N, 3) um
v = data["v"]   # (N, 3) m/s
np.savez("snapshot.npz", r=r, v=v)
```

也可以用构型库中的构型：

```python
config = np.load("collision_pressure/configs/N105/config_000.npz")
np.savez("snapshot.npz", r=config["r"])
```

使用 `--init-file` 时，`--n-ions` 参数会被忽略，离子数量从文件推断。

---

## 6. 碰撞力学详解

### 6.1 蒙特卡洛采样

每次碰撞独立采样四个随机量：

| 量 | 采样方法 | 物理含义 |
|----|---------|---------|
| H₂ 速度 $v_0$ | Maxwell-Boltzmann $v^2 \sim \Gamma(3/2, 2k_BT/m)$ | 分子热速度 |
| 碰撞参数 $b$ | 均匀分布 $[0, 3b_c]$ | 最近距离 |
| 被击中离子 | 均匀随机选一个 | 所有离子等概率被击中 |
| 入射方向 | 球面均匀分布 | 各向同性 |

H₂ 速度由 `--molecule-temp`（默认 10 K）控制。

### 6.2 散射角计算

```
collision.scattering_angle(ion, mol, v0, b) → theta (rad)
```

- $b < b_c$（临界碰撞参数）：$\theta = \pi$（后向散射）
- $b \geq b_c$：通过 Binet 方程 + 第一类完全椭圆积分计算

### 6.3 动量冲量

```
collision.post_collision_kick(ion, mol, v0, theta, direction) → dv (m/s)
```

质心系中的速度转移：

$$\Delta v_\parallel = \frac{M_m}{M_i + M_m} v_0 (1 - \cos\theta), \quad \Delta v_\perp = \frac{M_m}{M_i + M_m} v_0 \sin\theta$$

对于 Ba⁺ + H₂：$M_m/(M_i + M_m) \approx 2/137 \approx 1.5\%$，所以单次冲量很小，但足以扰动 zigzag 构型。

### 6.4 轨迹积分

冲量施加到被击中离子后，积分所有离子的运动方程（含完整库仑力 + 外势）：

$$m\ddot{r}_i = -\nabla_i \left[ V_\text{trap}(\mathbf{r}) + \sum_{j \neq i} \frac{q^2}{4\pi\varepsilon_0 |\mathbf{r}_i - \mathbf{r}_j|} \right]$$

使用 `scipy.integrate.solve_ivp`（RK45），默认积分 50 μs、2000 步。

通过 `--t-integrate-us` 调整积分时间。

---

## 7. 重构检测

碰撞后判断晶格是否发生了结构重构。两种检测器：

### 7.1 Zigzag Flip 检测器

```bash
--detector zigzag
```

检测指定轴（默认 x，`--flip-axis 0`）上离子位置的符号翻转。适用于线性阱中的 zigzag 晶格——两个简并态常在 y 方向符号相反，此时使用 `--flip-axis 1`。

**原理**：记录平衡态中每个离子在 flip 轴上的符号，碰撞后检查是否有离子符号反转。

### 7.2 拓扑检测器

```bash
--detector topology
```

通用检测器，基于 Delaunay 三角剖分比较碰撞前后的邻接关系：

1. 将 3D 位置投影到 2D 平面（默认 xoz）
2. 构建 Delaunay 三角剖分 → 邻接图
3. 计算配位数指纹
4. 比较碰撞前后的拓扑是否相同

适用于任意离子晶格（2D 晶体、壳层结构等），但计算量更大。

---

## 8. 压强估算与局限

### 模拟给出的

模拟**只能**给出 $P_\text{flip}$（碰撞导致重构的概率）和 $k_L$（Langevin 速率系数）。

### 压强推导

$$P = \frac{R_\text{obs} \cdot k_B T}{P_\text{flip} \cdot k_L}$$

其中：
- $R_\text{obs}$：实验观测的重构事件发生率（1/s），**必须由实验测量**
- $P_\text{flip}$：模拟给出的重构概率
- $k_L = e\sqrt{\alpha\pi/(\mu\varepsilon_0)}$：Langevin 速率系数（纯理论值）
- $T$：分子温度

### 为什么不能仅靠模拟得到压强？

因为 Langevin 碰撞率 $\gamma_\text{el} = n \cdot k_L$ 依赖于气体数密度 $n$，而 $n$ 正是 $P = nk_BT$ 中要求解的量。模拟无法知道"实际碰撞有多频繁"——它只回答"碰撞发生后，有多大概率导致重构"。

### 使用方法

模拟输出的是**压力系数** $C_P = k_BT / (P_\text{flip} \cdot k_L)$，单位 Pa·s。将它乘以实验测量的重构率即可得到压强：

$$P = C_P \times R_\text{obs}$$

```python
from collision_pressure import estimate_pressure, BA_135, H2_MOLECULE

P_flip = 0.65
T = 10.0  # K
R_obs = 0.01  # 实验观测到的重构率，1/s

P = estimate_pressure(P_flip, BA_135, H2_MOLECULE, T, reconfig_rate=R_obs)
# P ≈ 1.4e-9 Pa
```

---

## 9. 可视化

### 9.1 碰撞轨迹快照

```bash
--visualize trajectory --viz-n-trajectories 5
```

为前 N 次碰撞生成 8 帧静态图（2 行 × 8 列），上行为 zox 投影，下行为 zoy 投影。被击中离子标红，平衡位置灰色。

### 9.2 前后对比

```bash
--visualize before-after
```

生成一张左右对比图，分别展示平衡态和碰撞后的离子位置。离子按 flip 轴符号着色（蓝色正、橙色负），被击中离子用红圈标记。

### 9.3 批量统计

```bash
--visualize statistics
```

4 面板汇总图：
- 散射角分布（重构 vs 未重构）
- H₂ 速度分布
- 各离子重构概率
- 冲量大小分布

### 9.4 组合使用

```bash
--visualize all    # 生成所有三种图
```

### 9.5 构型库可视化

```bash
python -m collision_pressure.visualize_configs configs/N105/
```

展示构型库中能量最低和最高的各 5 个构型。

---

## 10. 构型预扫描

在运行碰撞模拟之前，可以先枚举势能面上的多个局部极小构型。

```bash
python -m collision_pressure scan \
  --trap-freq 2.0 3.0 0.1 \
  --n-ions 54 \
  --n-scans 200 \
  --output-dir configs/N54
```

这会从 200 个随机初猜出发，用 L-BFGS-B 最小化能量，合并拓扑等价的构型，保存到指定目录。

输出目录包含：
- `config_000.npz` ~ `config_NNN.npz` — 各构型的位置、能量、拓扑
- `summary.json` — 构型目录（能量、指纹、配位数统计）

---

## 11. Python API 用法

除命令行外，所有功能均可通过 Python API 调用。

### 11.1 完整碰撞模拟

```python
import numpy as np
from collision_pressure import (
    setup_fit_harmonic, find_equilibrium,
    BA_135, H2_MOLECULE,
    ZigzagFlipDetector,
    run_collision_scan,
    estimate_pressure, langevin_rate_coefficient,
)

# 1. 势场
fit, _ = setup_fit_harmonic(2e6, 3e6, 0.1e6, mass_amu=135.0)

# 2. 平衡构型
rng = np.random.default_rng(42)
eq = find_equilibrium(fit, n_ions=10, rng=rng, maxiter=1000)

# 3. 重构检测器
det = ZigzagFlipDetector(flip_axis=0, sort_axis=2)
det.register_equilibrium(eq.r_um)

# 4. 批量碰撞
scan = run_collision_scan(
    eq.r_um, fit, BA_135, H2_MOLECULE,
    T=10.0,          # H₂ 温度 (K)
    n_simulations=100,
    detector=det, seed=43,
)

print(f"P_flip = {scan.reconfig_prob:.4f}")

# 5. 压强系数
coeff = estimate_pressure(scan.reconfig_prob, BA_135, H2_MOLECULE, 10.0)
print(f"P = {coeff:.4e} * R_obs")
```

### 11.2 单次碰撞 + 轨迹记录

```python
from collision_pressure import run_single_collision

rng = np.random.default_rng(0)
result = run_single_collision(
    eq.r_um, fit, BA_135, H2_MOLECULE,
    T=10.0, rng=rng, detector=det,
    save_trajectory=True,    # 保存完整轨迹
)

print(f"Reconfigured: {result.reconfigured}")
print(f"Scattering angle: {result.theta:.4f} rad")
print(f"Kick: {np.linalg.norm(result.dv):.4f} m/s")
print(f"Trajectory shape: {result.trajectory.shape}")  # (6N, n_steps)
```

### 11.3 带初速度的碰撞

```python
# 从动力学轨迹中加载快照
snap = np.load("snapshot.npz")
r = snap["r"]   # (N, 3) um
v = snap["v"]   # (N, 3) m/s

det = ZigzagFlipDetector(flip_axis=0, sort_axis=2)
det.register_equilibrium(r)

scan = run_collision_scan(
    r, fit, BA_135, H2_MOLECULE,
    T=10.0, n_simulations=50,
    detector=det, seed=42,
    v_init_um_us=v,  # 传入初速度
)
```

### 11.4 使用实验势场

```python
from collision_pressure import setup_fit

fit, cfg = setup_fit(
    csv_path="data/monolithic20241118.csv",
    config_path="FieldConfiguration/configs/collision.json",
    fit_mode="quartic",
)
```

### 11.5 碰撞力学独立使用

```python
from collision_pressure import scattering_angle, post_collision_kick, sample_velocity
from collision_pressure.species import reduced_mass
import numpy as np

# 约化质量
mu = reduced_mass(BA_135, H2_MOLECULE)
print(f"Reduced mass: {mu/1.6605e-27:.4f} amu")

# 采样 H₂ 速度
rng = np.random.default_rng(42)
v0 = sample_velocity(T=10.0, mass_amu=2.016, rng=rng)
print(f"H2 speed: {v0:.2f} m/s")

# 计算散射角
theta = scattering_angle(BA_135, H2_MOLECULE, v0, b=5e-9)
print(f"Scattering angle: {np.degrees(theta):.2f} deg")

# 计算冲量
direction = np.array([1.0, 0.0, 0.0])
dv = post_collision_kick(BA_135, H2_MOLECULE, v0, theta, direction)
print(f"Kick magnitude: {np.linalg.norm(dv):.4f} m/s")
```

### 11.6 可视化 API

```python
from collision_pressure import (
    plot_trajectory_snapshots, plot_before_after, plot_batch_statistics,
)

# 轨迹快照
plot_trajectory_snapshots(
    result.trajectory, result.time_us, eq.r_um, result.hit_ion,
    n_snapshots=8, output="traj.png",
)

# 前后对比
r_final = result.trajectory[:3*N, -1].reshape(N, 3)
plot_before_after(
    eq.r_um, r_final, result.hit_ion,
    flip_axis=0, reconfigured=result.reconfigured, output="compare.png",
)

# 批量统计
plot_batch_statistics(scan.collisions, N, output="stats.png")
```

---

## 12. 文件格式

### 初始状态 npz (--init-file)

| 字段 | Shape | 单位 | 必需 |
|------|-------|------|------|
| `r` | `(N, 3)` | μm | 是 |
| `v` | `(N, 3)` | m/s | 否 |

### 碰撞结果 npz (--output)

| 字段 | Shape | 说明 |
|------|-------|------|
| `r0` | `(N, 3)` | 初始位置 (μm) |
| `reconfigured` | `(n_sim,)` | 是否重构 (bool) |
| `v0` | `(n_sim,)` | H₂ 速度 (m/s) |
| `b` | `(n_sim,)` | 碰撞参数 (m) |
| `theta` | `(n_sim,)` | 散射角 (rad) |
| `hit_ion` | `(n_sim,)` | 被击中离子索引 |
| `reconfig_prob` | scalar | 重构概率 |
| `langevin_coeff` | scalar | Langevin 系数 k_L (m³/s) |
| `pressure_coeff_Pa_s` | scalar | 压力系数 (Pa·s) |

### 构型库 (scan 输出)

每个 `config_XXX.npz`：

| 字段 | Shape | 说明 |
|------|-------|------|
| `r` | `(N, 3)` | 离子位置 (μm) |
| `energy_total_eV` | scalar | 总能量 |
| `energy_trap_eV` | scalar | 势阱能量 |
| `energy_coulomb_eV` | scalar | 库仑能量 |
| `success` | scalar | 优化是否收敛 |
| `nit` | scalar | 优化迭代次数 |
| `coord_numbers` | `(N,)` | 配位数 |
| `adj_matrix` | `(N, N)` | 邻接矩阵 |

---

## CLI 参数速查

### simulate 子命令

```
势场 (二选一):
  --trap-freq FX FY FZ     谐振势阱频 (MHz)
  --csv PATH               电场 CSV 文件

离子晶格:
  --n-ions N               离子数量 (默认 10)
  --ion-species SPEC       Ba135+ / Ba138+ / Yb171+ (默认 Ba135+)
  --mass-amu M             离子质量 (amu, 默认 135.0)
  --init-file PATH         预加载初态 npz

碰撞参数:
  --molecule MOL           H2 / He (默认 H2)
  --molecule-temp T        分子温度 K (默认 10.0)
  --n-simulations N        蒙特卡洛采样数 (默认 100)

检测器:
  --detector TYPE          zigzag / topology (默认 zigzag)
  --flip-axis AXIS         zigzag 翻转轴 0=x/1=y/2=z (默认 0)

积分:
  --t-integrate-us T       积分时间 μs (默认 50.0)
  --maxiter N              平衡构型优化最大迭代 (默认 1000)
  --softening-um S         库仑软化长度 μm (默认 0.001)

输出:
  --output PATH            保存结果 npz
  --visualize TYPE         none / trajectory / before-after / statistics / all
  --viz-output DIR         可视化输出目录
  --viz-n-trajectories N   记录轨迹的碰撞数 (默认 3)
```

### scan 子命令

```
  --n-ions N               离子数量 (默认 54)
  --n-scans N              随机初猜次数 (默认 200)
  --output-dir DIR         输出目录
  --trap-freq FX FY FZ     或 --csv PATH (二选一)
```
