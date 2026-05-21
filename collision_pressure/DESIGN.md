# H2 弹性碰撞压强估算模块 — 设计方案

> 基于 Pagano et al., "Cryogenic ion trap for high-fidelity quantum logic", Phys. Rev. A 102, 032616 (2020)

## 1. 模块定位与目标

新建独立包 `collision_pressure/`，实现通过模拟中性 H₂ 分子与俘获离子晶格的弹性碰撞，观察碰撞后离子晶格是否发生结构重构，从重构概率反推低温区背景气压的方法。

与论文仅处理 zigzag 晶格（二元 flip 判据）不同，本模块需支持任意构型的离子晶格，包括 zigzag 和 2D 晶体。

### 1.1 物理背景

在低温离子阱中，背景气体分子（主要是 H₂）与俘获离子的弹性碰撞可导致离子晶格发生结构重构（reconfiguration）。通过测量重构事件的发生率，可以估算背景气压：

$$P = \frac{\gamma_{\text{el}} \, k_B T}{P_{\text{flip}} \cdot e \sqrt{\frac{\alpha \pi}{\mu \varepsilon_0}}}$$

其中：
- $\gamma_{\text{el}}$ 是弹性碰撞速率
- $P_{\text{flip}}$ 是单次碰撞导致结构重构的概率（由蒙特卡洛模拟得到）
- $\alpha$ 是分子极化率
- $\mu$ 是约化质量
- $T$ 是分子温度

### 1.2 碰撞力学

离子-中性分子相互作用由极化势描述：

$$U(r) = -\frac{C_4}{r^4}, \quad C_4 = \frac{\alpha q^2}{2(2\pi\varepsilon_0)^2}$$

**Langevin 碰撞速率**：

$$\gamma = n \cdot e \sqrt{\frac{\alpha \pi}{\mu \varepsilon_0}}$$

**临界碰撞参数**：

$$b_c = \left(\frac{8 C_4}{\mu v_0^2}\right)^{1/4}$$

- 当 $b < b_c$：螺旋轨道，散射角 $\theta = \pi$（完全后向散射）
- 当 $b \geq b_c$：偏转轨道，散射角由 Binet 方程给出：

$$\theta = \pi - \frac{2\sqrt{2}}{\sqrt{1+x}} \, K\!\left(\frac{x}{1+x}\right)$$

其中 $x = \sqrt{1 - \frac{16 C_4 \mu^2 E}{L^4}}$，$K$ 为第一类完全椭圆积分。

**碰撞后速度**（质心系）：

$$v_x = \frac{M_m}{M_i + M_m} v_0 (1 - \cos\theta), \quad v_y = -\frac{M_m}{M_i + M_m} v_0 \sin\theta$$

---

## 2. 物理流程总览

```
步骤 0: (可选) 构型预扫描 ───── 从随机初猜出发，枚举势能面的多个局部极小 ── equilibrium 模块
步骤 1: 求平衡构型 r₀ ──────── equilibrium 模块
步骤 2: 计算声子模态 ────────── equilibrium.phonon 模块
步骤 3: 采样 H₂ 分子速度 (Maxwell-Boltzmann) + 碰撞参数 b
步骤 4: 计算散射角 θ (Langevin 碰撞力学)
步骤 5: 对被击中离子施加动量冲量
步骤 6: 用 ComputeKernel 积分后碰撞轨迹 (含完整库伦力+外势)
步骤 7: 检测结构重构 (通用判据，非仅 flip)
步骤 8: 统计重构概率 → 反推压强
```

---

## 3. 文件结构

```
collision_pressure/
├── __init__.py              # 包入口，导出公共 API
├── DESIGN.md                # 本设计文档
├── species.py               # 离子/分子物种参数 (质量, 极化率等)
├── collision.py             # 碰撞力学: Langevin 速率, 散射角, 动量冲量
├── sampling.py              # 蒙特卡洛采样: 速度, 碰撞参数, 碰撞离子选择
├── topology.py              # 构型拓扑表征: Delaunay 三角剖分, 邻接图, 构型指纹
├── reconfiguration.py       # 结构重构检测 (分层策略: 拓扑 → 几何)
├── config_scan.py           # (可选) 构型预扫描：枚举局部极小构型
├── simulation.py            # 单次碰撞模拟主循环 (步骤 3-7)
├── pressure.py              # 压强提取: 从重构概率到气压
├── cli.py                   # 命令行入口
└── configs/
    └── default.json         # 默认参数配置
```

---

## 4. 各模块详细设计

### 4.1 `species.py` — 物种参数

定义离子和中性分子的物理参数。

```python
@dataclass(frozen=True)
class Species:
    name: str
    mass_amu: float        # 原子质量单位
    polarizability: float  # SI, m³ (分子极化率 α)
    charge_ec: float = 1.0 # 元电荷数

# 预定义物种
H2_MOLECULE = Species("H2", 2.016, polarizability=8.04e-31)
BA_135 = Species("Ba135+", 135.0, charge_ec=1.0)   # 默认离子
BA_138 = Species("Ba138+", 138.0, charge_ec=1.0)
YB_171 = Species("Yb171+", 171.0, charge_ec=1.0)
```

**设计要点**：
- 预置常用物种常量，同时支持用户自定义
- 不硬编码物种，所有物理计算通过 `Species` 参数化
- 可扩展支持 He、N₂ 等其他背景气体

### 4.2 `collision.py` — 碰撞力学

纯函数设计，无项目内依赖，便于独立测试。

| 函数 | 公式 | 说明 |
|------|------|------|
| `langevin_rate(ion, mol, T, n)` | $\gamma = n \cdot e\sqrt{\alpha\pi/(\mu\varepsilon_0)}$ | Langevin 碰撞速率 |
| `critical_impact_param(ion, mol, v0)` | $b_c = (8C_4/(\mu v_0^2))^{1/4}$ | 临界碰撞参数 |
| `scattering_angle(ion, mol, v0, b)` | Binet 方程，含椭圆积分 | 散射角 θ |
| `post_collision_kick(ion, mol, v0, theta)` | 质心系动量冲量 | 施加到被击中离子的速度增量 |
| `polarization_coefficient(ion, mol)` | $C_4 = \alpha q^2/(2(2\pi\varepsilon_0)^2)$ | 极化系数 |
| `reduced_mass(ion, mol)` | $\mu = M_i M_m / (M_i + M_m)$ | 约化质量 |

**`scattering_angle` 详细逻辑**：

```python
def scattering_angle(ion, mol, v0, b):
    mu = reduced_mass(ion, mol)
    C4 = polarization_coefficient(ion, mol)
    E = 0.5 * mu * v0**2           # 碰撞能
    L = mu * v0 * b                 # 角动量
    bc = critical_impact_param(ion, mol, v0)

    if b < bc:
        return pi                    # 螺旋轨道，完全后向散射

    x = sqrt(1 - 16 * C4 * mu**2 * E / L**4)
    return pi - 2*sqrt(2) / sqrt(1+x) * ellipk(x / (1+x))
```

### 4.3 `sampling.py` — 蒙特卡洛采样

| 函数 | 说明 |
|------|------|
| `sample_velocity(T, mass_amu, rng)` | 从 Maxwell-Boltzmann 分布采样 H₂ 速度标量 |
| `sample_impact_parameter(b_max, rng)` | 均匀采样碰撞参数 $b \in [0, b_{\max}]$ |
| `sample_collision_ion(N, rng)` | 随机选择被碰撞的离子索引（均匀分布） |
| `sample_direction(rng)` | 随机入射方向（球面均匀分布，返回 3D 单位向量） |

**设计要点**：
- `rng` 参数接受 `numpy.random.Generator`，确保可复现性
- `b_max` 由 `critical_impact_param` 的数倍确定，确保覆盖所有有效碰撞
- 所有函数为纯函数，无副作用

### 4.4 `topology.py` — 构型拓扑表征

定义离子晶格构型的拓扑表示，作为构型比较和重构检测的基础设施。

**核心思想**：离子晶格的"结构"本质上是"谁和谁是邻居"。Delaunay 三角剖分直接从点集提取邻接关系，
天然与粒子标签无关，且为离散量，对热涨落鲁棒。

**关键类型**：

```python
@dataclass(frozen=True)
class TopologyFingerprint:
    """构型拓扑指纹，用于快速比较"""
    coord_seq: tuple[int, ...]    # 按升序排列的配位数序列，如 (3,3,4,4,4,5,6,6)
    n_boundary: int               # 边界（凸包）离子数
    shell_structure: tuple[int, ...]  # 各壳层离子数，如 (6, 12, 1) 表示外层6中层12中心1

@dataclass
class CrystalTopology:
    """完整的晶格拓扑"""
    adj_matrix: np.ndarray        # (N, N) 邻接矩阵，bool
    adjacency: list[list[int]]    # 各离子的邻居索引列表
    coord_numbers: np.ndarray     # (N,) 各离子配位数
    boundary: np.ndarray          # 边界离子索引
    fingerprint: TopologyFingerprint
```

**核心函数**：

| 函数 | 说明 |
|------|------|
| `build_topology(r, plane)` | 从位置数组构建 `CrystalTopology`；先用凸包确定边界，再对内部离子做 Delaunay 三角剖分，提取邻接关系 |
| `same_topology(t1, t2)` | 比较两个拓扑：先比指纹（O(N log N)），指纹相同再比邻接矩阵的图同构性 |
| `topology_distance(t1, t2)` | 拓扑距离：配位数分布的差异 + 图编辑距离的近似 |
| `project_to_plane(r)` | 将 3D 位置投影到阱平面（通常 xoz），用于 2D 三角剖分 |

**`build_topology` 流程**：

1. 将 3D 坐标投影到阱平面 → 2D 点集
2. `scipy.spatial.ConvexHull` → 边界离子
3. `scipy.spatial.Delaunay` → 三角剖分 → 邻接矩阵
4. 统计各离子配位数 → 排序 → 构型指纹
5. 按距中心距离分层 → 壳层结构

**构型指纹的比较效率**：

指纹相同的两个构型**不一定**拓扑相同（存在不同构但配位数分布相同的图），
但指纹不同的两个构型**一定**拓扑不同。因此：

- 指纹比较（O(N log N)）：快速排除不同构型
- 邻接矩阵精确比较（O(N²)）：仅在指纹相同时执行
- 对于典型离子数 N=10~100，两者都很快

### 4.5 `reconfiguration.py` — 结构重构检测（分层策略）

采用**两层检测**架构：先比较拓扑（离散、标签无关），拓扑不变时再比较几何（连续、标签相关）。

```python
class ReconfigResult:
    """重构检测结果"""
    reconfigured: bool            # 是否发生重构
    topology_changed: bool        # 拓扑是否改变
    geometry_changed: bool        # 几何是否改变（仅拓扑不变时有意义）
    target_config_idx: int | None # 匹配到的目标构型索引（预扫描模式）
```

```python
class ReconfigDetector(ABC):
    """结构重构检测器基类"""
    @abstractmethod
    def register_equilibrium(self, r0: np.ndarray) -> None: ...
    @abstractmethod
    def check(self, r_final: np.ndarray) -> ReconfigResult: ...

class TopologyDetector(ReconfigDetector):
    """第 1 层 + 第 2 层的通用检测器"""
```

**`TopologyDetector` 检测流程**：

```
第 1 层: 拓扑比较
    r_final → build_topology() → t_final
    与 t_initial 比较:
        → 拓扑不同 → 一定是重构 (topology_changed=True)
        → 拓扑相同 → 进入第 2 层

第 2 层: 几何比较（同拓扑框架内）
    对 zigzag 晶格:
        → 检查相邻离子 y 坐标的交替顺序是否翻转 (手性变化)
    对一般 2D/3D 晶格:
        → 检查有向三角形面积符号变化（镜像/手性变化）
        → 或: 用 Hungarian matching 找最优标签对应后比较坐标偏差
```

**第 2 层的必要性**：zigzag flip 不改变拓扑（邻居关系完全相同，只是 y 坐标正负号交换），
但实验上是可观测的结构变化。其他晶格也可能存在拓扑不变但几何结构改变的情况（如旋转对称构型的手性翻转）。

**与预扫描的结合**（当 `--scan-configurations` 启用时）：

```
第 1 层: 拓扑比较
    t_final 与所有预扫描构型的拓扑 {t_k} 比较
    → 找到匹配的拓扑 t_k*
    → 若 t_k* 不是初始构型 → 重构，并记录跃迁通道 k*

第 2 层: 几何比较
    → 在拓扑 k* 内部做几何精确匹配
```

### 4.5.1 `config_scan.py` — 构型预扫描（可选步骤）

对于一般 2D/3D 离子晶格，势能面存在多个局部极小值。碰撞后晶体可能落入不同的亚稳态，
而非唯一的目标态。预扫描这些构型可显著提升重构检测的物理准确性。

**动机**：

- 纯拓扑检测只能判断"拓扑是否改变"，无法识别"跃迁到了哪个已知亚稳态"
- 不同跃迁路径的能垒不同，碰撞后进入低能垒邻态的概率更高。不区分通道会导致 P_reconfig 物理含义模糊
- 计算成本可控：枚举平衡构型是静态优化问题（复用 `equilibrium` 模块），远比轨迹积分便宜

**方法**：

1. 从随机初始位置出发，多次运行 L-BFGS-B 最小化（复用 `equilibrium.total_energy_and_grad()`）
2. 对每个结果调用 `build_topology()` 构建拓扑
3. 用 `same_topology()` 合并拓扑等价的构型（而非逐元素坐标比较）
4. 输出：一组 `{r_k, E_k, t_k}` 代表各不等价的局部极小构型及其能量和拓扑

**拓扑合并优于坐标比较的原因**：

- 从不同初猜独立优化的构型，离子排列顺序任意，逐元素坐标比较无物理意义
- 拓扑比较天然与标签无关，且计算效率 O(N log N) 远优于 Hungarian matching 的 O(N³)
- 同一极小值附近的多次收敛必然产生相同拓扑，不会误合并不同构型

用户通过 `--scan-configurations N` 启用（N 为随机初猜次数，默认 0 = 不扫描）。

### 4.6 `simulation.py` — 单次碰撞模拟主循环

```
输入: equilibrium result (r0, fit), phonon result, 碰撞参数
输出: 是否发生重构 (bool), 轨迹数据 (可选)
```

**流程**：

1. 从 `equilibrium` 模块获取 r₀（平衡位置）、`FitResult3D`（外势拟合）
2. 从 `phonon` 模块获取 `PhononResult`（声子模态、频率）
3. 初始化 `ReconfigDetector` 并注册 r₀
4. 采样 H₂ 速度 v₀、碰撞参数 b、被击中离子 i
5. `collision.scattering_angle()` → θ
6. `collision.post_collision_kick()` → Δv（施加到离子 i）
7. 构建初始条件：`r_init = r0`，`v_init[i] += Δv`（其余为零）
8. 构建力函数（见下方）
9. 调用轨迹积分（见下方）
10. `detector.check(r_final)` → bool

**力函数构建策略**：

碰撞模拟需要包含完整力（外势 + 库伦），积分时间较短（微秒量级）。两种方案：

- **方案 A**（推荐）：复用 `FieldParser.force.build_force()` 构建外势力 + `ComputeKernel` 的 C++ 后端处理库伦力，获得 CUDA 加速
- **方案 B**（回退）：利用 `equilibrium.total_energy_and_grad()` 在 Python 层构建力函数，用 `scipy.integrate.solve_ivp` 积分，不需要 C++ 后端但速度较慢

**轨迹积分**：

- 积分时间：由 phonon result 的 $\omega_{\min}$ 决定，约几个最低声子周期
- 积分时间公式：$t_{\text{int}} = C_t \cdot 2\pi / \omega_{\min}$，$C_t$ 默认为 3-5
- 复用 `ComputeKernel` 的 RK4 / Velocity Verlet 积分器

### 4.7 `pressure.py` — 压强提取

```python
def estimate_pressure(
    reconfig_prob: float,    # 模拟得到的重构概率 P_reconfig
    ion: Species,            # 离子物种
    molecule: Species,       # 中性分子物种
    T_molecule: float,       # 分子温度 (K)
) -> float:
    """从重构概率估算背景气压 (Pa)

    P = γ_el · kB · T / (P_reconfig · e · √(απ/(με₀)))
    """
```

### 4.8 `cli.py` — 命令行入口

```
python -m collision_pressure \
    --field-data data/monolithic20241118.csv \
    --config FieldConfiguration/configs/collision.json \
    --n-ions 10 \
    --ion-species Ba135+ \
    --molecule H2 \
    --molecule-temp 10 \
    --n-simulations 1000 \
    --detector deviation \
    --deviation-threshold 0.05 \
    --integrator RK \
    --device cuda \
    --seed 42 \
    --output results.npz
```

**CLI 完整参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--field-data` | （必需） | 电场 CSV 文件路径 |
| `--config` | （必需） | 电压配置 JSON 路径 |
| `--n-ions` | 10 | 离子数量 |
| `--ion-species` | Ba135+ | 离子物种 |
| `--molecule` | H2 | 背景气体分子物种 |
| `--molecule-temp` | 10 | 分子温度 (K) |
| `--n-simulations` | 1000 | 蒙特卡洛采样次数 |
| `--detector` | deviation | 重构检测策略：flip / deviation / combined |
| `--deviation-threshold` | auto | 结构偏差阈值 (μm)，auto 表示由声子零点运动标定 |
| `--integrator` | RK | 积分器：RK / VV |
| `--device` | cpu | 计算设备：cpu / cuda |
| `--equilibrium-file` | None | 预计算的平衡构型 .npz，跳过步骤 1-2 |
| `--scan-configurations` | 0 | 构型预扫描随机初猜次数，0 = 不扫描 |
| `--seed` | None | 随机种子 |
| `--output` | None | 结果输出路径 |
| `--save-trajectories` | False | 保存所有碰撞轨迹 |

**工作流**：

1. 加载场数据和配置（复用 `FieldConfiguration` + `FieldParser`）
2. 求平衡构型（调用 `equilibrium` 模块，或通过 `--equilibrium-file` 加载已有结果）
3. （可选）构型预扫描：枚举势能面局部极小（`--scan-configurations` 启用）
4. 计算声子模态
5. 并行运行 N 次碰撞模拟（`multiprocessing.Pool` 或 `concurrent.futures.ProcessPoolExecutor`）
6. 统计重构概率（若启用了构型预扫描，可额外输出各跃迁通道的转移概率）
7. 估算压强
8. 保存结果和统计信息

---

## 5. 模块依赖关系

```
┌──────────────────────────────────────────────────────────┐
│                    collision_pressure/                    │
│                                                          │
│  cli.py ──→ simulation.py ──→ pressure.py                │
│               │      │                                   │
│               │      ├── sampling.py   (蒙特卡洛采样)     │
│               │      ├── collision.py  (碰撞力学)         │
│               │      ├── topology.py   (构型拓扑表征)     │
│               │      ├── reconfiguration.py (分层重构检测) │
│               │      └── config_scan.py (构型预扫描, 可选) │
│               │                                          │
│               ├── species.py (物种参数)                    │
│               │                                          │
│               ├── equilibrium/   (r₀, fit, phonon)  ← 复用│
│               ├── ComputeKernel/ (轨迹积分)          ← 复用│
│               ├── FieldParser/   (力函数)            ← 复用│
│               └── FieldConfiguration/ (陷阱参数)     ← 复用│
└──────────────────────────────────────────────────────────┘
```

**复用的现有模块**：

| 需求 | 复用的模块 | 方式 |
|------|-----------|------|
| 平衡构型 r₀ | `equilibrium.find_equilibrium` | 直接调用，或加载已有 `.npz` |
| 外势拟合 | `equilibrium.fit_potential_3d_quartic` | 通过 equilibrium 包获取 `FitResult3D` |
| 声子模态 | `equilibrium.solve_phonon_modes` | 获取本征向量用于重构检测 |
| 力函数 | `FieldParser.force.build_force()` | 构建 `force(r,v,t)` 可调用对象 |
| 轨迹积分 | `ComputeKernel.backend.CalculationBackend` | 积分碰撞后轨迹 |
| 陷阱参数 | `FieldConfiguration` | `Config`, `FieldSettings`, `Voltage` |
| 数据类型 | `utils.Frame`, `utils.Message` | 帧数据和控制消息 |

---

## 6. 数据格式

### 6.1 输入

- 电场 CSV：复用 `FieldParser.csv_reader.read()` 格式
- 电压配置 JSON：复用 `FieldConfiguration/configs/` 格式
- 预计算平衡构型 `.npz`：`equilibrium` 模块输出格式（含 `r` 字段）

### 6.2 输出

`.npz` 结果文件包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `r0` | (N, 3) | 平衡构型 (μm) |
| `reconfigured` | (n_sim,) bool | 各次碰撞是否发生重构 |
| `reconfig_prob` | scalar | 重构概率 |
| `pressure_estimate_Pa` | scalar | 最终压强估算 (Pa) |
| `impact_params` | (n_sim,) | 碰撞参数 b |
| `velocities` | (n_sim,) | H₂ 速度 (m/s) |
| `scattering_angles` | (n_sim,) | 散射角 θ (rad) |
| `hit_ions` | (n_sim,) int | 被击中的离子索引 |
| `transition_channels` | (n_sim,) int | 跃迁目标构型索引（仅预扫描模式，0=未重构） |
| `config` | dict | 所有输入参数快照 |

---

## 7. 扩展性考虑

- **多分子种类**：`species.py` 可扩展支持 He、N₂ 等背景气体
- **温度扫描**：CLI 支持 `--molecule-temp-range` 进行温度依赖性研究
- **碰撞参数扫描**：支持固定 b 扫描或自动确定 b_max
- **可视化**：可选输出碰撞轨迹动画（复用 `Plotter` 模块），或碰撞前后构型对比图
- **并行化**：每次碰撞模拟独立，天然适合 `multiprocessing.Pool` 或 `concurrent.futures.ProcessPoolExecutor`
- **批量模式**：支持对不同离子数、不同构型批量运行模拟
