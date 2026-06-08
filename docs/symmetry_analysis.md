# 势场对称性定量分析

本文档描述 ISM-Main 中势场对称性分析模块的设计原理、使用方法和结果解读。

> **相关模块**：对称性分析中的多项式奇偶性 (`p`) 和 Hessian 离轴项 (`h`) 指标被 `field_optimize` 模块用作电压优化的正则化惩罚项。详见 [阱频反向设计文档](field_optimize.md)。

## 概述

对称性分析模块（`field_visualize/symmetry.py`）对给定电势场的 CSV 格点数据和电压配置，从四个独立维度定量评估势场的空间对称性。每种分析分别对 **DC 静电势**、**RF 赝势**和**总势场（DC + 赝势）**进行。

四种分析类型的标识符：

| 标识 | 名称 | 开销 | 说明 |
|------|------|------|------|
| `m` | 镜面对称 (Mirror) | 低 | 势场关于坐标平面的反射对称性 |
| `r` | 旋转对称 (Rotational) | 低 | 势场关于坐标轴的柱对称性 |
| `p` | 多项式奇偶性 (Polynomial Parity) | 中 | 3D 多项式拟合后分析奇次项系数 |
| `h` | Hessian 非对角项 (Hessian) | 中 | 中心点 Hessian 矩阵的交叉耦合程度 |

> **`m` 和 `r` 仅需势场插值采样，速度较快；`p` 和 `h` 需要做 3D 多项式拟合，耗时较长。** 可按需选择分析类型以节省计算时间。

---

## CLI 使用

### 基本语法

```bash
python field_visualize.py \
  --csv <CSV路径> \
  --config <JSON配置路径> \
  --symmetry <类型列表> \
  [其他参数]
```

`--symmetry` 参数接受逗号分隔的分析类型标识：

```bash
# 仅计算镜面对称性
--symmetry m

# 计算镜面和旋转对称性
--symmetry m,r

# 计算全部四种
--symmetry all
# 等价于
--symmetry m,r,p,h
```

### 完整示例

```bash
python field_visualize.py \
  --csv data/circle_RF_r100.csv \
  --config FieldConfiguration/configs/circle_rf_r100.json \
  --const 0,0,0 \
  --x_range -80,80 --y_range -80,80 --z_range -300,300 \
  --symmetry m,r \
  --symmetry-n-mirror 15 \
  --symmetry-n-rot 60 \
  --symmetry-radar
```

### 相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--symmetry` | string | （空） | 分析类型：`m`, `r`, `p`, `h`, `all`，逗号分隔 |
| `--symmetry-n-mirror` | int | 10 | 镜面对称每轴采样点数（总点数 = N³） |
| `--symmetry-n-rot` | int | 50 | 旋转对称每轴采样点数 |
| `--symmetry-n-fit` | int | 8 | 多项式拟合每轴采样点数 |
| `--symmetry-radar` | flag | — | 输出雷达图 |
| `--symmetry-heatmap` | flag | — | 输出偏差热力图 |
| `--const` | string | `0,0,0` | 分析中心点坐标 (µm) |
| `--x_range` | string | `-100,100` | x 方向采样范围 (µm) |
| `--y_range` | string | `-100,100` | y 方向采样范围 (µm) |
| `--z_range` | string | `-100,100` | z 方向采样范围 (µm) |

---

## Python API

```python
from FieldConfiguration.constants import init_from_config
from FieldConfiguration.loader import field_settings_from_config
from FieldParser.calc_field import calc_field, calc_potential
from FieldParser.csv_reader import read as read_csv
from field_visualize.symmetry import compute_symmetry_report

# 1. 加载配置和格点数据
cfg, config = init_from_config("FieldConfiguration/configs/circle_rf_r100.json")
grid_coord, grid_voltage = read_csv(
    "data/circle_RF_r100.csv", None,
    normalize=True, dl=cfg.dl, dV=cfg.dV
)
n_voltage = grid_voltage.shape[1]
field_settings = field_settings_from_config(
    "data/circle_RF_r100.csv",
    "FieldConfiguration/configs/circle_rf_r100.json",
    n_voltage, cfg
)

# 2. 构建插值器
potential_interps = calc_potential(grid_coord, grid_voltage)
field_interps = calc_field(grid_coord, grid_voltage)

# 3. 运行对称性分析（选择需要的类型）
report = compute_symmetry_report(
    potential_interps,
    field_interps,
    field_settings.voltage_list,
    cfg,
    center_um=(0.0, 0.0, 0.0),
    range_um=((-80, 80), (-80, 80), (-300, 300)),
    which=frozenset("mr"),  # 仅计算镜面和旋转对称性
)

# 4. 访问结果
report.dc.mirror["yz"].coefficient        # DC 势关于 yz 平面的镜面对称系数
report.pseudo.rotational["z"].coefficient  # 赝势关于 z 轴的旋转对称系数
report.total.polynomial.s_parity_xy        # 总势关于 xy 平面的多项式奇偶性（若计算）
report.total.hessian.offdiag_ratio         # 总势 Hessian 非对角比（若计算）
```

`which` 参数接受 `frozenset`，包含以下任意组合：
- `frozenset("mrph")` — 全部四种（默认）
- `frozenset("m")` — 仅镜面对称
- `frozenset("mr")` — 镜面 + 旋转
- 以此类推

---

## 四种分析维度的数学原理

### 1. 镜面对称系数 (m)

**物理含义**：势场关于通过中心点的坐标平面（yz / xz / xy）是否具有反射对称性。

**计算方法**：

在采样区域内取 $N^3$ 个格点 $\mathbf{r}_i$，对每个点计算其关于坐标平面的镜像点 $\mathbf{r}_i^{\text{mirror}}$（例如 yz 平面：$x \to 2x_c - x$）。定义**对称基线**为两者的均值：

$$V_{\text{sym}} = \frac{V(\mathbf{r}) + V(\mathbf{r}^{\text{mirror}})}{2}$$

对称系数为非对称残差的 RMSE 相对于对称基线 RMS 的比值：

$$S_{\text{mirror}} = 1 - \frac{\text{RMSE}\big(V(\mathbf{r}),\; V(\mathbf{r}^{\text{mirror}})\big)}{\text{RMS}(V_{\text{sym}})}$$

其中：
- $\text{RMSE} = \sqrt{\frac{1}{N}\sum_i (V(\mathbf{r}_i) - V(\mathbf{r}_i^{\text{mirror}}))^2}$
- $\text{RMS}(V_{\text{sym}}) = \sqrt{\frac{1}{N}\sum_i V_{\text{sym},i}^2}$

- $S \in [0, 1]$，**1 = 完美镜面对称**
- 同时报告最大相对偏差和平均相对偏差（均相对于 $\text{RMS}(V_{\text{sym}})$）
- **归一化方式**：使用对称基线 RMS 而非 Vmax - Vmin，因此 S 不依赖势场绝对深度，反映的是非对称成分占对称成分的比例

**典型用途**：验证电极加工/装配的左右对称性、上下对称性。

### 2. 旋转对称系数 (r)

**物理含义**：势场关于某坐标轴的柱对称性，即绕该轴旋转 90° 后两个正交平面内的势场分布是否一致。

**计算方法**：

对旋转轴 $a$，取垂直于 $a$ 的两个正交平面，在两个平面内用相同的参数范围采样。例如对 z 轴：

- 平面 1（xz）：$V_1(s, c_y, t)$，其中 $s$ 沿 x 变化，$t$ 沿 z 变化
- 平面 2（yz）：$V_2(c_x, s, t)$，其中 $s$ 沿 y 变化，$t$ 沿 z 变化

定义对称基线 $V_{\text{sym}} = (V_1 + V_2) / 2$：

$$S_{\text{rot}} = 1 - \frac{\text{RMSE}(V_1,\; V_2)}{\text{RMS}(V_{\text{sym}})}$$

- $S \in [0, 1]$，**1 = 完美柱对称**
- 对于圆对称阱（如 `circle_RF`），z 轴旋转对称系数应接近 1
- 归一化方式与镜面对称相同，使用对称基线 RMS

**典型用途**：评估径向束缚的各向同性程度。

### 3. 多项式系数奇偶性 (p)

**物理含义**：势场在中心附近的多极展开中，奇阶成分（破坏对称性的项）的占比。

**计算方法**：

1. 在中心点附近做 3D 四阶多项式最小二乘拟合（35 项，$i+j+k \leq 4$）：

$$V(\mathbf{r}) = \sum_{i+j+k \leq 4} c_{ijk} \, u^i v^j w^k$$

其中 $u = (x - x_c)/L$、$v = (y - y_c)/L$、$w = (z - z_c)/L$ 为缩放坐标。

2. 若势场关于某坐标平面完美对称，则对应方向的奇次项系数为零（如关于 yz 平面对称则 $i$ 为奇的项均为零）。

3. 为消除不同阶数单项式的量级差异，使用 RMS 缩放：

$$\tilde{c}_{ijk} = c_{ijk} \cdot \sqrt{\frac{8}{(2i+1)(2j+1)(2k+1)}}$$

4. 对称系数：

$$S_{\text{parity}} = 1 - \frac{\|\tilde{c}_{\text{odd}}\|_2}{\|\tilde{c}\|_2}$$

- $S \in [0, 1]$，**1 = 所有奇次项为零（完美对称）**
- 同时报告最大的 5 个奇次项（诊断哪些非对称项贡献最大）

**典型用途**：精细评估势阱对称性，诊断导致对称性破缺的具体多极成分。

### 3.5 多项式拟合系数表

**物理含义**：对比势阱的谐波（二次）成分与非谐（高阶）成分的相对大小。

**输出内容**：

1. **二次项 (harmonic)**：$x^2, y^2, z^2$ 的缩放系数和原始系数。这些项决定阱频和主轴方向。
2. **非谐项 (anharmonic)**：所有其他项（按缩放系数绝对值降序排列的 top 5），包括：
   - 交叉耦合项 $xy, xz, yz$（标注 `cross`）——主轴不沿坐标轴时的二次交叉项
   - 三次、四次等高阶项——离子偏离中心时的非线性畸变

**缩放方式**：与奇偶性分析相同，$\tilde{c} = |c| \cdot \text{RMS}(x^i y^j z^k)$，消除阶数差异后可跨阶比较。

**输出示例**：

```
  [Total] R² = 0.99973
  Quadratic (harmonic):
    x^2       c̃= 1.234e-01   c= +4.560e-02
    y^2       c̃= 1.236e-01   c= +4.580e-02
    z^2       c̃= 2.150e-02   c= +7.950e-03
  Top anharmonic terms:
    x y       c̃= 3.210e-03   c= +2.200e-03  (cross)
    x^4       c̃= 1.050e-03   c= +5.300e-04
    x^2 z^2   c̃= 8.720e-04   c= +3.800e-04
```

**典型用途**：
- 评估谐波近似的适用范围：非谐项与二次项的比值越小，谐波近似越好
- 识别主导的交叉耦合方向（cross 项），判断是否需要坐标旋转
- 诊断高阶畸变的来源，指导电极设计优化

### 4. Hessian 非对角项分析 (h)

**物理含义**：势阱主轴与坐标轴的对齐程度。

**计算方法**：

在中心点计算势场的 $3 \times 3$ Hessian 矩阵（通过 3D 多项式拟合的解析二阶导数）：

$$\mathbf{H} = \begin{pmatrix} \kappa_{xx} & \kappa_{xy} & \kappa_{xz} \\ \kappa_{xy} & \kappa_{yy} & \kappa_{yz} \\ \kappa_{xz} & \kappa_{yz} & \kappa_{zz} \end{pmatrix}$$

报告指标：
- 6 个独立的 Hessian 分量（单位 V/µm²）
- 非对角项比值：

$$\text{ratio} = \frac{\max(|\kappa_{xy}|,\; |\kappa_{xz}|,\; |\kappa_{yz}|)}{\text{mean}(|\kappa_{xx}|,\; |\kappa_{yy}|,\; |\kappa_{zz}|)}$$

- ratio 越小，表示势阱主轴与坐标轴越对齐（对称性越好）

**典型用途**：评估势阱是否存在交叉耦合，是否可以通过坐标旋转消除。

---

## 结果解读

### 对称系数参考范围

| 系数 | 完美对称 | 优秀 | 良好 | 需注意 |
|------|---------|------|------|--------|
| 镜面 $S_{\text{mirror}}$ | 1.000 | > 0.99 | > 0.95 | < 0.90 |
| 旋转 $S_{\text{rot}}$ | 1.000 | > 0.99 | > 0.95 | < 0.90 |
| 奇偶性 $S_{\text{parity}}$ | 1.000 | > 0.99 | > 0.95 | < 0.90 |
| Hessian ratio | 0 | < 0.01 | < 0.05 | > 0.10 |

### 不同构型的典型结果

| 构型 | 镜面 yz/xz/xy | 旋转 z 轴 | Hessian ratio |
|------|--------------|----------|---------------|
| 圆对称 RF 阱 | ≈1.0 / ≈1.0 / ≈1.0 | ≈1.0 | ≈0 |
| 线形阱 | ≈1.0 / ≈1.0 / 较低 | 较低 | 取决于对齐 |
| 双层阱 | ≈1.0 / 较低 / ≈1.0 | 较低 | 较低 |

### 常见诊断场景

1. **镜面对称系数低** → 电极加工不对称或电压配置不平衡
2. **旋转对称系数低** → 径向各向异性（可能是设计意图，如线形阱）
3. **多项式奇偶性差 + 奇次项诊断** → 可以确定具体哪个多极成分导致对称性破缺
4. **Hessian 非对角比大** → 势阱主轴与坐标轴不对齐，可能需要坐标旋转

---

## 模块架构

```
field_visualize/
├── symmetry.py          # 对称性分析核心逻辑
│   ├── compute_mirror_symmetry()
│   ├── compute_rotational_symmetry()
│   ├── compute_polynomial_symmetry()  → equilibrium/potential_fit_3d.py
│   ├── compute_hessian_symmetry()     → equilibrium/potential_fit_3d.py
│   ├── compute_potential_symmetry()   # 单一势场类型的聚合
│   └── compute_symmetry_report()      # DC/pseudo/total 完整报告
├── plots.py
│   ├── print_symmetry_report()         # 终端文本输出
│   ├── plot_symmetry_radar()           # 雷达图
│   └── plot_symmetry_deviation_heatmap() # 偏差热力图
├── cli.py               # CLI 入口，--symmetry 参数解析
└── core.py              # compute_potentials(), um_to_norm() 等基础函数
```

### 数据流

```
CSV 格点 + JSON 电压配置
    │
    ├─ csv_reader.read() → grid_coord, grid_voltage
    ├─ calc_potential()  → potential_interps[]  (V 插值器)
    ├─ calc_field()      → field_interps[]      (E 插值器)
    │
    └─ compute_symmetry_report(which={"m","r","p","h"})
         │
         ├─ 对每种势场类型 (dc / pseudo / total):
         │    ├─ "m": 采样 N³ 点，V(r) vs V(mirror(r))     ← 快
         │    ├─ "r": 采样 N²×2 点，比较正交平面             ← 快
         │    ├─ "p": 3D quartic 拟合 → 奇次项系数分析       ← 中
         │    └─ "h": 3D quartic 拟合 → Hessian 解析导数     ← 中
         │
         └─ SymmetryReport(dc, pseudo, total)
```

---

## 性能参考

以 `circle_RF_r100.csv`（~60k 格点）为例，在普通工作站上的典型耗时：

| 分析类型 | 采样点数 | 单次耗时（DC+pseudo+total） |
|---------|---------|--------------------------|
| m（镜面） | 10³/次 | < 1s |
| r（旋转） | 50²/次 | < 2s |
| p（多项式） | 8³=512 点拟合 | ~3-5s |
| h（Hessian） | 8³=512 点拟合 | ~3-5s |

> `p` 和 `h` 共享同一个 3D 多项式拟合结果，当两者同时计算时仅做一次拟合，额外开销很小。
