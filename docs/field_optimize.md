# 阱频反向设计：从目标频率反推电极电压

> 适用于 `field_optimize` 模块 — 离子阱势场的反向设计工具

## 目录

1. [概述](#1-概述)
2. [数学原理](#2-数学原理)
3. [CLI 使用](#3-cli-使用)
4. [Python API](#4-python-api)
5. [输出格式](#5-输出格式)
6. [参数详解](#6-参数详解)
7. [完整示例](#7-完整示例)
8. [性能与调优](#8-性能与调优)
9. [模块架构](#9-模块架构)
10. [常见问题](#10-常见问题)

---

## 1. 概述

### 问题背景

在离子阱实验中，**阱频** (trap frequency) 是描述离子势阱束缚强度的最核心参数：

$$f_i = \frac{1}{2\pi}\sqrt{\frac{2q\kappa_{ii}}{m}} \quad (i = x, y, z)$$

其中 $\kappa_{ii}$ 是势场在中心处的 Hessian 对角分量（曲率），$q$ 为离子电荷，$m$ 为离子质量。

传统的正向流程为：

```
电极电压 → 势场叠加 → 多项式拟合 → 曲率 κ → 阱频 f
```

但在实际实验调参中，经常需要解决**反向问题**：给定期望的阱频 $(f_x, f_y, f_z)$，应该设置什么样的电极电压？

### 模块功能

`field_optimize` 模块实现了从目标阱频到电极电压的自动搜索：

1. **频率匹配**：优化电极电压（DC 偏置，可选 RF 幅值）使势场阱频逼近目标值
2. **对称性正则化**：以势场对称性作为惩罚项，确保得到的势场物理合理
3. **快速收敛**：利用势场的线性叠加特性预计算基函数矩阵，每次目标函数评估仅需 ~0.5 ms（DC-only 模式）

### 适用场景

- 实验中快速找到产生特定阱频的电压配置
- 阱设计验证：确认给定电极几何能否实现目标频率组合
- 参数空间探索：研究电压-阱频的映射关系
- 对称性优化：在保持阱频的同时改善势场对称性

---

## 2. 数学原理

### 2.1 势场的线性叠加

势场是各电极贡献的线性叠加（对于 DC 电极）和二次叠加（对于 RF 赝势）：

$$V_{\text{total}}(\mathbf{r}) = \underbrace{\sum_{i \in \text{DC}} V_{\text{bias},i} \cdot \Phi_i(\mathbf{r})}_{V_{\text{dc}}} + \underbrace{\frac{e}{4m\Omega^2}\left(\frac{V_{\text{dl}}}{V_{\text{dl}}}\right)^2 \left|\sum_{j \in \text{RF}} V_{0,j} \cdot \mathbf{E}_j(\mathbf{r})\right|^2}_{V_{\text{pseudo}}}$$

其中 $\Phi_i(\mathbf{r})$ 是第 $i$ 个电极的单位电势基函数（从 CSV 格点数据插值），$\mathbf{E}_j(\mathbf{r})$ 是电场基函数。

**关键性质**：
- DC 部分是 $V_{\text{bias}}$ 的**线性函数** → 优化 landscape 凸（仅频率项时）
- RF 赝势是 $V_0$ 的**二次函数** → 光滑但略有非线性
- 阱频 $f \propto \sqrt{\kappa}$，其中 $\kappa$ 是势场的曲率 → 单调递增关系

### 2.2 目标函数

$$\mathcal{L}(\mathbf{x}) = w_{\text{freq}} \sum_{i \in \{x,y,z\}} \left(\frac{f_i(\mathbf{x}) - f_i^{\text{target}}}{f_i^{\text{target}}}\right)^2 + w_{\text{parity}} \sum_{i} (1 - S_{\text{parity},i}(\mathbf{x})) + w_{\text{offdiag}} \cdot R_{\text{offdiag}}(\mathbf{x})$$

其中：

- **频率误差项**：三个轴向阱频的相对误差平方和，确保匹配目标
- **奇偶性惩罚**：$S_{\text{parity}} \in [0,1]$ 为多项式奇偶性系数（1 = 完美对称），复用 `field_visualize/symmetry.py` 中的 `_parity_coefficient()`
- **Hessian 离轴惩罚**：$R_{\text{offdiag}} = \max|\kappa_{\text{offdiag}}| / \overline{|\kappa_{\text{diag}}|}$，惩罚主轴交叉耦合

### 2.3 快速预计算

优化循环中最频繁的操作是"给定电压 → 计算阱频"。由于基函数 $\Phi_i, \mathbf{E}_i$ 只取决于 CSV 格点数据（固定），可以**预计算**它们在采样点的值：

$$\text{DC-only: } V_{\text{total}}(s) = \mathbf{\Phi}(s) \cdot \mathbf{V}_{\text{bias}} + V_{\text{pseudo, const}}(s)$$

其中 $\mathbf{\Phi}(s)$ 是 $n_{\text{pts}} \times n_{\text{elec}}$ 的矩阵，每次迭代仅需一次矩阵-向量乘法。这使得每次目标函数评估的时间从 ~10 ms（完整插值+拟合）降至 ~0.5 ms。

### 2.4 优化器

使用 `scipy.optimize.minimize`，默认 **L-BFGS-B** 方法：

- 支持 box 约束（电压上下界）
- 利用数值差分梯度（变量少，~10 个维度，代价极低）
- 典型 50-200 次评估收敛，总耗时 1-10 秒
- 可选 Nelder-Mead 作为 fallback（无约束，对不光滑 landscape 更鲁棒）

---

## 3. CLI 使用

### 基本语法

```bash
python -m field_optimize --csv <CSV路径> --config <JSON路径> --target-freq fx fy fz [选项]
```

### 必选参数

| 参数 | 说明 |
|------|------|
| `--csv` | 电场 CSV 路径；可仅传文件名自动在 `data/` 下查找 |
| `--config` | 电压配置 JSON 路径；可仅传文件名自动在 `FieldConfiguration/configs/` 下查找 |
| `--target-freq` | 目标阱频 (MHz)，三个浮点数：`fx fy fz` |

### 最简示例

```bash
python -m field_optimize --csv default.csv --config default.json --target-freq 2.0 3.0 0.1
```

> **注意**：当范围参数以负数开头（如 `-50,50`）时，argparse 可能将负号误认为选项前缀。请使用 `=` 语法：`--x-range=-50,50`

### 输出示例

```
============================================================
  电极电压优化结果
============================================================

            目标       优化前       优化后     误差%
------------------------------------------------------------
  f_x (MHz)     2.0000     1.5234     2.0012   0.058%
  f_y (MHz)     3.0000     2.8765     2.9987   0.043%
  f_z (MHz)     0.1000     0.0834     0.1003   0.284%

  电极     类型     初始 V       优化 V       变化
------------------------------------------------------------
      RF   RF   V0=275.00→275.00  V_bias=0.00→0.00
      U1   DC       0.0000      -2.3451    -2.3451
      U2   DC       0.0000       1.2345    +1.2345
      ...

收敛: 成功 (47 iter, 384 eval)
目标函数: 1.23e-01 → 2.45e-06
信息: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
============================================================
```

---

## 4. Python API

### 基础用法

```python
from pathlib import Path
from FieldConfiguration.constants import init_from_config
from FieldConfiguration.loader import field_settings_from_config
from FieldParser.calc_field import calc_field, calc_potential
from FieldParser.csv_reader import read as read_csv
from field_optimize import optimize_voltages, OptimizationConfig

# 1. 加载配置
cfg, config = init_from_config("FieldConfiguration/configs/default.json")
grid_coord, grid_voltage = read_csv(
    "data/default.csv", None, normalize=True, dl=cfg.dl, dV=cfg.dV
)
n_voltage = grid_voltage.shape[1]
field_settings = field_settings_from_config(
    "data/default.csv", "FieldConfiguration/configs/default.json",
    n_voltage, cfg
)

# 2. 构建插值器（一次性）
potential_interps = calc_potential(grid_coord, grid_voltage)
field_interps = calc_field(grid_coord, grid_voltage)

# 3. 配置优化参数
opt_config = OptimizationConfig(
    target_freq_MHz=(2.0, 3.0, 0.1),    # 目标阱频 (MHz)
    center_um=(0.0, 0.0, 0.0),           # 评估中心 (µm)
    fit_range_um=((-50, 50), (-20, 20), (-150, 150)),  # 各轴拟合范围 (µm)
    w_freq=1.0,                          # 频率误差权重
    w_parity=0.1,                        # 对称性惩罚权重
    w_offdiag=0.1,                       # Hessian 惩罚权重
)

# 4. 运行优化
result = optimize_voltages(
    potential_interps, field_interps, cfg,
    field_settings.voltage_list, opt_config
)

# 5. 访问结果
print(f"优化成功: {result.success}")
print(f"目标频率: {result.target_freqs_MHz}")
print(f"优化后频率: {result.optimized_freqs_MHz}")
print(f"迭代次数: {result.n_iterations}")

# 优化后的电压列表
for v in result.optimized_voltages:
    print(f"  {v['name']}: V_bias={v['V_bias']:.4f} V")
```

### 仅优化频率（无对称性惩罚）

```python
opt_config = OptimizationConfig(
    target_freq_MHz=(2.0, 3.0, 0.1),
    w_parity=0.0,      # 关闭奇偶性惩罚
    w_offdiag=0.0,      # 关闭 Hessian 惩罚
)
```

### 同时优化 RF 幅值

```python
opt_config = OptimizationConfig(
    target_freq_MHz=(2.0, 3.0, 0.1),
    optimize_rf_v0=True,               # 同时优化 RF V0
    v0_rf_bounds=(50.0, 500.0),         # RF 幅值搜索范围
)
```

### 高精度拟合

```python
opt_config = OptimizationConfig(
    target_freq_MHz=(2.0, 3.0, 0.1),
    fit_degree=4,                        # 四阶多项式拟合
    n_fit_pts=300,                       # 更多采样点
    tol=1e-10,                           # 更严格的收敛条件
    maxiter=500,                         # 更多迭代
)
```

---

## 5. 输出格式

输出 JSON 与现有 `FieldConfiguration/configs/` 中的配置格式**完全兼容**，可直接用于 `main.py`、`field_visualize` 等模块：

```json
{
  "_comment": "Optimized voltage configuration from field_optimize",
  "g": 0.1,
  "voltage_list": [
    {"type": "rf", "name": "RF", "V0": 275.0, "V_bias": 0.0, "frequency": 35.28},
    {"type": "dc", "name": "U1", "V_bias": -2.345123},
    {"type": "dc", "name": "U2", "V_bias": 1.234567},
    {"type": "dc", "name": "U3", "V_bias": 0.567890},
    {"type": "dc", "name": "U4", "V_bias": -1.234567}
  ],
  "_optimization": {
    "target_freq_MHz": [2.0, 3.0, 0.1],
    "achieved_freq_MHz": [2.0012, 2.9987, 0.1003],
    "iterations": 47,
    "n_evaluations": 384,
    "success": true,
    "timestamp": "2026-06-08T14:30:00"
  }
}
```

### 验证优化结果

```bash
# 用 field_visualize 验证优化后的阱频
python field_visualize.py --csv default.csv --config optimized.json

# 检查对称性
python field_visualize.py --csv default.csv --config optimized.json --symmetry p,h
```

---

## 6. 参数详解

### 评估参数

| 参数 | CLI | 默认值 | 说明 |
|------|-----|--------|------|
| 评估中心 | `--center` | `0,0,0` | 评估阱频的中心点坐标 (µm) |
| x 轴拟合范围 | `--x-range` | `-50,50` | x 方向 1D 多项式拟合范围 (µm) |
| y 轴拟合范围 | `--y-range` | `-20,20` | y 方向 1D 多项式拟合范围 (µm) |
| z 轴拟合范围 | `--z-range` | `-150,150` | z 方向 1D 多项式拟合范围 (µm) |
| 多项式阶数 | `--fit-degree` | `2` | 1D 多项式拟合阶数（2 或 4） |
| 采样点数 | `--n-fit-pts` | `200` | 每轴 1D 采样点数 |

### 权重参数

| 参数 | CLI | 默认值 | 说明 |
|------|-----|--------|------|
| 频率权重 | `--w-freq` | `1.0` | 频率相对误差的权重 |
| 奇偶性权重 | `--w-parity` | `0.1` | 多项式奇偶性惩罚权重（设 0 关闭） |
| Hessian 权重 | `--w-offdiag` | `0.1` | Hessian 离轴比惩罚权重（设 0 关闭） |

**权重调节建议**：
- 默认值（1.0 / 0.1 / 0.1）在大多数情况下适用
- 若更关心精确频率匹配：增大 `--w-freq` 或减小 `--w-parity`、`--w-offdiag`
- 若更关心势场对称性：增大 `--w-parity` 和 `--w-offdiag`（如 0.5 或 1.0）
- 若不需要对称性约束：设置 `--w-parity=0 --w-offdiag=0` 可显著加速

### 优化参数

| 参数 | CLI | 默认值 | 说明 |
|------|-----|--------|------|
| DC 电压边界 | `--v-bias-bounds` | `-100,100` | DC 偏置电压搜索范围 (V) |
| RF 幅值边界 | `--v0-rf-bounds` | `50,500` | RF 幅值搜索范围 (V) |
| 优化 RF | `--optimize-rf-v0` | 关 | 同时优化 RF 幅值 V0 |
| 最大迭代 | `--maxiter` | `200` | 优化器最大迭代次数 |
| 收敛容限 | `--tol` | `1e-8` | 优化器收敛判据 |
| 优化方法 | `--method` | `L-BFGS-B` | 优化方法（`L-BFGS-B` 或 `Nelder-Mead`） |

### 对称性评估参数

| 参数 | CLI | 默认值 | 说明 |
|------|-----|--------|------|
| 拟合模式 | `--symmetry-fit-mode` | `quartic` | 3D 多项式拟合模式 |
| 采样点数 | `--symmetry-n-pts` | `6` | 对称性评估每轴采样点数（总计 N³） |

### 势场平滑

| 参数 | CLI | 默认值 | 说明 |
|------|-----|--------|------|
| 平滑方向 | `--smooth-axes` | `z` | SG 平滑坐标轴（`none` 关闭） |
| SG 参数 | `--smooth-sg` | `11,3` | window_length,polyorder |

### 输出

| 参数 | CLI | 默认值 | 说明 |
|------|-----|--------|------|
| 输出路径 | `--out` | 自动生成 | 输出 JSON 文件路径 |

---

## 7. 完整示例

### 示例 1：基础 DC 电压优化

```bash
python -m field_optimize \
  --csv default.csv \
  --config default.json \
  --target-freq 2.0 3.0 0.1
```

以默认配置为初始猜测，优化 7 个 DC 电极偏置电压，使阱频逼近 (2.0, 3.0, 0.1) MHz。

### 示例 2：带对称性约束的精细优化

```bash
python -m field_optimize \
  --csv default.csv \
  --config default.json \
  --target-freq 2.0 3.0 0.1 \
  --w-parity 0.5 \
  --w-offdiag 0.3 \
  --fit-degree 4 \
  --out optimized_strict.json
```

增强对称性惩罚权重，使用四阶多项式拟合以提高频率计算精度。

### 示例 3：同时优化 RF 幅值

```bash
python -m field_optimize \
  --csv default.csv \
  --config default.json \
  --target-freq 1.5 2.5 0.08 \
  --optimize-rf-v0 \
  --v0-rf-bounds=100,400 \
  --maxiter 500
```

当目标阱频与初始配置差距较大时，可同时调整 RF 幅值以扩大搜索空间。

### 示例 4：纯频率优化（无对称性约束）

```bash
python -m field_optimize \
  --csv default.csv \
  --config default.json \
  --target-freq 2.0 3.0 0.1 \
  --w-parity=0 \
  --w-offdiag=0
```

关闭对称性惩罚，仅匹配频率。速度最快。

### 示例 5：Python API 完整流程

```python
"""使用 field_optimize 的完整 Python 脚本示例"""
from field_optimize import optimize_voltages, OptimizationConfig
from FieldConfiguration.constants import init_from_config
from FieldConfiguration.loader import field_settings_from_config
from FieldParser.calc_field import calc_field, calc_potential
from FieldParser.csv_reader import read as read_csv

# 加载
cfg, config = init_from_config("FieldConfiguration/configs/default.json")
grid_coord, grid_voltage = read_csv(
    "data/default.csv", None, normalize=True, dl=cfg.dl, dV=cfg.dV
)
n_voltage = grid_voltage.shape[1]
fs = field_settings_from_config(
    "data/default.csv", "FieldConfiguration/configs/default.json",
    n_voltage, cfg
)

# 插值器
pot = calc_potential(grid_coord, grid_voltage)
fld = calc_field(grid_coord, grid_voltage)

# 优化
opt_config = OptimizationConfig(
    target_freq_MHz=(2.0, 3.0, 0.1),
    w_parity=0.2,
    w_offdiag=0.2,
)
result = optimize_voltages(pot, fld, cfg, fs.voltage_list, opt_config)

# 输出
if result.success:
    print(f"优化成功！迭代 {result.n_iterations} 次")
    for v in result.optimized_voltages:
        if v["type"] == "dc":
            print(f"  {v['name']}: V_bias = {v['V_bias']:.4f} V")
    for axis in "xyz":
        f = result.optimized_freqs_MHz[f"f_{axis}"]
        print(f"  f_{axis} = {f:.4f} MHz")
```

---

## 8. 性能与调优

### 典型性能

| 模式 | 每次评估 | 总评估数 | 总耗时 |
|------|---------|---------|--------|
| DC-only, 无对称性 | ~0.5 ms | 100-200 | < 1 s |
| DC-only, 含对称性 | ~15 ms | 100-200 | 2-5 s |
| DC+RF, 含对称性 | ~15 ms | 200-400 | 5-10 s |

### 加速技巧

1. **关闭对称性惩罚**：`--w-parity=0 --w-offdiag=0` 可将评估时间降至 ~0.5 ms
2. **减小对称性采样点**：`--symmetry-n-pts 5`（125 点 vs 默认 216 点）
3. **使用二阶拟合**：`--fit-degree 2`（比四阶更快，对大多数阱已足够）
4. **减少采样点**：`--n-fit-pts 100`（从 200 减至 100）

### 收敛问题排查

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| 频率误差大 | 电压边界过窄 | 增大 `--v-bias-bounds` 范围 |
| 优化未收敛 | 初始值离目标太远 | 尝试 Nelder-Mead：`--method Nelder-Mead` |
| 某轴频率为 NaN | 该方向 anti-trapping | 放宽电压边界，调整初始配置 |
| 结果对初始值敏感 | 多个局部最优 | 用 Nelder-Mead 或多起点搜索 |
| 对称性变差 | 惩罚权重太小 | 增大 `--w-parity` 和 `--w-offdiag` |

---

## 9. 模块架构

### 文件结构

```
field_optimize/
├── __init__.py       # 公开 API：optimize_voltages, OptimizationConfig, OptimizationResult
├── __main__.py       # python -m field_optimize 入口
├── cli.py            # CLI 参数解析、路径解析、报告输出、JSON 导出
├── optimizer.py      # optimize_voltages() 核心循环
├── objective.py      # FastEvaluator 预计算 + compute_objective()
└── types.py          # 数据类定义
```

### 数据流

```
CSV 格点 + JSON 电压配置
    │
    ├─ csv_reader.read()    → grid_coord, grid_voltage
    ├─ calc_potential()     → potential_interps[]   (一次构建)
    ├─ calc_field()         → field_interps[]       (一次构建)
    ├─ build_voltage_list() → voltage_list
    │
    └─ build_fast_evaluator()   ← 预计算 Phi_1d, E_1d, Phi_3d, E_3d
         │
         └─ scipy.optimize.minimize(compute_objective)
              │
              ├─ 矩阵-向量乘法: V_dc = Phi_1d @ V_bias    ← ~0.01 ms
              ├─ 1D 多项式拟合 → κ₂ → f                   ← ~0.1 ms
              ├─ 频率误差计算                                ← ~0.001 ms
              ├─ [可选] 3D quartic 拟合 + 对称性惩罚        ← ~10 ms
              │
              └─ 收敛 → 优化后 voltage_list → JSON 输出
```

### 依赖关系

```
field_optimize/
  ├─ 复用 FieldParser/potential_fit.py        (fit_potential_1d, k2_to_trap_freq_MHz)
  ├─ 复用 field_visualize/core.py             (compute_potentials, build_grid_1d, um_to_norm)
  ├─ 复用 field_visualize/symmetry.py         (_parity_coefficient)
  ├─ 复用 equilibrium/potential_fit_3d.py     (fit_potential_3d_quartic, hessian_fit_3d)
  └─ 复用 FieldConfiguration/loader.py        (build_voltage_list, field_settings_from_config)
```

---

## 10. 常见问题

### Q: 优化后的电压值是否物理可实现？

模块通过 `--v-bias-bounds` 和 `--v0-rf-bounds` 限制搜索范围。默认值 ±100 V 覆盖大多数实验条件。请根据实际电源输出能力设置合理的边界。

### Q: 为什么有时某个轴的频率无法匹配？

可能原因：
1. 该方向的势场形状无法通过调节现有电极电压实现目标频率（电极几何限制）
2. 电压边界过窄
3. 目标频率组合在物理上不可实现（如径向频率过高需要更强的 RF 幅值）

解决方案：放宽电压边界，或使用 `--optimize-rf-v0` 同时调节 RF 幅值。

### Q: 对称性惩罚是否影响频率精度？

是的，存在 trade-off：对称性权重越大，频率误差可能略大。但通常在 $w_{\text{parity}} = w_{\text{offdiag}} = 0.1$ 的默认设置下，频率精度仍然很好（相对误差 < 1%），同时势场对称性显著改善。

### Q: DC-only 和 DC+RF 模式如何选择？

- **DC-only**（默认）：适用于微调阱频、RF 幅值已经固定的情况。速度快，landscape 简单
- **DC+RF**：适用于需要大幅调整阱频的情况。RF 幅值影响赝势（二次关系），搜索空间更大但收敛可能更慢

### Q: 如何验证优化结果？

```bash
# 方法 1：用 field_visualize 验证阱频和对称性
python field_visualize.py --csv <csv> --config optimized.json --symmetry p,h

# 方法 2：用 main.py 运行模拟验证动力学行为
python main.py --csv <csv> --config optimized.json --N 10 --time 10 --plot
```

### Q: 输出 JSON 可以直接用于其他模块吗？

可以。输出格式与 `FieldConfiguration/configs/` 中的配置完全兼容。唯一的区别是包含一个额外的 `_optimization` 字段（以 `_` 开头，不影响解析）和 `_comment` 字段。
