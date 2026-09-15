# 多项式系数势（直接指定多项式系数构造动力学势场）

本文档描述 `--poly-potential` 力场来源：跳过 COMSOL 网格 CSV，由用户在一个 JSON 文件内**显式写出多项式系数**，直接构造离子动力学所需的外势场。

> **相关模块**：装载层在 `equilibrium/potential_fit_3d.py`，力构建在 `FieldParser/force.py`，经 `main.py::_build_force` 接入动力学。与 `--csv`（网格插值）、`--trap-freq`（理想谐振势）三者**互斥**，为 `FieldSettings` 的第三种势场来源。

## 目录

1. [动机](#1-动机)
2. [数学形式与量纲](#2-数学形式与量纲)
3. [JSON 文件格式](#3-json-文件格式)
4. [term-label 语法规则](#4-term-label-语法规则)
5. [CLI 使用](#5-cli-使用)
6. [三种力场来源对比](#6-三种力场来源对比)
7. [Python API](#7-python-api)
8. [系数从哪来](#8-系数从哪来)
9. [物理含义与局限](#9-物理含义与局限)
10. [完整示例](#10-完整示例)

---

## 1. 动机

默认动力学力（`--csv`）来自对 COMSOL 格点势 `np.gradient` + 三维线性插值；`--poly-fit` 虽用多项式解析梯度，但系数仍是从 CSV **拟合**得到的。两种方式都依赖网格数据。

而很多场景希望**直接用解析多项式定义势场**，绕开网格：

- 用理想/手调的多项式势做扫描、对照实验、算法验证；
- 把 `equilibrium` 模块拟合出的系数直接喂回动力学；
- 构造任意（含高阶交叉项）的光滑势场，消除插值噪声。

`--poly-potential` 即为此提供：用户在 JSON 里写出单项式 → 系数，程序解析为多项式、取解析梯度作为力。

## 2. 数学形式与量纲

势场模型与 `equilibrium` 的 3D 高次拟合**完全同形式**：

$$V(\mathbf{r}) = \sum_{ijk} c_{ijk}\, u^i v^j w^k,\qquad u=\frac{x-x_0}{L},\ v=\frac{y-y_0}{L},\ w=\frac{z-z_0}{L}$$

- `u, v, w` 为**无量纲**缩放坐标，`L = scale_um`（µm）为半跨度，`(x_0,y_0,z_0) = center_um`（µm）；
- `V` 单位为**伏特 (V)**。

**关键（量纲）**：因 `u,v,w` 无量纲，每个系数 `c_{ijk}` 的量纲就是 **V**——与 `fit_potential_3d_quartic` 拟合输出、`potential_fit_coeff.json` 中的系数量纲完全一致。高阶项数值可能很大（因 `u∈[-1,1]` 需放大系数），属正常现象。

离子受力（与其它模式一致的归一化约定）：

$$\mathbf{F} = q\,\mathbf{E} - \gamma\mathbf{v},\qquad \mathbf{E} = -\nabla V$$

梯度由多项式**解析微分**得到（`grad_fit_3d`，单位 V/µm），再经 `dl/dV` 转为内核使用的无量纲电场。

## 3. JSON 文件格式

```json
{
  "coefficients": {
    "1": 0.0,
    "x^2": 0.513,
    "y^2": 4.635,
    "z^2": 0.0236,
    "x^4*y^3": -206.78,
    "x^2*y^2*z^2": 0.327
  },
  "center_um": [0.0, 0.0, 0.0],
  "scale_um": 100.0,
  "potential_offset_V": 0.0
}
```

| 字段 | 类型 | 必填 | 默认 | 含义 |
|------|------|------|------|------|
| `coefficients` | `{term-label: float}` | 是 | — | 单项式标签 → 系数 (V)，见 [§4](#4-term-label-语法规则) |
| `center_um` | `[x0, y0, z0]` (µm) | 否 | `[0,0,0]` | 缩放中心；不写则系数无定义，强烈建议显式给出 |
| `scale_um` | `float` (µm) | 否 | `100.0` | 缩放半跨度 L；不写则系数无定义，强烈建议显式给出 |
| `potential_offset_V` | `float` (V) | 否 | `0.0` | 势能零点平移；**对力无影响**（常数项梯度为零），可省略 |

> **只列出你要的项**：未列出的单项式系数视为 0。可只写二次项（等价理想谐振势），也可写出任意高阶交叉项。`coefficients` 中不需要的项直接省略即可——这正是相比"按阶数枚举"更灵活之处。

> **常数项不影响动力学**：`"1"`（常数）与 `potential_offset_V` 都只平移势能零点，梯度为零，不改变力。仅当后续要做能量比较时才有意义。

## 4. term-label 语法规则

`coefficients` 的键是单项式标签，由 `quartic_3d_term_label` 定义、`parse_term_label` 反向解析：

- **变量**：仅 `x`、`y`、`z`；
- **因子**：`var`（一次）或 `var^n`（`n` 为非负整数），如 `x`、`z^4`；
- **组合**：因子间以 `*` 连接，如 `x*y`、`x^2*y^3*z`；
- **顺序无关**：`y*x` 与 `x*y` 等价（均解析为 `(1,1,0)`），但同一变量不得重复出现；
- **次数上限**：每变量次数 `0..4`（与 `(5,5,5)` 系数张量、quartic 高次拟合一致）；越界报错；
- **常数项**：`"1"`（或空串）表示 `(0,0,0)`。

| 标签 | 指数 (i,j,k) | 合法 |
|------|--------------|------|
| `"1"` | (0,0,0) | ✅ |
| `"z"` | (0,0,1) | ✅ |
| `"x^2"` | (2,0,0) | ✅ |
| `"x*y*z"` | (1,1,1) | ✅ |
| `"x^4*y^3"` | (4,3,0) | ✅ |
| `"y*x"` | (1,1,0) | ✅（顺序无关） |
| `"x^5"` | — | ❌ 次数 >4 |
| `"w"` | — | ❌ 非法变量 |
| `"x*x"` | — | ❌ 同变量重复 |

非法标签或同一单项式被多次指定都会在装载时抛 `ValueError`。

## 5. CLI 使用

```bash
python main.py --N <N> --time <T> --poly-potential <poly.json> [--g <γ>] [其它通用参数]
```

- `--poly-potential <path>`：多项式系数势 JSON 路径。**仅传文件名**（如 `example.json`）时自动在 `configs/poly_potential/` 下查找；带子路径则相对仓库根；也可用绝对路径；
- 与 `--csv`、`--trap-freq` **互斥**（argparse 强制，三选一）；
- `--g`：耗散强度 γ，与其它模式一致（默认 `0.1`）；设 `--g 0` 关闭阻尼；
- 其余参数（`--device`、`--calc-method`、`--plot`、`--init_file` 等）照常使用。

## 6. 三种力场来源对比

| 来源 | 触发 | 势场来源 | 力的计算 | RF 微运动 |
|------|------|----------|----------|-----------|
| 网格插值（默认） | `--csv` | COMSOL 网格 | `np.gradient` + 线性插值 | ✅ 分辨（依 voltage_list 时变） |
| 多项式拟合 | `--csv --poly-fit` | COMSOL 网格 → 拟合 | 解析梯度（仍从网格拟合系数） | ✅ 分辨 |
| 理想谐振势 | `--trap-freq` | 由三轴阱频解析构造 | `F=-kr`（固定二次） | ❌ 时不变 |
| **多项式系数势** | **`--poly-potential`** | **用户 JSON 显式系数** | **解析梯度（任意阶多项式）** | ❌ 时不变 |

> `--poly-potential` 本质是 `--trap-freq` 的**任意阶多项式推广**：两者都给出时不变解析力，区别在于后者固定为 `ax²+ay²+az²`，前者允许任意单项式组合（含交叉项与高阶项）。

## 7. Python API

### 7.1 高层：从 JSON 加载并构建力

```python
from pathlib import Path
import numpy as np
from equilibrium.potential_fit_3d import load_poly_potential_json
from FieldParser.force import build_poly_potential_force

# fit: FitResult3D（coeffs 为 (5,5,5) 张量，含 center_um / scale_um）
fit = load_poly_potential_json("my_potential.json")

# cfg: FieldConfiguration.constants.Config（提供 dl, dV）；charge 单位 e
force = build_poly_potential_force(fit, cfg, charge=np.array([1.0]), gamma=0.1)

F = force(r, v, t)   # r,v: (N,3) 无量纲；F: (N,3) 外力（不含库仑，由内核叠加）
```

### 7.2 直接由系数字典构造（不经 JSON）

```python
from equilibrium.potential_fit_3d import fit_result_from_coeff_map

fit = fit_result_from_coeff_map(
    {"x^2": 0.5, "y^2": 4.6, "z^2": 0.02, "x^2*y^2*z^2": 0.3},
    center_um=(0.0, 0.0, 0.0),
    scale_um=100.0,
)
```

### 7.3 接口一览

| 函数 | 位置 | 作用 |
|------|------|------|
| `parse_term_label(label) -> (i,j,k)` | `equilibrium/potential_fit_3d.py` | 单项式标签 → 指数（`quartic_3d_term_label` 的反函数） |
| `fit_result_from_coeff_map(coeff_map, center_um, scale_um, potential_offset_V) -> FitResult3D` | 同上 | `{标签: V}` → 拟合结果对象 |
| `load_poly_potential_json(path) -> FitResult3D` | 同上 | 读 JSON（兼容导出格式）→ 拟合结果对象 |
| `build_poly_potential_force(fit, cfg, charge, gamma) -> force` | `FieldParser/force.py` | 拟合结果 → 动力学力 callable |
| `FieldSettings.poly_potential` | `FieldConfiguration/field_settings.py` | JSON 路径字段（与 `csv_filename`/`trap_freq_MHz` 互斥） |

> 与其它力场一样，`build_poly_potential_force` 使用**模块级状态**（供 `fork` 子进程继承），因此单进程内不要同时实例化多个多项式势力（后设置的会覆盖前者）——通过 `main.py` / CLI 使用时无此问题。

### 7.4 势场可视化（field_visualize）

`--poly-potential` 也可直接用于 `field_visualize` 做 1D/2D 可视化与阱频/对称性分析，与 CSV 路径完全一致：

```bash
# 1D 势曲线
python -m field_visualize --poly-potential 1000_poly_sym.json --vary z --x-range -150,150 --n-pts 500
# 2D 热力图（xoy 平面）
python -m field_visualize --poly-potential 1000_poly_sym.json --vary x,y --x-range -80,80 --y-range -80,80
# 阱频（poly 二次系数 → 三轴频率）
python -m field_visualize --poly-potential 1000_poly_sym.json --freq --z-range -50,50
# 对称性 / Laplace 分解同理（--symmetry / --laplace）
```

实现上 `field_visualize/core.py::load_poly_field_bundle(poly_path, *, config_path=None)` 把 `FitResult3D` 包装成"单电极 DC、`V_bias=1`"的 `FieldBundle`：
- `potential_interps` 返回 `eval_fit_3d(r·dl_um)/dV`（盆地外发散值置 NaN，复用全链路 NaN 过滤）；
- `field_interps` 复用 `FieldParser/poly_force._make_field_callable`；
- `compute_potentials` 据此给出 `V_dc=V_poly`、`V_pseudo=0`、`V_total=V_poly`。**poly 指定的是总势**（DC + RF 赝势已合并、不可分离），故绘图经 `source_label` 强制走 `both_zero` 分支画 `V_total`、标 "Total potential"，而非按 `V_pseudo=0` 误判为 dc_only / 标 "Static potential"。

整条管线（`plot_1d`/`plot_2d`/`plot_bilayer`/阱频/对称性/Laplace）零改动复用。要点：
- poly 是**总势**（DC + RF 赝势合并，时不变、不分辨 RF 微运动），`--show-rf-amp` 因 poly 不分离 RF 幅度而自动跳过；
- 与 `--csv` 互斥；`--config` 可选（`dl/dV` 在 µm↔归一化往返中抵消，不提供时用合成默认频率 35.28 MHz，不影响结果）；
- `eval_fit_3d` 返回平移势 `V_shifted = V_true − V_min_ref`（不加回 `potential_offset_V`）；对可视化无影响（形状/曲率/对称性与零点平移无关）；
- 采样范围超出 `scale_um` 时 CLI 打印警告——多项式外推不可信（见 §9）；
- `--freq` 的阱频用硬编码 Ba135 质量（`field_visualize` 模块预存局限，非 poly 特有）。

## 8. 系数从哪来

两条路径，**二者产出的系数逐字段、逐量纲一致**——`--poly-potential` 读取的 JSON 与 `equilibrium` 拟合导出的 JSON 是同一种文件。

1. **手写**：按 [§3](#3-json-文件格式) 直接编辑 JSON，填入所需单项式系数（V）。
2. **从 config 拟合导出（往返，复现真实阱时推荐）**：`equilibrium` 模块对 CSV+config 的**总势**（DC 静电势 + RF 赝势的时间平均等效势）做 3D 多项式拟合，导出**同格式** JSON。

### 8.1 为什么 config 拟合系数与 polypotential 格式一致

二者使用**完全相同**的势场模型：

$$V(\mathbf{r}) = \sum_{ijk} c_{ijk}\, u^i v^j w^k,\qquad u=\frac{x-x_0}{L},\ v=\frac{y-y_0}{L},\ w=\frac{z-z_0}{L}$$

- `fit_potential_3d_quartic`（`equilibrium/potential_fit_3d.py`）的拟合就是在上述缩放坐标 `u,v,w` 上做的最小二乘，系数 `c_ijk` 量纲 = **V**；
- `write_potential_fit_coeff_json` 把拟合的 `center_um`、`scale_um`（即 `L`，由拟合范围自动算出）、`potential_offset_V` 与 `coefficients`（term-label→V）**原样写入** JSON；
- `load_poly_potential_json` 读回同样的字段、按同一模型重建 `FitResult3D`。

因此拟合导出的系数**无需任何换算**即可作为 `--poly-potential` 输入。已实测验证（default config + monolithic20241118.csv，`--fit-mode 4`）：

- 拟合 `R² = 0.99987`，`scale_um = 150.0`（z 轴半跨度主导）；
- 导出→读回的系数张量逐元素一致（容差 1e-12）；
- 读回后的解析梯度与原 CSV 网格势的数值梯度在拟合域内吻合至百分之几量级（四阶截断误差，与 R² 一致）。

### 8.2 命令（用现成工具，无需新代码）

生成系数文件**不需要新功能**——仓库已有两个入口都调用同一对函数（`fit_potential_3d_quartic` + `write_potential_fit_coeff_json`）导出**同格式** JSON：

**推荐：`equilibrium.fit_potential`（轻量，专为"拟合→导出系数"而建，不做平衡求解）**

```bash
# 拟合 config 总势并导出系数到 equilibrium/results/potential_fit_coeff.json
python -m equilibrium.fit_potential --csv <csv> --config <json> \
    --fit-mode 4 --n-pts 100 \
    [--symmetry-axes x,z] \
    [--x-range -50,50 --y-range -20,20 --z-range -100,100 --center 0,0,0] \
    [--plot-fit-report]   # 可选：1D/2D/残差/梯度误差可视化报告
```

**备选：`equilibrium.find_equilibrium`（重量级，顺带求解平衡构型/声子）**

```bash
python -m equilibrium.find_equilibrium --csv <csv> --config <json> \
    --fit-mode 4 [--symmetry-axes x,z] [--fit-n-pts-x 100 --fit-n-pts-y 40 --fit-n-pts-z 300]
```

两者写出的文件**逐字段一致**（均含 `center_um`/`scale_um`/`potential_offset_V`/`coefficients`），可直接喂动力学：

```bash
python main.py --N 50 --time 10 \
    --poly-potential equilibrium/results/potential_fit_coeff.json
```

> **无需在 `field_visualize` 里再加导出**。`field_visualize` 的多项式相关功能分两类：(1) `--freq`/`--fit` 是**逐轴 1D** 拟合（提取阱频/绘曲线），不含交叉项，无法直接拼出 3D 系数；(2) `--symmetry p`（多项式奇偶性）**已经**调用同一个 `fit_potential_3d_quartic` 做完整 3D 拟合，只是结果用于对称性指标、不落盘。因此 3D 拟合 + 系数导出在仓库中**只实现一次**（`equilibrium/potential_fit_3d.py` 的两个函数），三个入口（`fit_potential` / `find_equilibrium` / `field_visualize.symmetry`）共用——要拿系数文件直接用上面任一入口即可，避免重复造轮子。

> **字段有效性**：导出文件中的 `csv`/`config`/`fit_mode` 字段对 `--poly-potential` 无意义（被忽略）；`coefficients` + `center_um` + `scale_um` 才是有效输入。`potential_offset_V` 与常数项 `"1"` 只平移势能零点、梯度为零，**不影响力**。

> **拟合基底**：`fit_mode` 为非负整数 N（总次数 i+j+k≤N，项数 C(N+3,3)，如 4→35、6→84）；`symmetry_axes`（x/y/z 子集）可在拟合前剔除对称轴奇次项、强制关于 center_um 镜面对称。注意 `--poly-potential` 的显式系数读取（`fit_result_from_coeff_map`）目前限定每变量次数 ≤4（(5,5,5) 张量），故 JSON 系数文件往返仅对 N≤4 完整成立；N>4 的拟合可直接用于动力学（`--poly-fit`），不受此限。

> **`scale_um` 随工具/范围不同**：`fit_potential` 默认 z 范围 ±100 → `scale_um=100`；`find_equilibrium` 默认 z 范围 ±150 → `scale_um=150`。两者各自正确（`scale_um` 由拟合范围算出并写入），只要用同一文件配套使用即一致。

### 8.3 ⚠️ 旧版导出文件的陷阱

**2026-07-10 之前**生成的 `potential_fit_coeff.json` **不含** `center_um`/`scale_um`（旧格式）。这类文件若直接交给 `--poly-potential`，会被静默回退到默认 `scale_um=100`——而真实拟合的 `scale_um` 往往是 150（甚至更大），**力幅度因此系统性错误**。

- 自 2026-07-10 起，`load_poly_potential_json` 对缺失 `center_um`/`scale_um` 的文件会发出 `UserWarning`，提示用最新版 `find_equilibrium` 重新导出。
- 该导出文件是 gitignored 的派生产物（`equilibrium/results/`）。遇到旧文件最稳妥的做法是删除后重跑 §8.2 第 1 步重新生成。

> 该路径给出的是**时不变**总势（DC + RF 赝势平均），不分辨 RF 微运动——这与 `--poly-potential` 本身的时不变特性一致（见 [§9](#9-物理含义与局限)）。如需 RF 时变分辨，仍须用 `--csv` 或 `--csv --poly-fit`。

## 9. 物理含义与局限

- **时不变力，不分辨 RF 微运动**：单个多项式势 `V(r)` 给出 `F=-q∇V`，是静态势（含 RF 赝势的等效平均意义上的）。如需分辨 RF 快运动，须用 `--csv`/`--poly-fit`（依 `voltage_list` 的 `cos(ωt)` 时变组合）。
- **外区域发散风险**：高阶多项式在 `center ± scale_um` 之外会快速发散。当前**不做越界裁剪**（与谐振势一致），离子若飞出有效区可能产生极大/NaN 力。请按预期运动幅度合理设置 `scale_um`，必要时收窄初始条件。
- **不含库仑软化相关项**：离子-离子库仑力仍由 C++ 内核按既有逻辑叠加，与势场来源无关。
- **耗散默认开启**：γ 默认 `0.1`（与 `--csv` 一致），用 `--g` 调整或置 0。

## 10. 完整示例

仓库内置一个可运行的示例：[`configs/poly_potential/example.json`](../configs/poly_potential/example.json)（Ba-138 谐振阱 + 小 `z^4` 非谐项，阱频 2.0/2.5/0.5 MHz，系数由 `make_ideal_trap_fit` 生成并验证）：

```bash
python main.py --N 50 --time 10 \
    --poly-potential configs/poly_potential/example.json --plot
```

下面演示手写一个弱非谐势（二次 + 四阶修正）：

`my_potential.json`：

```json
{
  "coefficients": {
    "x^2": 0.5,
    "y^2": 4.6,
    "z^2": 0.02,
    "x^4": 0.001,
    "y^4": -0.05
  },
  "center_um": [0.0, 0.0, 0.0],
  "scale_um": 100.0
}
```

```bash
python main.py --N 50 --time 10 --poly-potential my_potential.json \
    --device cpu --calc-method VV --plot
```

纯 Python 接口（加载 JSON → 构建力 callable）见 [§7.1](#71-高层从-json-加载并构建力)；若要复用主程序的参数解析与 `run()` 编排，可用 `Interface.cli.parse_and_build` 组装 `ParsedRun`（其 `field_settings` 设为含 `poly_potential` 的 `FieldSettings`）再调 `main.run`。


---

**测试**：`tests/test_poly_coeff_force.py` 覆盖 term-label 解析往返、JSON 装载、导出-读回梯度一致性、缺失 `scale_um` 的告警、二次势解析力对照（含多离子/阻尼/电荷缩放）。config 拟合→导出→读回的端到端一致性（系数张量逐元素一致、域内梯度对齐）已用 default config + monolithic20241118.csv 实测验证。
