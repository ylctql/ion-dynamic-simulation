# 势场谐性指标（quadratic-only R²）

本文档描述 `field_visualize` 模块 `--freq` 输出中附加的**势场谐性指标**：每轴纯二次模型 R²。

> **相关模块**：该指标由 `compute_trap_freqs_at_point()` 在计算阱频时一并返回，与阱频共享同一 `V_total`（DC + RF 赝势）拟合管线，无需额外 CLI 开关。

## 动机

阱频 `f_x/f_y/f_z` 来自对总势 V_total 做多项式拟合后提取的阱底曲率 `k2`，是一个**局部（无穷小）量**——只刻画阱中心原点附近一个小邻域。仅看三个阱频，用户无法判断：

- 在多大空间范围内，这个"谐振近似"（抛物势）仍然成立？
- 越过该范围后，非谐成分增长有多快？

谐性指标正是为了回答这两个问题：它衡量势场在**整个扫描范围**内被一个纯抛物线描述得多好，从而把"阱频的有效适用范围"显式化。

## 定义

对每个轴 `axis ∈ {x, y, z}`，在 `--x-range/--y-range/--z-range` 给定的扫描范围内，沿该轴对 `V_total` 做**纯二次（degree=2）**最小二乘拟合：

```
V(x) = a + b·x + c·x²
```

并报告该拟合的决定系数 R²（`fit_potential_1d` 返回值）。该 R² 与 `--freq-fit-degree` **相互独立**——即使阱频提取用了 4 阶拟合（仅为更准地定位中心），谐性 R² 始终用 degree=2，从而把"势场有多谐"与"阱频怎么提"解耦。

实现：`field_visualize/trap_freq.py::quadratic_fit_r2(coord_um, V_total)`；在 `compute_trap_freqs_at_point` 的逐轴循环中以键 `r2_quad_{axis}` 返回（加键，不改动既有 `f_{axis}` 键）。

## 解读

| R² | 含义 |
|----|------|
| ≈ 1.000 | 扫描范围内势场近乎理想抛物势，阱频可描述**整个区域**而非仅原点附近 |
| 偏低（如 0.95） | 非谐成分在范围边缘显著增长，阱频仅在中心附近有效 |

**范围相关**：R² 是针对所选扫描范围定义的。范围越宽（边缘越深入非谐区），R² 越低；把范围收窄到阱中心附近，R² 会回升趋近 1。这本身是有用信息——对比"宽范围"与"窄范围"的 R² 差值，可估计非谐项的强度。典型用法：先在预期的离子运动幅度量级上设定 `--x-range` 等，再看该范围内 R² 是否仍接近 1。

> 注意：R² 不直接等于某一阶非谐常数。它是一个**范围积分**指标，综合了四阶、六阶及更高阶非谐项在该范围内的总贡献。若需分离各阶非谐常数，参见 `trap_stability` 模块的 `anh4/anh6`（Taylor 系数无量纲化）；若需 DC/RF 分量的多极分解，参见 `--laplace`。

## CLI 用法

```bash
python -m field_visualize --csv <csv> --config <json> --freq \
    [--x-range=-100,100] [--y-range=-20,20] [--z-range=-150,150] \
    [--freq-fit-degree 2|4] [--freq-n-pts 200]
```

输出示例（节选）：

```
============================================================
Trap frequencies at (x=0.0, y=0.0, z=0.0) μm
Fit ranges: x[-100,100] y[-20,20] z[-150,150] μm  (freq fit degree: 2)
============================================================
  f_x: 1.1287 MHz
  f_y: 2.0371 MHz
  f_z: 0.1393 MHz

--- Harmonicity: quadratic-only R² over scan range (1.000 = pure harmonic) ---
  x: R² = 0.95403
  y: R² = 0.99998
  z: R² = 0.99880
```

上例中 x 轴在 ±100 μm 范围 R² 仅 0.954（非谐明显），而 y/z 轴在该范围仍近乎纯谐。将 `--x-range` 收窄到 ±20 μm 时 x 轴 R² 升至 ≈0.9997——说明 x 方向阱频只在中心 ±20 μm 量级内严格有效。

## Python API

```python
from field_visualize.trap_freq import compute_trap_freqs_at_point, quadratic_fit_r2

freqs = compute_trap_freqs_at_point(
    potential_interps, field_interps, voltage_list, cfg,
    xc_um, yc_um, zc_um,
    x_range_um=(-100, 100), y_range_um=(-20, 20), z_range_um=(-150, 150),
    n_pts=200, fit_degree=2,
)
# freqs = {"f_x": ..., "f_y": ..., "f_z": ...,
#          "r2_quad_x": ..., "r2_quad_y": ..., "r2_quad_z": ...}
```

返回字典在既有 `f_x/f_y/f_z` 基础上**加键** `r2_quad_{x,y,z}`，向后兼容（`field_optimize` 等下游仅按 `f_*` 键访问，不受影响）。

## 局限

- **依赖扫描范围**：R² 的绝对数值随 `--x/y/z-range` 变化，跨配置比较时须统一范围。
- **仅一维逐轴**：沿各轴独立拟合，不捕捉离轴交叉耦合（如 `xy` 项）。交叉耦合见 `--symmetry h`（Hessian 离轴项）或 `--symmetry p`（多项式奇偶性）。
- **赝势非调和性**：`V_total` 含 RF 赝势 `V_pseudo ∝ |∇Φ|²`，本身不满足 Laplace 方程；R² 描述的是离子实际感受到的总势的谐性，这与 `--laplace`（严格适用于 DC/RF 电势）的用途不同。
