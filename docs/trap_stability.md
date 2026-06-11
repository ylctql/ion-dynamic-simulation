# trap_stability — Mathieu 稳定性参数计算

从实际离子阱场几何（CSV + JSON）计算 Mathieu a/q 参数、secular 频率、无量纲非谐常数，并判断第一稳定区。

## 物理背景

### Mathieu 方程与 a/q 参数

离子在 Paul 阱中的运动由 Mathieu 方程描述：

$$\frac{d^2u}{d\xi^2} + (a - 2q\cos 2\xi)\,u = 0, \qquad \xi = \frac{\Omega t}{2}$$

其中 $\Omega$ 为 RF 驱动角频率。参数 $a$ 来源于 DC 势，$q$ 来源于 RF 势的振幅。

本模块从**实际场几何**提取这些参数：沿各轴对 DC 势和 RF 幅值势做多项式拟合，从二次系数 $c_2$（Taylor 系数，含 $1/2!$ 因子）计算：

$$a = \frac{8e\,c_{2,\text{DC}}}{m\Omega^2}, \qquad q = \frac{4e\,c_{2,\text{RF}}}{m\Omega^2}$$

其中 $c_2 = \Phi''(x_0)/2!$ 为势场在陷阱中心处的 Taylor 二次系数，单位 V/μm²。

### Secular 频率

绝热近似下，各轴 secular 频率：

$$f_{\text{sec}} = \frac{\Omega}{4\pi}\sqrt{a + \frac{q^2}{2}}$$

模块同时计算**总势阱频**（直接拟合 DC + 赝势的总势曲率），通常更接近实验值。

### 非谐常数

对势场做 6 阶 Taylor 展开：

$$\Phi(x) = c_2 x^2 + c_4 x^4 + c_6 x^6 + \cdots, \qquad c_{2k} = \frac{\Phi^{(2k)}(x_0)}{(2k)!}$$

使用与动力学模拟相同的无量纲化（$dV$ 为特征电压，$dl$ 为特征长度），定义无量纲非谐常数：

$$\tilde{c}_{2k} = \frac{c_{2k} \cdot dl_{\mu m}^{2k}}{dV}$$

其中 $dl_{\mu m} = dl \times 10^6$（将米转换为 μm）。这些常数即为无量纲势场展开的系数。

> **与标准 Mathieu 参数的关系**：由 $dV = m\Omega^2 dl^2/(4e)$ 可证 $a = 2\tilde{c}_{2,\text{DC}}$，$q = \tilde{c}_{2,\text{RF}}$。2 阶无量纲系数与标准 Mathieu 参数等价。4 阶和 6 阶系数表征阱的非谐性，出现在非线性方程（Duffing–Mathieu 型）中，不决定线性稳定性边界。

## 快速使用

```bash
# 基本用法（自动检测陷阱中心）
python -m trap_stability --csv monolithic20241118.csv --config default.json

# 仅传文件名时自动在默认目录查找
python -m trap_stability --csv default.csv --config default.json

# 指定陷阱中心和离子种类
python -m trap_stability --csv default.csv --config default.json \
    --center 0,0,0 --species Ca40+

# 输出 JSON
python -m trap_stability --csv default.csv --config default.json --out result.json
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv` | （必填） | 电场 CSV 路径；仅文件名时在 `data/` 下查找 |
| `--config` | `""` | 电压 JSON 路径；仅文件名时在 `FieldConfiguration/configs/` 下查找 |
| `--center` | 自动检测 | 陷阱中心坐标 `X,Y,Z` (μm) |
| `--x-range` | `-50,50` | x 轴拟合范围 (μm) |
| `--y-range` | `-20,20` | y 轴拟合范围 (μm) |
| `--z-range` | `-150,150` | z 轴拟合范围 (μm) |
| `--n-fit-pts` | `200` | 每轴采样点数 |
| `--fit-degree` | `6` | 多项式拟合最高阶数：`2`（仅 a/q）、`4`（+anh4）、`6`（+anh4/anh6） |
| `--smooth-axes` | `z` | 势场平滑方向；`none` 关闭 |
| `--smooth-sg` | `11,3` | Savitzky-Golay 窗口长度,阶数 |
| `--species` | `Ba135+` | 离子种类 |
| `--out` | （可选） | JSON 输出路径 |

## 输出示例

```
============================================================
  Ion Trap Stability Analysis
============================================================
  Species:       Ba135+ (134.906 amu)
  RF frequency:  35.28 MHz (Omega = 2.216e+08 rad/s)

  Mathieu Parameters:
     axis       a              q
     x     -4.123456e-03   1.234567e-01
     y     -4.123456e-03   1.234567e-01
     z      8.246912e-03   0.000000e+00

  Secular Frequencies (adiabatic approximation):
     f_x =   0.8234 MHz    f_y =   0.8234 MHz    f_z =   0.1234 MHz

  Trap Frequencies (total potential curvature):
     f_x =   0.8250 MHz    f_y =   0.8250 MHz    f_z =   0.1235 MHz

  Anharmonic Constants (dimensionless, Taylor c_{2k}·dl^{2k}/dV):
       axis       anh4_dc         anh4_rf         anh6_dc         anh6_rf
       x     -1.234567e-05    2.345678e-06   -3.456789e-08    4.567890e-09
       y     -1.234567e-05    2.345678e-06   -3.456789e-08    4.567890e-09
       z      1.234567e-04    0.000000e+00    2.345678e-07    0.000000e+00

  Stability: [STABLE] (stable in first stability region)
============================================================
```

## JSON 输出格式

```json
{
  "species": "Ba135+",
  "mass_amu": 134.905683,
  "omega_rf_rad_s": 2.216e+08,
  "freq_rf_MHz": 35.28,
  "a": {"x": -0.00412, "y": -0.00412, "z": 0.00825},
  "q": {"x": 0.1235, "y": 0.1235, "z": 0.0},
  "f_secular_MHz": {"x": 0.8234, "y": 0.8234, "z": 0.1234},
  "f_trap_MHz": {"x": 0.8250, "y": 0.8250, "z": 0.1235},
  "k2_dc_V_per_um2": {"x": -1.2e-04, "y": -1.2e-04, "z": 2.4e-04},
  "k2_rf_amp_V_per_um2": {"x": 3.5e-03, "y": 3.5e-03, "z": 0.0},
  "anh4_dc": {"x": -1.2e-05, "y": -1.2e-05, "z": 1.2e-04},
  "anh4_rf": {"x": 2.3e-06, "y": 2.3e-06, "z": 0.0},
  "anh6_dc": {"x": -3.5e-08, "y": -3.5e-08, "z": 2.3e-07},
  "anh6_rf": {"x": 4.6e-09, "y": 4.6e-09, "z": 0.0},
  "is_stable": true,
  "stability_note": "stable in first stability region",
  "csv": "/path/to/data.csv",
  "config": "/path/to/config.json",
  "center_um": [0.0, 0.0, 0.0],
  "n_fit_pts": 200
}
```

## 模块结构

| 文件 | 内容 |
|------|------|
| `stability.py` | `StabilityResult` 数据类；`compute_stability_from_field()` 核心计算；`find_trap_center()` 自动中心检测；`check_stability_region()` 第一稳定区判断 |
| `cli.py` | CLI 入口；`create_parser()`、报告输出、JSON 序列化 |
| `__init__.py` | 模块公开接口 |

## 算法流程

```
CSV + JSON
  ├─ init_from_config()     → Config (dl, dt, dV, Omega)
  ├─ read_csv() + smooth    → grid_coord, grid_voltage
  ├─ calc_potential/field()  → 插值器列表
  ├─ build_voltage_list()    → Voltage 列表
  ├─ find_trap_center()      → (xc, yc, zc)  [可选自动检测]
  │     ├─ 粗 3D 网格搜索 V_total 最小值
  │     └─ 各轴 1D 二次拟合精化
  └─ compute_stability_from_field(fit_degree=2|4|6)
        ├─ 各轴采样 DC/RF/总势
        ├─ fit_degree 阶多项式拟合 → c₂ [c₄] [c₆] (Taylor 系数)
        ├─ a = 8e·c₂_DC/(m·Ω²)
        ├─ q = 4e·c₂_RF/(m·Ω²)
        ├─ f_sec = Ω/(4π)·√(a + q²/2)
        ├─ f_trap = √(2e·k₂_total/m)/(2π)
        ├─ anh4 = c₄·dl⁴/dV  (无量纲)
        └─ anh6 = c₆·dl⁶/dV  (无量纲)
```

## 依赖关系

- `FieldConfiguration/` — Config、ion_species、voltage_list
- `FieldParser/` — CSV 读取、势场插值、多项式拟合（`potential_fit.py`）
- `field_visualize/core.py` — `compute_potentials()`、`build_grid_1d()`
- `scipy.constants` — 物理常数
- `numpy.linalg` — 多项式拟合

## 测试

```bash
pytest tests/test_trap_stability.py -v
```

测试覆盖：稳定性判据、secular 频率公式、CLI 解析、a/q 教科书公式验证、合成势非谐常数、实际场积分测试。
