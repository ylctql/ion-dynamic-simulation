# radial_trap 用户教程

> 适用于 `radial_trap` 模块 — 2D 径向多项式势阱：晶格条件 + 平衡构型 + 解析 micromotion + 面内声子
> + 势场分布图 + 阱频反算 + 格点 CSV 拟合 + 交互 UI。
> 无需电场 CSV、无需 C++ 构建（纯 numpy/scipy）。系数推导、输出格式细节与 v1 局限见技术参考
> [radial_trap.md](radial_trap.md)；本文只讲"怎么用"。

## 目录

1. [模块解决什么问题](#1-模块解决什么问题)
2. [快速上手](#2-快速上手)
3. [五个势系数怎么选](#3-五个势系数怎么选)
4. [读懂控制台输出](#4-读懂控制台输出)
5. [常见任务](#5-常见任务)
6. [作为 Python 库使用](#6-作为-python-库使用)
7. [FAQ](#7-faq)

---

## 1. 模块解决什么问题

给定五个解析势系数 (A, B, D, E, F) + RF 频率 + 离子物种 + 离子数，模块在径向平面 (x, y)
内直接构造总势（RF 赝势 + RF bias + DC），一步给出：

- **4 条晶格成形条件**逐条判定（设计文档口径的 PASS/FAIL 报告）
- **平衡构型**（L-BFGS-B 总能量最小化，含库仑相互作用；默认链沿 x 轴）
- 各离子**解析 micromotion** 幅度 a_mm 与过剩比 ρ——不需要跑任何轨迹
- 可选**面内声子**谱（`--phonon`）、**势场分布图**（`--plot-potential`）
- **阱频反算**（`--from-freq`：目标 f_x/f_y → A,E）、**格点 CSV 拟合 A/B**
  （`--fit-ab`，§5.9）与**滑块交互 UI**（`--ui`）

与相邻模块的关系：

| 想做什么 | 用哪个 |
|---|---|
| 从实测电场 CSV 出发求平衡/声子 | `equilibrium` |
| 实测场的阱频/对称性/拉普拉斯分解诊断 | `field_visualize` |
| **自己设计一个径向势场，立刻看晶格长什么样** | `radial_trap`（本模块） |
| 把设计出的势交给动力学跑轨迹 | `radial_trap --export-poly --export-fz` → `main.py --poly-potential` |

设计动机来自笔记《关于x方向非谐势的讨论.md》：考察 RF 四极+十六极基合成的 2D 势在什么条件下
能成形理想晶格、高阶 RF 场带来多少过剩 micromotion。

---

## 2. 快速上手

```bash
python -m radial_trap --N 2 --phonon --plot
```

完整输出（默认参数 A=5e-3, E=−3.5e-4, f_RF=35.28 MHz, Ba135+，链沿 x）：

```text
=== 径向势阱参数 ===
  φ_rf: A=0.005 V/µm², B=0 V/µm⁴   RF bias: D=0
  φ_DC: E=-0.00035 V/µm², F=0 V/µm⁴
  f_RF=35.28 MHz, 物种=Ba135+, N=2, seed=0
=== 晶格条件（设计文档 4 条） ===
  (1) 压制交叉项  DB+F=0.000e+00, |B|=0.000e+00  →  PASS
  (2) 压制 y 高次  −16αAB+DB+F=0.000e+00  →  PASS
  (3) y 束缚  c_y2=7.138766e-04 V/µm²  →  PASS  (f_y=5.086 MHz)
  (4) x 束缚  c_x2=1.387664e-05 V/µm²  →  PASS  (f_x=0.7091 MHz)
  α_eff=3.63877 µm²/V, q_x=0.2911, q_y=-0.2911
  四次项相对重要性 eps4_x=0, eps4_y=0 (L_ref=10 µm)
=== 平衡构型 ===
  converged=True, nit=16, |grad|=2.27e-11 eV/µm, E_total=4.596429e-04 eV
  x ∈ [-2.350, 2.350] µm, y ∈ [0.000, 0.000] µm
=== 解析 micromotion（一阶 RF 谐波峰值） ===
  |a_mm|: max=0.3420 µm, mean=0.3420 µm
  excess 比 ρ = a_mm/((q/2)·r): [1.0000, 1.0000]  (四极极限=1)
=== 面内声子（4 模，频率降序，负值=不稳定） ===
  [ 0]     5.0858 MHz
  [ 1]     5.0362 MHz
  [ 2]     1.2282 MHz
  [ 3]     0.7091 MHz
npz 输出: /home/ylcyyds/ism-space/ism-main/radial_trap/results/radial_N2.npz
图输出: /home/ylcyyds/ism-space/ism-main/radial_trap/results/radial_N2.png
```

三行速读：4 条件全 PASS → 这是一个合格的设计点；两离子沿弱轴 x 排成链
（间距 4.70 µm，y=0）；ρ=[1,1] → 纯四极 RF 场没有过剩 micromotion（见 §4）。

产物默认写到 `radial_trap/results/`（npz 数据 + PNG 双联图，目录已 gitignore）。
图内文本为英文（原因见 FAQ），左图是 xoy 构型 + 每离子 micromotion 峰峰值线段，
右图是 |a_mm| 与 ρ 随离子序号的汇总。

---

## 3. 五个势系数怎么选

| 系数 | 单位 | 默认 | 物理作用 | 调大的效果 |
|---|---|---|---|---|
| `--A` | V/µm² | 5e-3 | RF 四极（决定 Mathieu q 与两轴阱频） | q、f_x、f_y 同步升高（q_x=0.291 @ 默认） |
| `--B` | V/µm⁴ | 0 | RF 十六极（高阶赝势） | 引入过剩 micromotion（ρ 偏离 1），破坏条件 (1)(2) |
| `--D` | — | 0 | RF 电极直流偏置（D·φ_rf） | DA 项重新分配 c_x2/c_y2；过强直接压垮 y 囚禁 |
| `--E` | V/µm² | -3.5e-4 | DC 四极（各向异性，符号定链轴） | E<0（默认）→ x 弱轴、链沿 x；变号则链轴翻转；\|E\| 过大弱轴失束缚 |
| `--F` | V/µm⁴ | 0 | DC 十六极 | 常用于精确抵消 DB（取 F = −D·B，见 §5.2） |

关键派生量（决定一切的其实是两个二次系数）：

```text
c_x2 = 4αA² + DA + E        （E<0 时 x = 弱轴 = 链轴，默认）
c_y2 = 4αA² − DA − E        （y = 紧/净轴）
f_a  = √(2Q·c_a2/m)/2π ，   q_x = 4QA·1e12/(mΩ²) （与 E 无关）
```

默认参数下 4αA² = 3.6388e-4 V/µm²——这是 E 与 DA 可用度的"预算"：
4αA²+DA+E → 0 时 x 失束缚（条件 (4) FAIL，f_x 变 nan），4αA²−DA−E → 0 时
y 失束缚（条件 (3) FAIL）。默认 E=−3.5e-4 刻意贴近 x 侧下缘——f_x=0.71 MHz
是设计想要的弱轴（对应 3D 晶格的轴向，见 §5.1），|E| 再大就撑破了。

推荐选参流程：

1. **定 A**：由目标 Mathieu q 反推（默认 A=5e-3 → q≈0.29，常规工作点）。
2. **定 E**：直接给目标阱频最省事——`--from-freq FX FY`（§5.6，默认
   f_x=0.71 / f_y=5.09 MHz，紧/弱比 ≈ 7.2 深处线性区）；手调 E 时注意
   线性链要求 (f_紧/f_弱)² > 7ζ(3)/2 ≈ 4.21（见 §5.1）。
3. **按需加 B/D/F**：每加一项就对照 4 条件检查（尤其 DB+F 与 DA 对弱轴系数的侵蚀），
   并看 eps4 是否还在"二次项主导"范围（eps4 = |s₄|·L_ref²/c₂，L_ref=10 µm）。

---

## 4. 读懂控制台输出

对照 §2 的输出块，自上而下五节：

**晶格条件（4 条）**

| 编号 | 判据 | 不满足的后果 |
|---|---|---|
| (1) | DB+F→0 **且** B→0 | x²y² 交叉项使链变形 |
| (2) | −16αAB+DB+F→0 **且** B→0 | y⁴ 高次项显著 |
| (3) | c_y2 > 0 | y 无二次囚禁（f_y=nan，离子可能飞向边界） |
| (4) | c_x2 > 0 | x 无二次囚禁 |

(1)(2) 的趋零容差由 `--cond-tol`（默认 1e-12）控制；FAIL 只打 WARNING，**仍继续求解**
（高次项可能仍束缚，物理上对应双阱/zigzag 一类构型，见 FAQ）。

**α_eff / q 行**：α_eff 是赝势系数（用**物种自身质量**），q 是 Mathieu 稳定性参数——
全仓约定 RF 幅度为零到峰，q≈0.29 属常规工作点。

**平衡构型节**：`converged` 是否收敛、`|grad|` 收敛精度；`x ∈ / y ∈` 给出构型跨度。
若出现 `⚠ 有离子贴近 x/y 边界`（hit_bounds），说明非囚禁或 `--x-range/--y-range` 太小。

**解析 micromotion 节**：|a_mm| 为一阶 RF 谐波**峰值**（µm）。ρ 是过剩比——
纯四极场（B=0）恒等于 1（恒等式 a_mm = (q/2)·r）；B≠0 时 ρ 随 |r| 偏离 1，
即高阶 RF 场导致的**过剩 micromotion**，这正是设计文档关心的量。

**面内声子节**（`--phonon`）：2N 个面内模按频率降序；**负值 = 不稳定模**（构型不是
局部极小）。N=2 的 4 个模有解析对照：{f_t, √(f_t²−f_a²), √3·f_a, f_a}
（a = 链轴/弱轴，t = 横向紧轴；默认链沿 x，如 §2 的 5.086/5.036/1.228/0.709）。

npz / JSON 报告的键名与格式见技术参考 [radial_trap.md](radial_trap.md) §5。

---

## 5. 常见任务

### 5.1 线性链 / zigzag / 链轴翻转

默认参数（E=−3.5e-4）的 N=10 基态就是 **x 轴线性链**——x 是刻意取的弱轴
（f_x=0.71 MHz，对应 3D 晶格的轴向：理想晶格全部分布在 xoz 平面，径向模块
里表现为离子集中在 x 轴上），紧/弱轴频率比 f_y/f_x ≈ 7.2 远超无限链 zigzag
阈值 (f_紧/f_弱)² > 7ζ(3)/2 ≈ 4.21（即 f_紧/f_弱 > 2.05）：

```bash
python -m radial_trap --N 10
```

```text
=== 平衡构型 ===
  converged=True, nit=47, |grad|=3.09e-10 eV/µm, E_total=1.802653e-02 eV
  x ∈ [-10.707, 10.707] µm, y ∈ [-0.000, 0.000] µm
=== 解析 micromotion（一阶 RF 谐波峰值） ===
  |a_mm|: max=1.5585 µm, mean=0.8205 µm
  excess 比 ρ = a_mm/((q/2)·r): [1.0000, 1.0000]  (四极极限=1)
```

判断线性链的依据：y 分量 ≈ 0、x 坐标单调等间距（两端稍松）；
更严格的判据是 `--phonon` 无负频模。

**弱化 |E| → zigzag**：各向异性降到阈值以下时，链离开 x 轴（3D 对应离开
xoz 面）弯成 zigzag——这是物理正确的基态，不是求解错误：

```bash
python -m radial_trap --E=-1e-4 --N 10
```

```text
=== 平衡构型 ===
  converged=True, nit=56, |grad|=5.20e-10 eV/µm, E_total=3.985723e-02 eV
  x ∈ [-2.833, 2.833] µm, y ∈ [-1.572, 1.572] µm
```

（f_x=3.09 / f_y=4.10 MHz，紧/弱比 1.33 < 2.05 → 基态 zigzag。）

**E 变号 → 链翻转到 y 轴**：镜像参数 E=+3.5e-4 使 c_x2 ↔ c_y2 互换、
x/y 角色对调，链翻到 y 轴（总能量与默认完全一致——严格镜像）。注意此时
弱轴是 y，默认 y ±10 µm 装不下 ±10.7 µm 的链，需放宽范围：

```bash
python -m radial_trap --E=3.5e-4 --y-range=-13,13 --N 10
```

```text
=== 平衡构型 ===
  converged=True, nit=38, |grad|=6.86e-10 eV/µm, E_total=1.802653e-02 eV
  x ∈ [-0.000, 0.000] µm, y ∈ [-10.707, 10.707] µm
```

### 5.2 检验 DB+F 精确抵消（条件 (1) 的双重性）

条件 (1) 要求 **DB+F→0 且 B→0** 两项同时成立。取 D=0.5、B=1e-7、F=−5e-8
（DB = 0.5×1e-7 = 5e-8 = −F，交叉项系数 −6(DB+F) 精确为零）：

```bash
python -m radial_trap --B 1e-7 --D 0.5 --F=-5e-8 --N 2
```

```text
  (1) 压制交叉项  DB+F=0.000e+00, |B|=1.000e-07  →  FAIL
  (2) 压制 y 高次  −16αAB+DB+F=-2.911e-08  →  FAIL
  (3) y 束缚  c_y2=-1.786123e-03 V/µm²  →  FAIL  (f_y=nan MHz)
  (4) x 束缚  c_x2=2.513877e-03 V/µm²  →  PASS  (f_x=9.544 MHz)
  ...
  x ∈ [-0.415, 0.415] µm, y ∈ [-10.000, -10.000] µm
  ⚠ 有离子贴近 x/y 边界（疑似非囚禁或范围不足）
  ...
  excess 比 ρ = a_mm/((q/2)·r): [0.9880, 0.9960]  (四极极限=1)
```

两个教训一次看清：

- DB+F=0.000e+00 **精确抵消**，但 |B|≠0 → 条件 (1) 仍 FAIL——十六极 RF 场即使交叉项
  抵消，仍通过 −16αAB 项（条件 (2)）和过剩 micromotion（ρ=[0.988, 0.996] ≠ 1，
  偏离随 |r| 增大）起作用。
- 该参数下 DA = 2.5e-3 远超 4αA² = 3.64e-4，把 c_y2 压成负值 → 条件 (3) FAIL、
  f_y=nan、离子被推到 y 边界（hit_bounds ⚠ 行）——顺带演示了"FAIL 仅警告仍求解"。

### 5.3 导出总势给 main.py 跑动力学

```bash
python -m radial_trap --N 10 --export-poly total_poly.json
python main.py --poly-potential total_poly.json --N 10 --time 5
```

导出的是总势 9 项系数（含 B≠0 时的六次项）。注意：poly 势力场**时不变**，
动力学只含 secular 运动、不含 RF micromotion（见 FAQ）。

**3D 演化务必加 `--export-fz FZ_MHZ`**：模块总势只定义在 xoy 平面、不含任何
z 项，直接导出会让 main.py 的 3D 动力学 z 方向零囚禁——库伦作用把离子沿 z
摊开成片，径向一维链在 3D 里只是鞍点（会"变两层"甚至成面）。加
`--export-fz 0.55` 即在导出势上附加简谐轴向囚禁 c_z2·z²（对应真实阱的端帽
DC），f_z 的量级要求见 §5.8 设计律 3：

```bash
python -m radial_trap --N 20 --A 5.152845e-4 --E=-3.788107e-6 \
    --export-poly total_poly_fz.json --export-fz 0.55
python main.py --poly-potential total_poly_fz.json --N 20 --time 5
```

### 5.4 参数存档与复现

```bash
python -m radial_trap --json params.json --report result.json
```

```json
{"A": 5e-3, "E": -3.5e-4, "N": 10, "x_range": [-30, 30], "y_range": [-10, 10]}
```

`--json` 是**严格键集**（A, B, D, E, F, freq_rf_mhz, species, N, x_range, y_range,
seed, softening_um），未知键直接报错。优先级：CLI 显式旗标 > `--json` > 内置默认。
`--report` 输出与 npz 同构的分节 JSON（NaN → null），适合入库存档。
UI 的 Save JSON 按钮写出的也是这一格式（§5.7）。

### 5.5 看径向平面势场长什么样

```bash
python -m radial_trap --N 10 --plot-potential
```

输出 2×2 面板 PNG（默认 `radial_trap/results/potential_N10.png`）：RF 赝势
α|E_rf|²（恒 ≥0）、RF bias D·φ_rf、DC、总势+离子散点。跨零面板自动用关于 0
对称的 RdBu_r、单符号面板用 viridis；色标做 0.5–99.5% 稳健裁剪（B≠0 时角落
r⁴ 爆涨不会撑坏色标）。默认 D=0 时 bias 面板恒零，显示 "zero everywhere"
占位。可与 `--plot` 同开（一次跑出构型图 + 势场图）。

### 5.6 由目标阱频反算 A、E

知道想要的两轴阱频时不必手调 E 试错——`--from-freq FX FY` 直接反解
（c_x2+c_y2=8αA² 与 E、D 无关，可唯一反推；代数见技术参考 §6）：

```bash
python -m radial_trap --from-freq 0.7091 5.086 --N 10
```

```text
阱频反算: 目标 f_x=0.7091 MHz, f_y=5.086 MHz  (f_RF=35.28 MHz, 物种=Ba135+, D=0)
  → A=5.000163e-03 V/µm², E=-3.500227e-04 V/µm²  (达成 q_x=0.2911)
```

恰好反算回默认 A、E（q 不变——|q| 只由 f_x²+f_y² 决定）。规则：

- species / freq_rf / D 可来自 `--json`（优先级 CLI 显式 > --json > 内置默认）
- 与显式 `--A`/`--E` 或 `--json` 里的 A/E **冲突即报错**（反算会覆盖它们，须移除其一）
- 只定二次项：B/F 不受影响，仍需按 §3 流程检查条件 (1)(2)

### 5.7 滑块交互调参（--ui）

```bash
python -m radial_trap --ui                              # 桌面环境（WSLg/X11/macOS）
python -m radial_trap --ui --from-freq 0.7091 5.086     # 以反算结果为滑块初值
```

窗口布局：左主面板总势 + 离子 + micromotion 线段，右读出（系数、f/q、
4 条件 PASS/FAIL、链跨度、|a_mm|），底部 A/B/D/E/F 五个系数行（每行
滑块 + 数值框 + 范围框）、N / x / y 区域框、DB+F=0 约束开关、f_x/f_y
目标频率框与 Solve / Save 按钮：

- **拖动滑块** → 面板/读出实时刷新（只重算势场，不重解平衡，保证跟手；
  v1.3 起重绘节流 ≤30 fps + **重解防抖**——松开鼠标/按键并静止 ~0.35 s
  才热启动重解一次，方向键自动重复、滑块连点不再逐次重解卡界面）；
  colorbar 水平放在面板下方（不占宽度），x 域拉宽时等比画面尽量少缩小
- **数值框**（每系数一行）→ 直接输入精确值，Enter 或点框外提交；超出
  滑块范围会**自动扩程**而不是拒绝。**范围框**（"rng"）→ 输入 "lo,hi"
  重建滑块量程
- **N / x µm / y µm 框**（第 6 行）→ N 改离子数（热启动增删后重解）；
  "lo,hi" 改势场绘制区域（网格与色条随之重建）
- **键盘方向键**：聚焦数值框 ↑↓ 步进（Shift ×5）；N 框 ↑↓ ±1
  （Shift ±5）；区域/范围框 ↑↓ 以中心为锚放宽/收窄（Shift 细调）；
  鼠标悬停滑块轴 ←→↑↓ 步进、Home/End 跳端点。松开按键才重解平衡
- **DB+F=0 约束开关** → 勾选后 F = −D·B 自动随 B/D 联动（F 控件隐藏，
  行内灰字显示联动值），是压平 x²y 交叉项即条件 (1) 的精确抵消手段；
  取消勾选 F 冻结在当前值，恢复独立可调
- **Solve A,E** → 按文本框目标频率反算并回写滑块（§5.6 同一代数；框内
  数值随当前阱频刷新，改掉即为目标）
- **Save JSON** → 当前参数写 `radial_trap/results/ui_params.json`
  （--json 严格键集，`--json` 读回即复现；约束开关状态不存档，F 的
  联动结果以数值保存）

无桌面环境（SSH/无头）会明确报错——UI 需要交互式 matplotlib 后端。

### 5.8 实战：N=20/30/40 离子沿 x 线性链（微 MM 版 / 频率底线版）

推荐参数组 F1（Ba135+，f_RF=35.28 MHz，纯四极 B=D=F=0，q_x=0.03、r=10）：

```bash
python -m radial_trap --A 5.152845e-4 --E=-3.788107e-6 --N 20 \
    --x-range=-100,100 --y-range=-14,14 --phonon \
    --export-poly total_poly_fz.json --export-fz 0.55
```

A=5.152845e-4 V/µm²、E=−3.788107e-6 V/µm²。达成：f_x=0.0527 /
f_y=0.5266 MHz（r=10）、4/4 条件 PASS、链严格在 y=0（跨度 182.7 µm）、
中心/边缘间距 8.11/13.92 µm、最外离子 |a_mm|=1.37 µm——仅为边缘间距的
0.10 倍、中心间距的 0.17 倍；40 个面内声子全正（最低模即 COM-x=f_x）；
3D 验证（export + f_z=0.55 MHz）：3 个随机云初值全部收敛到同一单排链
（能量 1.2299e-2 eV 相同），60 个 3D 声子全正，模块冷启动 5 个随机
seed 全部落回链。

紧凑备选 F2（q_x=0.05、r=10）：`--A 8.588075e-4 --E=-1.052252e-5
--x-range=-75,75 --y-range=-12,12 --export-fz 0.90`。跨度 130 µm 更短，
f_x=0.0878 MHz；代价是最外 |a_mm|=1.63 µm（边缘间距的 0.16 倍）。两组
均通过上述全部检查，按"要更小 micromotion（F1）还是要更短链（F2）"取舍。

**更长链 N=30/40——频率底线版（推荐）**：晶格稳定定量条件取 f_x ≥ 0.3 MHz、
f_y ≥ 1 MHz。f_y 底线自动满足（zigzag 要求 r ≥ r_min(N) ≥ 12，故
f_y = r·f_x ≥ 3.6 MHz），真正起约束的是 f_x 底线；它把 q_x 钉到
0.22/0.28、端部 MM 推到与端部间距同量级（设计律 6——和定则所致，
四次项绕不开）：

```bash
# N=30：f_x=0.30、r=13（r_min(30)≈12.0，留 8% 裕量）、q_x=0.222
python -m radial_trap --A 3.808667e-3 --E=-2.086510e-4 --N 30 \
    --x-range=-45,45 --y-range=-15,15 --phonon \
    --export-poly total_poly_fz.json --export-fz 3.9
# N=40：f_x=0.30、r=16.5（r_min(40)≈15.5，留 7% 裕量）、q_x=0.281
python -m radial_trap --A 4.828682e-3 --E=-3.368844e-4 --N 40 \
    --x-range=-51,51 --y-range=-15,15 --phonon \
    --export-poly total_poly_fz.json --export-fz 4.95
```

N=30：f_x=0.300/f_y=3.90 MHz，跨度 71.1 µm，中心/边缘间距 2.02/4.01 µm，
最外 |a_mm|=3.94 µm（边缘间距的 0.98 倍）；N=40：f_x=0.300/f_y=4.95 MHz，
跨度 82.2 µm，间距 1.71/3.78 µm，最外 |a_mm|=5.78 µm（1.53 倍）。两者
4/4 条件 PASS、链严格 y=0（|y|~5e-7 µm）、面内与 3D 声子全正（最低模
=COM-x=0.300000 MHz）、冷启动 5 seed 与 3D 云初值 3/3 全部收敛单排链
（3D 与 2D 总能量一致到 ~1e-14 eV）。端部 MM 与间距同量级是频率底线的
物理代价而非选参失误——要小 MM 见下方低频备选。

**低频长链备选**（只需链稳定、可接受 f_x ≪ 0.3 MHz、追求最小 MM 时；
r 取 ~1.25·r_min，q_x 相应下调）：

```bash
# N=30：q_x=0.02、r=15
python -m radial_trap --A 3.435230e-4 --E=-1.702415e-6 --N 30 \
    --x-range=-235,235 --y-range=-15,15 --phonon \
    --export-poly total_poly_fz.json --export-fz 0.352
# N=40：q_x=0.015、r=19
python -m radial_trap --A 2.576423e-4 --E=-9.608206e-7 --N 40 \
    --x-range=-460,460 --y-range=-15,15 --phonon \
    --export-poly total_poly_fz.json --export-fz 0.264
```

N=30：f_x=0.0235/f_y=0.352 MHz，跨度 388.7 µm，中心/边缘间距
11.03/21.90 µm，最外 |a_mm|=1.94 µm（边缘间距的 0.089 倍）；N=40：
f_x=0.0139/f_y=0.264 MHz，跨度 637.2 µm，间距 13.24/29.27 µm，最外
|a_mm|=2.39 µm（0.082 倍）。同样 4/4 PASS、面内与 3D 声子全正、冷启动
与 3D 云初值全部收敛单排链（3D 与 2D 能量一致 ~1e-15 eV）。MM 比频率
底线版小一个量级（比值 0.08–0.09 vs 0.98–1.53）的代价是阱频低一个
量级、链长 5–8 倍。**注意 r=10 在 N≥30 已不够**（直线链变鞍点，收敛到
zigzag）——横向比必须随 N 增长。

**怎么调出来的（可复用的设计律）**：

1. **先定 q_x**：纯四极下 a_mm,x=(q_x/2)·x（§4 恒等式），最外离子
   a_mm ≈ q_x·L/4；谐势链形状只由 N 决定 ⇒ 逐离子 a_mm/d ≈
   q_x(N−1)/4，**与 r 无关**。"a_mm < 间距" ⇔ q_x < 4/(N−1)（N=20 →
   0.21）。ρ≡1 表明这是纯四极本征 micromotion——此模型内只能靠降 q_x
   压小。但标度很弱：a_mm ∝ q^(1/3)·(1+r²)^(1/3)（链长同时随 f_x^(−2/3)
   变长、最外离子更远），q 0.05→0.03 只省 16%（1.63→1.37 µm），链却
   130→183 µm——两者不可兼得，F1/F2 即两端
2. **q_x 定 A、r 定 E**：A = q_x/(16α)；E = 4αA²(1−r²)/(1+r²)（等价于
   `--from-freq`；f_x ∝ q_x/√(1+r²)，选大 r 即薄径向扁势）
3. **横向比 r 和轴向比 f_z/f_x 都要过有限 N 的 zigzag 边界**：r_min 与 q
   无关（技术参考 §7），且**随 N 增长**——实测 N=20/30/40 分别 ≈8.4/
   12.0/15.5（拟合 r_min≈0.60·N^0.88；不是教科书的 2.05，也和方向
   无关——面内 y 与 3D z 一致）。常规留 ~25% 裕量（N=20 取 10）；
   频率底线版为压 q_x 只留 7–8%（N=30 取 13、N=40 取 16.5）——裕量
   收紧后须用 `--phonon` 逐点核验 zigzag 模仍为正。**这正是"两层"问题
   的根源**：导出势若不加 f_z，3D 中 z 零囚禁、链摊成片；加了但
   f_z/f_x ≲ r_min(N)，链会沿 z zigzag 成双层。f_z 取 ≳1.25·r_min·f_x
   （F1: 0.55、频率底线版 N=30: 3.9、N=40: 4.95 MHz），对应
   --export-fz 的参数
4. **检查压边界**：span 恰等于 2×x-range 的解是离子顶着求解边界（数据
   作废），放宽 x-range 重跑
5. **负 B 压端部 micromotion 不可行**：RF 十六极 B<0 确实能经 ρ 机制
   （|E_full|<|E_quad|）把链端 excess 比压到 0.5 以下、端部 |a_mm| 最深
   降到 ~0.3 µm，但同时引入 x⁴ 软化项（16αAB<0）使晶格失稳（声子谱
   出 −16~−25 MHz 负模、链解体）——稳定与降 MM 不可兼得，放弃
6. **频率底线被和定则钉死（四次项绕不开）**：c_x2+c_y2=8αA² 与
   E/D/B/F **全部无关**（§5.6 反算所用的同一条代数），故
   q_x = 2·f_x·√(1+r²)/f_RF 只由 (f_x, r) 决定。f_x≥0.3 且 r≥r_min
   ⇒ q_x ≥ 0.22/0.28（N=30/40；f_y=r·f_x≥3.6 MHz，1 MHz 底线自动
   满足），端部比 a_end/d_o ≈ 4.4q/5.4q ≈ 0.98/1.53——**A 无自由度
   降 q，B/D/F 不进和定则**。实测两条四次项路径都不通（N=30、
   f_x=0.3 工作点）：B<0 的 x⁴ 软化被高 q 放大到与端部二次囚禁
   同量级，B=−1e-8 已双阱失稳、链解体（律 5 的高 q 加强版）；F>0
   只换 4–7% 的 a_mm 下降，端距压缩还抬高横向库仑负载，F≈5e-10
   即弯链分岔。结论：要频率底线就接受 a_end~d_o，要小 MM 只能
   退回低频长链备选

### 5.9 从二维格点 CSV 拟合 A、B（--fit-ab）

有径向平面 RF 电势的格点数据（仿真导出或实测插值）时，不必手估 A/B——
`--fit-ab CSV` 对 Laplace 基做线性最小二乘（常数项吸收任意零点偏移，
电势差常数不影响场）：

```bash
python -m radial_trap --fit-ab rf_grid.csv --N 10 --phonon
```

```text
格点拟合 A/B: rf_grid.csv（165 点, x∈[-40,40], y∈[-25,25] µm）
  A=4.987e-03 V/µm², B=-2.31e-08 V/µm⁴, V0=0.012 V
拟合质量: R²=0.99998, 残差 RMS=3.2e-05 V
```

CSV 格式（自动识别）：**Comsol 导出**（`%` 元数据行剥离 + `% Length unit`
坐标单位 m/cm/mm/um/nm 自动换算到 µm，电势列名任意如 `esbe.V (V)`，
多表达式导出只保留一个电势表达式）；**简单表头**（列名大小写不敏感，
x 列 x/x_um/x(um)/…，电势列 v/v_rf/phi/phi_rf/potential/…，恰 3 列且
前两列为 x/y 时电势列名不限）；**无表头**恰 3 列数值按 x, y, V 顺序。
单位：无 `% Length unit` 行时 x/y=µm、V=伏特（与模块一致），支持
UTF-8 BOM。拟合值作为本次运行的 A/B 继续常规流程（可与 --N/--phonon/
--export-poly 等任意组合）；与 --A/--B/--from-freq 或含 A/B 的 --json
冲突报错；R²<0.9 告警（格点与 Laplace 基偏差大，A/B 仅粗估）。

从 Comsol 导出的操作建议：Global Evaluation/派生值导出格点时只选径向
平面 (x,y) 与**一个**电势表达式，且只对 RF 电极加激励——直接导出含 DC
的总电势（默认 `esbe.V` 全场）会把 DC 二次分量混入 A，R² 显著偏低
即此征兆。

三点注意：**① 拟合范围**——格点覆盖远大于链展宽时更高阶 Laplace 项
（六阶等）会泄漏进 A/B（实测 ±300 µm 全域拟合把 A 抬高 ~50%），用
`--fit-ab-range RX RY`（µm，以阱心为中心）限制到离子区，经验取链展宽
×1.5~2（如 N=30 span 71 µm → `--fit-ab-range 80 80`），并核对 A/B 对
R 的稳定性；**② B 的可辨识性**取决于格点横向范围（x⁴ 基幅度 ∝ x_max⁴，
范围过小则 B 与常数/四极近简并、对噪声敏感）——限域也别小于链展宽；
E/D/F 不在拟合之列（RF 格点只约束 RF 势），仍按 §3/§5.6 选取。与
--from-freq 的分工见技术参考 §6。R² 偏低时先查两件事：导出的是否纯
RF 势（含 DC 总势会把 DC 二次分量混入 A），以及是否需要限域（对称类
内高阶项泄漏——对称化/滤波无济于事，定量见技术参考 §6）。

---

## 6. 作为 Python 库使用

```python
import numpy as np
from radial_trap.types import RadialTrapParams
from radial_trap.potential import (
    build_total_potential_fit,
    evaluate_conditions,
    invert_trap_freqs_to_params,
    total_potential_V,
)
from radial_trap.lattice import (
    find_radial_equilibrium,
    micromotion_amplitude,
    solve_inplane_phonons,
)

params = RadialTrapParams(A_V_per_um2=5e-3, n_ions=10)   # 其余取默认（E=−3.5e-4，链沿 x）

conds = evaluate_conditions(params)              # 4 条件 + f/q/eps4 派生量
fit = build_total_potential_fit(params)          # FitResult3D（可 eval/grad/hessian）
eq = find_radial_equilibrium(fit, params)        # eq.r_eq_um: (N, 3) µm
mm = micromotion_amplitude(eq.r_eq_um, params)   # mm.a_mm_um / mm.mag_um / mm.excess_ratio
ph = solve_inplane_phonons(fit, eq.r_eq_um, params)   # ph.freq_hz_signed（负值=不稳定）

A, E, q_x = invert_trap_freqs_to_params(          # 阱频反算（§5.6）
    0.7091, 5.086, params.freq_rf_mhz, params.species_name)
xx, yy = np.meshgrid(                            # 径向平面总势网格 (V)
    np.linspace(*params.x_range_um, 201),
    np.linspace(*params.y_range_um, 141), indexing="ij")
v = total_potential_V(xx, yy, params)

print(f"f_x={conds.f_trap_x_mhz:.3f} MHz, q={conds.q_mathieu_x:.4f}")
print(eq.r_eq_um[:2])          # 前两个离子的平衡位置（y 分量 = 0，链沿 x）
print(mm.mag_um.max())         # 最大 micromotion 峰值 (µm)
```

绘图用 `radial_trap.plots.plot_potential_maps`（§5.5）；交互窗类
`radial_trap.ui.RadialTrapUI` 的 `refresh()` / `save_json()` 不弹窗、可无头调用。

---

## 7. FAQ

**负数参数报 "expected one argument"？**
含逗号的范围（`--x-range -10,10`）必然被 argparse 当成旗标；部分负数科学计数法形式
在某些 Python 版本同样中招。**统一用 `=` 语法**：`--x-range=-10,10`、`--F=-5e-8`、
`--E=-1.5e-4`——`=` 形式总是安全。

**为什么链在 x 轴而不是 y 轴？**
设计笔记的 3D 理想晶格全部分布在 xoz 平面；本模块只看径向平面，对应表现就是
离子集中在 x 轴（x = 弱/晶格轴，y = 紧束缚净轴）。默认 E=−3.5e-4 即此取向；
E 变号链即翻转到 y 轴（§5.1）。

**导出势跑 main.py 后晶格变两层/成面？**
径向模块把 z 钉在 0 平面，导出势不含任何 z 项——3D 动力学里 z 零囚禁，
一维链只是鞍点，库伦作用会把离子沿 z 摊开。解决：`--export-fz FZ_MHZ`
附加简谐轴向囚禁（f_z/f_x ≳ r_min(N)≈0.60·N^0.88——N=20 约 8.5，
N 越大要求越高；不足时链会沿 z zigzag 成双层，见 §5.8 设计律 3）。

**条件 (3)(4) FAIL 了程序为什么还继续跑？**
FAIL 仅打 WARNING 仍求解：违反 (3)(4) 只是失去**二次**囚禁，B≠0 时
16αB²(x²+y²)³ 六次项仍可能束缚（物理上对应双阱/zigzag 构型）。
此时盯住输出里的 f_y=nan 与 `⚠ 有离子贴近 x/y 边界`（hit_bounds）。

**ρ 一堆 NaN？**
ρ = a_mm/((q/2)·r) 在 |r_a| < 1e-9 µm 或 q_a = 0 处无定义（z 轴恒 NaN；
恰好在轴上的分量、N=1 原点也是）。链沿 x 时 ρ_y 常为 NaN，属正常。

**图里为什么全是英文？**
matplotlib 默认字体（DejaVu Sans）没有 CJK 字形，中文会渲染成方框；图内文本因此
统一用英文（µm、ρ 等 Greek 字形正常）。控制台输出与文档不受影响，仍为中文。

**解析 micromotion 能和 main.py 轨迹对比验证吗？**
不能直接对比：`--export-poly` 导出的力场时不变（无 RF 相位），轨迹只含 secular 运动。
解析 micromotion 的验证目前依赖两个恒等式：B=0 时 a_mm=(q/2)·r（ρ≡1）、
B≠0 时 ρ ≡ E_full/E_quad（逐点代数恒等）。时变 RF 力扩展是后续工作。

**hit_bounds=True 怎么办？**
离子贴到 `--x-range/--y-range` 边界：要么构型非囚禁（先查条件 (3)(4) 与弱轴系数），
要么链比范围长（放宽 range；注意默认 x ±30 / y ±10 µm——链沿 y 的翻转演示就
需要放宽 y，见 §5.1）。

**怎么复核一个结果？**
`--json` 存档 + `--report` 报告可完整复现；平衡是局部优化，极端参数（浅阱+大 N）
建议换 `--seed` 复核是否同一能量。
