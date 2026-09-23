# radial_trap — 2D 径向多项式势阱模拟

以 Laplace 多项式基**解析**指定径向平面 (x,y) 的 RF+DC 势，计算离子链平衡构型、
各离子解析 micromotion 幅度与晶格成形条件；可选面内声子。不需要电场 CSV，
无需 C++ 构建（纯 numpy/scipy）。

设计动机与系数推导见笔记《关于x方向非谐势的讨论.md》；本模块是其可计算实现。

用户教程（快速上手/选参/常见任务/FAQ）: [radial_trap_tutorial.md](radial_trap_tutorial.md)；
本文件为技术参考。

运行: `python -m radial_trap --N 10`（默认参数即有效演示：q≈0.29，
f_x≈0.71 MHz，f_y≈5.09 MHz，线性链沿 x 轴，4 条件全过）

## 1. 物理模型

约定与全仓一致：**RF 幅度为零到峰**（V₀·cos Ωt），β/a_mm 为一阶 RF 谐波
**峰值**（µm），Mathieu q = 4Q·k₂/(mΩ²)。

RF 电势取 x,y 镜像对称的 2D Laplace 解（无低次项，RF null 恰在原点）：

```
φ_rf = A(x²−y²) + B(x⁴−6x²y²+y⁴)      [V]
```

RF 场按设计文档取 **E_rf = +∇φ_rf**（物理电场为 −∇φ，仅分量符号相反，
不影响模长与条件判断）：

```
E_rf = (2Ax+4Bx³−12Bxy²,  −2Ay−12Bx²y+4By³)   [V/µm，零到峰]
```

赝势用**物种自身质量**（不同于 field_visualize 硬编码 Ba135 的路径）：

```
φ_pp[V] = α_eff·|E_rf[V/µm]|²,   α_eff = Q·1e12/(4·m_kg·Ω²)  [µm²/V]
```

（= 赝势能量 Ψ[J] = Q²|E|²/(4mΩ²) 除以 Q 的伏特形式——单电荷离子即除以
元电荷 e；Note 推导用能量量纲 α_E = Q²/4mΩ²，代码全篇统一电势量纲
α_V = α_E/Q = Q·1e12/(4mΩ²)，势/系数均为 V，符号条件不变。）

静态部分：

- RF bias：D·φ_rf（无量纲 D——RF 电极上的直流偏置，与 φ_rf 同空间形状）
- DC：E(x²−y²) + F(x⁴−6x²y²+y⁴)（**E/F 为势系数，非电场记号**）


| 参数  | 单位    | 含义                              |
| --- | ----- | ------------------------------- |
| A   | V/µm² | RF 四极系数（q ∝ A）                  |
| B   | V/µm⁴ | RF 十六极系数（引入高次赝势与过剩 micromotion） |
| D   | —     | RF bias 无量纲系数                   |
| E   | V/µm² | DC 四极系数                         |
| F   | V/µm⁴ | DC 十六极系数                        |


**轴向定位（v1.1 起默认）**：设计笔记对应的 3D 理想晶格全部分布在 xoz 平面
（y=0）；本模块只考虑径向平面 (x,y)，正确表现为**离子集中在 x 轴上**——即
x 取弱/晶格轴（E<0 → c_x2<c_y2），y 为紧束缚净轴。默认 E=−3.5e-4 给出
f_x≈0.71 / f_y≈5.09 MHz（ω_y/ω_x≈7.2，深处线性区）；E 变号即互换 x/y
角色（链翻转到 y 轴）。

## 2. 总势与晶格条件

φ_tot = φ_pp + D·φ_rf + φ_DC 展开为 9 个 x,y 偶次单项式（最高 6 次；
B² 部分已归并 16αB²(x²+y²)³）：


| 单项式        | 系数 (V/µm^n)            |
| ---------- | ---------------------- |
| x²         | c_x2 = 4αA² + DA + E   |
| y²         | c_y2 = 4αA² − DA − E   |
| x⁴         | s_x4 = 16αAB + DB + F  |
| y⁴         | s_y4 = −16αAB + DB + F |
| x²y²       | −6(DB+F)               |
| x⁶, y⁶     | 各 16αB²                |
| x⁴y², x²y⁴ | 各 48αB²                |


设计文档的 4 条 2D 晶格成形条件（CLI 用 `--cond-tol`，默认 1e-12，判 (1)(2)）：

1. **压制交叉项**：DB+F → 0 且 B → 0（否则 x²y² 项使链变形）
2. **压制 y 方向高次**：−16αAB+DB+F → 0 且 B → 0（否则 y⁴ 显著）
3. **y 束缚**：c_y2 > 0
4. **x 束缚**：c_x2 > 0

(3)(4) 违反仅**警告**仍求解：B≠0 时 16αB²(x²+y²)³ 六次项仍可束缚
（物理上对应双阱/zigzag 一类构型）；离子贴边时报告 `hit_bounds=True`。

派生诊断量（报告与 npz 均输出）：

- 轴阱频 f_a = √(2Q·c_a2/m)/2π（复用 `k2_to_trap_freq_MHz`）；c_a2 ≤ 0 时 NaN
- Mathieu q_x = 4QA·1e12/(mΩ²)，q_y = −q_x
- 四次项相对重要性 eps4_a = |s_a4|·L_ref²/c_a2（L_ref=10 µm）——刻画设计文档
"二次项主导"近似在多大范围内成立
- xy_degenerate：c_x2 ≈ c_y2（相对差 < 1e-6）时链取向简并，初始条件决定结果



## 3. 平衡求解与解析 micromotion

**平衡**：L-BFGS-B 最小化外势+库伦总能量（复用 `equilibrium.energy`
`total_energy_and_grad`，单位 eV）。初始位置按 `--seed` 均匀撒在
x/y range 内（z 钉在 0 平面，窄 bounds ±1e-6 µm 防漂移——z 方向无势场依赖）。
随机初始化的健壮性：不同 seed 收敛到同一能量（N=10 双 seed 实测一致）。

**热启动（v1.1）**：`find_radial_equilibrium(..., r_init_um=<上一解>)` 可选参数
以既有构型为初值——UI 连续微调场景使用（相邻参数同一收敛点、迭代数不增、
防简并附近取向翻转）。

**解析 micromotion**（每离子，一阶 RF 谐波峰值）：

```
a_mm = (Q·1e12/(m_kg·Ω²))·E_rf(r_eq)      [µm，矢量，z 分量恒 0]
```

四极极限恒等式 **a_mm,a = (q_a/2)·r_a**（q_y = −q_x）。据此定义 excess 比：

```
ρ_a = a_mm,a / ((q_a/2)·r_a)
```

- B=0：ρ ≡ 1（数值验证 rtol 1e-9）；|r_a| < 1e-9 µm 或 q_a=0 处无定义 → NaN
- B≠0：ρ = E_full,a/E_quad,a 偏离 1，随 |r| 增大——即高阶 RF 场导致的
**过剩 micromotion**（文档结论：二次主导的 RF 场通过 micromotion 限制
x 方向可用范围，ρ 即其定量刻画）

### 非理想四极（B≠0）的具体计算

一般推导（一阶 Floquet / 快慢分离）：把运动拆为慢 secular 部分 R(t) 与
快 RF 微扰 ξ(t)，RF 项按 m·ξ̈ = Q·E_rf(R)·cos Ωt 线性响应，得

```
ξ(t) = (Q·1e12/(m_kg·Ω²))·E_rf(R)·cos Ωt      [µm，零到峰]
```

即 **a_mm 在静态平衡位置 r_eq 处取 E_rf 的值**（`lattice.py
micromotion_amplitude` 即此实现；E_rf 取 +∇φ_rf 约定 → a_mm 与 E_rf 同号，
换物理电场约定仅整体反号，模长/ρ 不变）。secular 运动时同一式逐时刻给出
乘性调制 a_mm(R(t))——与 motion_analysis 处理的 x(t) ≈ X_sec·[1+(q/2)cos Ωt]
图像一致。

B≠0 时 E_rf 含三次项，代入整理（q_y = −q_x，(Q·1e12/(mΩ²))·2A = q_x/2）：

```
a_mm,x = (q_x/2)·x·[1 + (2B/A)(x² − 3y²)]
a_mm,y = (q_y/2)·y·[1 + (2B/A)(3x² − y²)]
|a_mm| = (Q·1e12/(m_kg·Ω²))·|E_rf(r_eq)|
```

方括号因子即 excess 比的**闭式解** ρ_x = 1+(2B/A)(x²−3y²)、
ρ_y = 1+(2B/A)(3x²−y²)（与逐点代数值 E_full/E_quad 恒等）。链上（y=0）
简化为 ρ_x = 1 + (2B/A)x²——端部离子 |r| 最大、偏离最远；B<0 时端部
|ρ_x| < 1 即"压端部 micromotion"的机制（代价是 x⁴ 软化失稳，见教程
§5.8 设计律 5）。

适用范围：一阶（q 量级）结果，忽略 O(q²) 的二次谐波与 secular–RF 参变
耦合；q ≪ 1 且 Ω ≫ ω_secular 时成立（本模块默认参数 f_RF=35.28 MHz、
q ≤ 0.3 均满足）。



## 4. CLI

```bash
python -m radial_trap                                   # 默认演示参数（线性链沿 x 轴）
python -m radial_trap --N 2 --phonon                    # N=2 + 面内声子
python -m radial_trap --B 1e-7 --D 0.5 --F=-5e-8        # DB+F=0 精确抵消演示
python -m radial_trap --json params.json --report r.json
python -m radial_trap --N 10 --plot --plot-potential --export-poly total_poly.json
python -m radial_trap --from-freq 0.7091 5.086 --N 10   # 目标阱频反算 A,E（恰回默认）
python -m radial_trap --fit-ab rf_grid.csv --N 10       # 格点 CSV 拟合 A,B 再运行
python -m radial_trap --ui                              # 滑块交互窗（需桌面环境）
```


| 参数                    | 默认                                  | 说明                                                                                      |
| --------------------- | ----------------------------------- | --------------------------------------------------------------------------------------- |
| --A/--B/--D/--E/--F   | 5e-3 / 0 / 0 / -3.5e-4 / 0          | 势系数（见上表）                                                                                |
| --freq-rf-mhz         | 35.28                               | RF 频率                                                                                   |
| --species             | Ba135+                              | 离子种类（ION_SPECIES）                                                                       |
| --N                   | 10                                  | 离子数                                                                                     |
| --x-range / --y-range | -30,30 / -10,10                     | 初始化与囚禁边界 (µm)；**负数须 = 语法**：`--x-range=-10,10`                                           |
| --seed                | 0                                   | 初始化随机种子                                                                                 |
| --softening-um        | 1e-3                                | 库伦软化长度                                                                                  |
| --scale-um            | 100                                 | FitResult3D 归一化长度（不影响物理结果）                                                              |
| --maxiter / --tol     | 5000 / 1e-15                        | L-BFGS-B 参数（gtol 固定 1e-12 保证间距精度）                                                       |
| --cond-tol            | 1e-12                               | 条件(1)(2) 趋零判定容差                                                                         |
| --phonon              | off                                 | 求解面内声子                                                                                  |
| --phonon-print-modes  | 10                                  | 控制台打印模式数（频率降序）                                                                          |
| --json                | —                                   | 参数存档读入；**严格键集** A,B,D,E,F,freq_rf_mhz,species,N,x_range,y_range,seed,softening_um；未知键报错 |
| --out                 | radial_trap/results/radial_N{N}.npz | npz 输出                                                                                  |
| --report              | —                                   | JSON 报告输出                                                                               |
| --plot / --plot-out   | off                                 | 双联图 PNG（构型+micromotion 线段 / 幅度与 ρ）                                                      |
| --export-poly         | —                                   | 总势系数 JSON（main.py --poly-potential 可读回）                                                 |
| --export-fz           | —                                   | 配合 --export-poly：附加简谐轴向囚禁 c_z2·z²（参数=目标 f_z MHz）。不加则导出势无 z 项，3D 演化 z 无囚禁、链摊成片（§7） |
| --plot-potential      | off                                 | 势场 2×2 分布图 PNG（赝势/RF bias/DC/总势+离子，§6）                                                  |
| --potential-out       | results/potential_N{N}.png          | 势场图输出路径（默认 radial_trap/results/）                                                        |
| --from-freq           | —                                   | 目标阱频 FX FY (MHz) 反算 A,E（§6；与显式/--json 的 A/E 冲突报错）                                       |
| --fit-ab              | —                                   | 二维格点 CSV 拟合 RF 势 A,B 作为本次运行参数（§6；与 --A/--B/--from-freq/含 A/B 的 --json 冲突报错）                 |
| --ui                  | off                                 | 滑块交互窗（需交互式 matplotlib 后端，Agg 下报错；§6）                                                    |


参数优先级：**CLI 显式 > --json > 内置默认**。

负数值参数推荐一律用 `=` 语法（`--F=-5e-8`、`--E=-1.5e-4`）——含逗号的范围与
部分负数科学计数法形式会被 argparse 误认成旗标，`=` 形式总是安全。

**--from-freq 解析顺序**：加载 --json → 解析 species/freq_rf_mhz/D（优先级同上：
CLI 显式 > --json > 内置默认）→ 冲突检查 → 反算写回 A/E → 常规合并。四种冲突
（显式 --A、显式 --E、--json 含 A、--json 含 E）→ 报错并列出全部冲突项（反算
结果会覆盖它们，须移除其一）；反算后 |q|≥0.9 时告警接近 Mathieu 第一稳定区边界。

## 5. 输出格式

**npz**（`--out`）：`params_`*（RadialTrapParams 全字段）、*`cond_`*
*（LatticeConditions 全字段，含 4 个 pass 旗标、f_trap_*/q_mathieu_*/eps4_*）、
`r_eq_um`/`r_init_um`、`a_mm_um`/`a_mm_mag_um`/`a_mm_excess_ratio`、
`energy_total_eV`/`grad_norm_eV_per_um`/`converged`/`hit_bounds`/`n_iter`/`message`、
`--phonon` 时 `phonon_freqs_hz`（负值=不稳定模）。

**JSON 报告**（`--report`）：与 npz 同构的分节结构
（params/conditions/equilibrium/micromotion/phonon），NaN → null。

**--plot-potential**（`--potential-out`，默认 `radial_trap/results/potential_N{N}.png`）：
径向平面势场 2×2 分布图 PNG（赝势/RF bias/DC/总势+离子散点），详见 §6。

**--export-poly**：`write_potential_fit_coeff_json` 格式
（coefficients 项标签 → V，含 x^6 项——依赖 potential_fit_3d 的每变量次数
上限 6）。衔接动力学：

```bash
python main.py --poly-potential total_poly.json --N 2 --time 5
```

注意 poly 力场**时不变**（secular 动力学，不含 RF micromotion）。

**--export-fz FZ_MHZ**（与 --export-poly 同用）：模块总势定义在 xoy 径向平面，
z 恒钉 0、**无任何 z 项**；导出势若不加轴向囚禁，`main.py --poly-potential`
的 3D 演化中 z 方向零囚禁——库伦作用把离子沿 z 摊开成片、径向一维链只是
鞍点（3N 声子含 N 个负 z 模）。该旗标按目标轴向阱频 f_z 附加简谐项
c_z2·z²（c_z2 = m ω_z²/(2Q)，复用 `trap_freq_MHz_to_k2`；对应真实阱的
端帽 DC 轴向囚禁），`z^2` 标签原生往返于标签解析器。单排链要求
f_y/f_x 与 f_z/f_x **均 ≳ r_min(N) ≈ 0.60·N^0.88**（有限 N zigzag 边界，
见 §7；N=20/30/40 分别 ≳ 8.4/12.0/15.5），建议留裕量（常规 ~25%；
为压 q_x 也可收至 7–8%，须以 `--phonon` 全谱核验——频率底线设计即此，
实例见教程 §5.8）。

## 6. 势场分布图 / 阱频反算 / 格点 CSV 拟合 / UI（v1.1–v1.5）



### 势场分布图

`--plot-potential`（`plot_potential_maps`）输出径向平面 2×2 面板：

- **RF 赝势** α|E_rf|²（viridis，恒 ≥ 0）
- **RF bias** D·φ_rf（RdBu_r，clim 关于 0 对称）
- **DC** E(x²−y²)+F(x⁴−6x²y²+y⁴)（RdBu_r）
- **总势 + 离子散点**

色标用 `np.nanpercentile` 稳健裁剪（默认 0.5–99.5%——B≠0 时角落 r⁴/r⁶ 爆涨
必须裁）；跨零面板关于 0 对称配色，单符号面板用顺序 cmap。D=0 时 bias 面板
恒零 → 平色 pcolormesh + "zero everywhere" 注记（contourf 空 levels 守卫）。
组分有解析函数 `pseudopotential_V` / `bias_potential_V` / `dc_potential_V` /
`total_potential_V`，恒等式 pp+bias+dc == total 由构造成立（测试 rtol 1e-12）。

### 阱频反算

二次系数 c_x2=4αA²+DA+E、c_y2=4αA²−DA−E 相加时 E 与 D 消去：
**c_x2+c_y2 = 8αA²**，给定两轴阱频即可反解：

```
A = √((c_x2+c_y2)/(8α)) ≥ 0     （A>0 → q_x>0；|q| 完全由 f_x²+f_y² 决定）
E = (c_x2−c_y2)/2 − D·A          （D≠0 时 E 平移 −D·A，c 分配不变）
```

`invert_trap_freqs_to_params(f_x, f_y, freq_rf_mhz, species, D=0.0)` 先以
`trap_freq_MHz_to_k2`（`FieldParser.k2_to_trap_freq_MHz` 的精确逆）把目标频率
换成 c_a2 再解上式；f≤0 或 c_x2+c_y2≤0 → ValueError。**局限**：反算只定
二次项——B/F 与高阶行为不受约束，需另行选值并检查条件(1)(2)；A 被 √ 唯一
确定，无参数余地换 q。

CLI `--from-freq FX FY`：解析顺序与四种冲突见 §4；示例 `--from-freq 0.7091 5.086` 恰好反算回默认 A=5e-3、E=−3.5e-4（q_x≈0.291）。

### 格点 CSV 拟合 A、B（--fit-ab）

从二维格点电势数据反演 RF 势的四极与十六极系数：`fitting.py` 对
Laplace 基做**线性最小二乘**（常数项吸收任意零点偏移，电势差常数不
影响场）：

```
V(x, y) ≈ V₀ + A·(x²−y²) + B·(x⁴−6x²y²+y⁴)
```

- `load_radial_grid_csv(path)` → (x, y, v) 数组。三种格式自动识别：
  **Comsol 导出**（`%` 前缀元数据行全部剥离，`% Length unit,<unit>` 的
  m/cm/mm/um/nm 坐标单位自动换算到 µm——电势列名任意，如 `esbe.V (V)`，
  多表达式导出须只保留一个电势表达式）；**简单表头**（列名大小写不敏感，
  x 取 {x, x_um, x(um), x[um]}，y 同理，电势取 {v, v_rf, phi, phi_rf,
  potential, u, v_volt}，恰 3 列且前两列为 x/y 时电势列名不限）；**无表头**
  且恰 3 列数值时按 x, y, V 顺序解析。单位：无 `% Length unit` 行时 x/y
  按 µm 解析，V 一律伏特（与模块一致）；支持 UTF-8 BOM；含 NaN/Inf 的行
  （Comsol 在几何奇点/域外格点的典型导出形态）自动剔除并发 RuntimeWarning，
  `fit_rf_ab` 对直接传入的数组含非有限值则显式报错——否则单个 NaN 会使
  lstsq 全系数变 NaN 且 R² 兜底成 1.0，静默掩盖故障
- `fit_rf_ab(x, y, v, fit_range_um=None)` → `FitABResult`（A、B、V₀ +
  n_points、残差 RMS、R²、x/y 范围——限域时反映实际参与拟合的子范围）。
  B 的可辨识性取决于格点横向范围（x⁴ 基幅度 ∝ x_max⁴，范围过小则 B 与
  常数/四极近简并、对噪声敏感）——建议覆盖链的预计展宽（数十 µm 量级）
- CLI `--fit-ab CSV [--fit-ab-range RX RY]`：拟合值作为本次运行的 A/B
  继续常规流程（可与 --N/--phonon/--export-poly 等任意组合）；与
  --A/--B/--from-freq 或含 A/B 的 --json 冲突报错；R²<0.9 时告警（格点
  与 Laplace 基偏差大，A/B 仅粗估；未限域时附 --fit-ab-range 建议）
- **拟合范围与高阶泄漏（定量）**：格点覆盖远大于链展宽时，更高阶
  Laplace 项会泄漏进 [1, q2, q4] 最小二乘的 A/B。实测（Comsol 2D 导出、
  RF 电极 1V/DC 接地、±300×±100 µm、601×201 格点）：全域拟合 R²=0.871、
  A 被抬高 52%（1.19e-6 vs 全谐波收敛值 7.83e-7 V/µm²）、B 偏 2.7×；
  谐波分解显示主导剩余为**六阶项** q6=Re z⁶（RMS 贡献 8.4e-3 V，与 q4
  本身的 1.1e-2 V 同量级），其次八阶/边缘；而全部奇次（非对称）谐波
  ≤5e-5 V、四镜像对称化对 R² 无改善（0.8708→0.8709）——低 R² 的来源
  是**对称类内的高阶项而非左右不对称**，对称化/滤波无用，正确做法是
  `--fit-ab-range` 限制到离子区：R=100 µm 时 R²=0.998、A 误差 0.4%、
  B 收敛到 −5.9e-12。经验取链展宽 ×1.5~2（N=30 span 71 µm → R≈60–80），
  并核对 A/B 对 R 的稳定性
- **噪声评估：为何不提供滤波/去噪选项**。数据源是 FEM 确定性解，无随机
  噪声；即便混入（如降低导出精度位数、实测插值），线性最小二乘在
  ~10⁴–10⁵ 格点上的 1/√N 平均已将其压制（实测同一导出加 σ=1 mV 高斯
  噪声 → A 误差 0.2%、B 1.1%）。预平滑反而净伤害：干净数据上高斯平滑
  自身引入偏差（σ_smooth=3 µm → A 偏 0.25%、10 µm → 2%、30 µm → 13%，
  超过它所消除的噪声误差），且平滑使残差空间相关、R² 虚高，令 R²<0.9
  告警失效。对稀疏尖峰坏点，中值滤波虽实测有效（1% 格点被 0.1 V 尖峰
  污染时 B 误差 13% → 0.25%），但 FEM 导出不产生此形态——真遇到时先
  检查数据本身（NaN 行已自动剔除），去偏仍首选 `--fit-ab-range`
- **注意：拟合的是 RF 势**——Comsol 里请只对 RF 电极加激励（或单独导出
  RF 电势表达式）再导出；直接导出含 DC 的总电势（如默认 `esbe.V` 全场）
  会把 DC 二次分量混入 A，R² 显著偏低即此征兆

与 `--from-freq` 的分工：阱频反算给二次系数 A、E（B 不受阱频约束），
格点拟合直接给 RF 势的 A、B——两者都定 A，不可同用；E/D/F 仍需另行
选取（常用 E 由目标 f_x 反算或直接取 0，见教程 §5.6/§5.9）。

### UI（`--ui`）

`python -m radial_trap --ui` 打开 matplotlib.widgets 交互窗（无新依赖；
默认参数即初值，可与 --from-freq / --json 组合定初值）：

- 左主面板：总势 pcolormesh（原位 `set_array/set_clim/set_cmap` 更新，无
collection 重建；仅区域改变时重建）+ 离子散点 + micromotion 线段，
**colorbar 水平置于面板下方**（v1.3，fraction=0.075/pad=0.10，不占面板
宽度——x 域拉宽时等比画面尽量少缩小）；右侧
读出：系数、c_x2/c_y2、f_x/f_y、q、4 条件 PASS/FAIL、N/链跨度/|a_mm|
- 底部 5 系数行 = 滑块 + **数值框** + **范围框**（滑块初界 A∈[1e-4,2e-2]、
B∈[±5e-7]、D∈[±1]、E∈[±4e-4]、F∈[±2e-7]）：数值框直接输入精确值
（Enter/失焦提交，越界**自动扩展滑块范围**而非拒绝）；范围框 "lo,hi"
提交后按新界重建滑块。第 6 行 **N / x / y 区域框**：N 热启动增删离子后
重解（1..200），x/y 重建网格与 pcolormesh/colorbar（跨度 1e-3..1e4 µm）
- **键盘方向键**：聚焦数值框 ↑↓ 步进（Shift ×5，1%/5% 滑块跨度）、N 框
↑↓ ±1（Shift ±5）、区域/范围框 ↑↓ 以中心为锚放宽/收窄（Shift 细调）；
悬停滑块轴 ←→↑↓ 步进（1%/5% 跨度）、Home/End 跳端点
- **DB+F=0 约束开关**（CheckButtons）：勾选后 F := −D·B 随 B/D 联动，
F 滑块/文本框隐藏禁调（行内灰字显示联动值）——条件 (1) 交叉项的精确
抵消手段；取消勾选 F 冻结在当前值恢复独立可调。约束状态不存档，联动
结果以数值进 Save JSON
- 右下 f_x/f_y 目标频率框（随当前阱频刷新，改掉即目标）+ **Solve A,E**
（反算回写滑块，越界裁剪并提示）+ **Save JSON**（--json 严格键集，写
`radial_trap/results/ui_params.json`）
- 更新流防事件风暴，v1.3 起两级：逐事件（拖动/键盘步进）只走廉价路径
（条件 + 势网格 + 原位刷新），且全图重绘节流至多 ~30 fps（`draw_idle`
间隔下限，拖动/按键收尾帧 force 直通不丢末帧）；**重解本身再防抖
~0.35 s**——release/键松开只置挂起标记并重启单发定时器
（`canvas.new_timer`，回调在 GUI 主线程），交互静止后才热启动重解一次
——方向键自动重复、滑块连点不再逐事件重解（v1.2 及之前卡死的根因）。
Slider 无 on_release API，以 release 事件落在滑块轴上判定；Agg 等无
定时器后端立即冲刷（`_TIMER_BACKENDS` 允许清单），无头可测性不变
- `RadialTrapUI.refresh(compute_eq=...)` / `set_coeff_value` /
`set_coeff_range` / `set_n_ions` / `set_region` / `set_constraint` /
`save_json()` 不调 `show()`，无头可测；需交互式 matplotlib 后端
（WSLg/X11/macOS），Agg 等无头后端 `parser.error` 明确提示



## 7. 面内声子

`--phonon` 在平衡位置线性化（复用 `equilibrium.phonon.solve_phonon_modes`，
`dof_indices` 取每离子 (3i, 3i+1) 共 2N 个面内自由度）。N=2 解析谱
（测试 T13，rtol 1e-6；a = 链轴/弱轴，t = 横向紧轴——默认链沿 x 即
a=x、t=y）：


| 模式                 | 频率           |
| ------------------ | ------------ |
| 轴向 COM             | f_a          |
| 轴向 stretch         | √3·f_a       |
| 横向 COM             | f_t          |
| 横向 stretch（zigzag） | √(f_t²−f_a²) |


（横向 stretch ω² = ω_t²−ω_a²，James 1998；库伦横向耦合在平衡处恰贡献 −mω_a²。）

**zigzag**：链沿弱轴 a 线性稳定的无限链阈值为 (ω_t/ω_a)² > 7ζ(3)/2 ≈ 4.21
（即 ω_t/ω_a > 2.05）。默认参数 ω_y/ω_x ≈ 7.2 深处线性区，N=10 基态即
x 轴线性链（3D 对应晶格卧在 xoz 面内）；弱化 |E| 使比值逼近阈值，链先
离开 x 轴（3D 对应离开 xoz 面）进而 zigzag；E 变号则 x/y 角色互换，
链翻转到 y 轴。

**有限 N 修正**：上述 2.05 是无限均匀链 LDA 阈值；有限谐和阱链的边界大幅
上移且 **与 q 无关**——以中心间距 d₀ ∝ c_x2^(−1/3) 标度，库伦横向项
∝ 1/d₀³ ∝ c_x2，与阱频平方同幂，在 r=f_y/f_x 中相消，r_min 只由 N 决定。
定量上定义横向库伦系数 c ≡ X/f_x²（X 为最近邻库伦横向耦合频率平方），
N=20 实测 **c ≈ 70–74**（模块面内 zigzag 谱与 3D z 方向声子两种独立测法
一致），比均匀链 LDA 值 7ζ(3)/2 ≈ 4.21 高一个量级——谐和阱有限链中心
间距只有自然库伦长度 (k_eQ²/(mω_a²))^{1/3} 的 ~0.38 倍，横向库伦耦合
相应增强 ~17.7 倍。

**r_min 随 N 增长**（y 钉扎直线链 + 面内声子最小模符号对 r 二分，收敛
容差 0.02）：N=20 → **8.40**、N=30 → **11.99**、N=40 → **15.46**，与
q 无关（N=30 在 q=0.02/0.035 两种取值下二分一致；q=0.22 的频率底线
设计取 r=13≈1.08·r_min 全谱仍正，进一步佐证），经验拟合
**r_min ≈ 0.60·N^0.88**（三点偏差 <0.5%）。故 N=20 边界 √c ≈ 8.5–8.6
（q=0.03/0.05 一致，面内 y 与 3D z 方向一致），远高于 2.05；但
"r=10 一劳永逸"只对 N=20 成立——N≥30 时 r=10 的直线链已是鞍点
（N=30 收敛到 zigzag、max|y|≈2.9 µm；N=40 max|y|≈5.3 µm），横向比须
按 N 放大并留裕量（低频版 ~25%：N=30 取 15、N=40 取 19；频率底线版
7–8%：13/16.5，以 --phonon 全谱核验）。**3D 的 z 方向同受
此约束**：导出势不加 f_z 时 z 零囚禁（§5 --export-fz）；即使加了，
f_z/f_x ≲ r_min(N) 时链也会沿 z zigzag 成双层——这正是"径向平面一维链
到 3D 变两层"的根源（20/30/40 离子整定实例见教程 §5.8）。

## 8. 解析验证（tests/test_radial_trap.py）

- RF 场 = 手式 + 网格数值梯度（内部点）
- α_eff 单位链独立校验（SI 能量形式）
- FitResult3D 求值 == 独立解析总势（rtol 1e-10）；scale 不变性；原点 Hessian
- 4 条件：四极 only 全过 / E 过大 y 失束缚 / DB+F 精确抵消时 B≠0 仍 FAIL
- N=1 居中；N=2 间距 = 2(k_eQ·1e6/(8c_y2))^{1/3}；E 变号链翻转（镜像构型）
- micromotion 恒等式 a_mm,a = ±(q_a/2)r_a（q 由测试内 SI 常数重算）
- B≠0：ρ ≡ E_full/E_quad 逐轴恒等；B=0：ρ=1
- N=2 面内声子 4 模解析谱（上表）
- CLI：解析/坏物种/--json 未知键 SystemExit；--json 往返 + CLI 覆盖；
npz 键形状；不稳定配置完成不崩；--plot 冒烟；--export-poly 往返
（load_poly_potential_json → eval 等于原 fit，rtol 1e-12）
- v1.1：势组分恒等 pp+bias+dc == total（rtol 1e-12）；k2 正逆往返；阱频反算
往返/含 D/简并/非法输入 ValueError；热启动 == 冷启动（r_eq allclose 且
n_iter 不增）；默认参数链在 x 轴 + CLI 默认 npz 新默认值；--from-freq 往返
  - 四种冲突 + 非法频率 exit 2；--plot-potential 冒烟（零 bias 面板与 B≠0
  爆涨裁剪两情形）；UI 构造/Solve/保存冒烟（Agg）+ --ui 后端守卫
- v1.2：UI 数值框精确设值（越界自动扩程）/范围框重建滑块/N 与区域框
（热启动增删离子、网格重建）；DB+F=0 约束（F 联动、控件隐藏、取消冻结、
程序化 set_constraint 幂等）；键盘步进（滑块轴 ←→ 1%/5%、聚焦框 ↑↓、
重解推迟到 key_release）
- v1.3：重解防抖（set_val 置挂起标记；无头后端 release 立即冲刷、非滑块
  轴 release 无副作用、`_flush_resolve` 清标记）
- v1.4：`augment_fit_with_axial_confinement`（z² 系数 = c_z2·L²、eval 差
  = c_z2·z²、Hessian zz → f_z 往返 rtol 1e-12、幂等、原 fit 不变）；
  --export-fz CLI（JSON 含 z^2 且与默认势之差恰为 c_z2·z²；缺
  --export-poly / f_z ≤ 0 → exit 2）

