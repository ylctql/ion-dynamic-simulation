# radial_trap — 2D 径向多项式势阱模拟

以 Laplace 多项式基**解析**指定径向平面 (x,y) 的 RF+DC 势，计算离子链平衡构型、
各离子解析 micromotion 幅度与晶格成形条件；可选面内声子。不需要电场 CSV，
无需 C++ 构建（纯 numpy/scipy）。

设计动机与系数推导见笔记《关于x方向非谐势的讨论.md》；本模块是其可计算实现。

运行: `python -m radial_trap --N 10`（默认参数即有效演示：q≈0.29，
f_x≈4.31 MHz，f_y≈2.78 MHz，4 条件全过）

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

（= 赝势能量 Ψ[J] = Q²|E|²/(4mΩ²) 除以 Q 的伏特形式；文档推导用能量形
α = Q²/4mΩ²，代码统一 V 形式，符号条件不变。）

静态部分：

- RF bias：D·φ_rf（无量纲 D——RF 电极上的直流偏置，与 φ_rf 同空间形状）
- DC：E(x²−y²) + F(x⁴−6x²y²+y⁴)（**E/F 为势系数，非电场记号**）

| 参数 | 单位 | 含义 |
|---|---|---|
| A | V/µm² | RF 四极系数（q ∝ A） |
| B | V/µm⁴ | RF 十六极系数（引入高次赝势与过剩 micromotion） |
| D | — | RF bias 无量纲系数 |
| E | V/µm² | DC 四极系数 |
| F | V/µm⁴ | DC 十六极系数 |

## 2. 总势与晶格条件

φ_tot = φ_pp + D·φ_rf + φ_DC 展开为 9 个 x,y 偶次单项式（最高 6 次；
B² 部分已归并 16αB²(x²+y²)³）：

| 单项式 | 系数 (V/µm^n) |
|---|---|
| x² | c_x2 = 4αA² + DA + E |
| y² | c_y2 = 4αA² − DA − E |
| x⁴ | s_x4 = 16αAB + DB + F |
| y⁴ | s_y4 = −16αAB + DB + F |
| x²y² | −6(DB+F) |
| x⁶, y⁶ | 各 16αB² |
| x⁴y², x²y⁴ | 各 48αB² |

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

## 4. CLI

```bash
python -m radial_trap                                   # 默认演示参数
python -m radial_trap --N 2 --phonon                    # N=2 + 面内声子
python -m radial_trap --B 1e-7 --D 0.5 --F -5e-8        # DB+F=0 精确抵消演示
python -m radial_trap --json params.json --report r.json
python -m radial_trap --N 10 --plot --export-poly total_poly.json
```

| 参数 | 默认 | 说明 |
|---|---|---|
| --A/--B/--D/--E/--F | 5e-3 / 0 / 0 / 1.5e-4 / 0 | 势系数（见上表） |
| --freq-rf-mhz | 35.28 | RF 频率 |
| --species | Ba135+ | 离子种类（ION_SPECIES） |
| --N | 10 | 离子数 |
| --x-range / --y-range | -10,10 / -80,80 | 初始化与囚禁边界 (µm)；**负数须 = 语法**：`--x-range=-10,10` |
| --seed | 0 | 初始化随机种子 |
| --softening-um | 1e-3 | 库伦软化长度 |
| --scale-um | 100 | FitResult3D 归一化长度（不影响物理结果） |
| --maxiter / --tol | 5000 / 1e-15 | L-BFGS-B 参数（gtol 固定 1e-12 保证间距精度） |
| --cond-tol | 1e-12 | 条件(1)(2) 趋零判定容差 |
| --phonon | off | 求解面内声子 |
| --phonon-print-modes | 10 | 控制台打印模式数（频率降序） |
| --json | — | 参数存档读入；**严格键集** A,B,D,E,F,freq_rf_mhz,species,N,x_range,y_range,seed,softening_um；未知键报错 |
| --out | radial_trap/results/radial_N{N}.npz | npz 输出 |
| --report | — | JSON 报告输出 |
| --plot / --plot-out | off | 双联图 PNG（构型+micromotion 线段 / 幅度与 ρ） |
| --export-poly | — | 总势系数 JSON（main.py --poly-potential 可读回） |

参数优先级：**CLI 显式 > --json > 内置默认**。

## 5. 输出格式

**npz**（`--out`）：`params_*`（RadialTrapParams 全字段）、`cond_*`
（LatticeConditions 全字段，含 4 个 pass 旗标、f_trap_*/q_mathieu_*/eps4_*）、
`r_eq_um`/`r_init_um`、`a_mm_um`/`a_mm_mag_um`/`a_mm_excess_ratio`、
`energy_total_eV`/`grad_norm_eV_per_um`/`converged`/`hit_bounds`/`n_iter`/`message`、
`--phonon` 时 `phonon_freqs_hz`（负值=不稳定模）。

**JSON 报告**（`--report`）：与 npz 同构的分节结构
（params/conditions/equilibrium/micromotion/phonon），NaN → null。

**--export-poly**：`write_potential_fit_coeff_json` 格式
（coefficients 项标签 → V，含 x^6 项——依赖 potential_fit_3d 的每变量次数
上限 6）。衔接动力学：

```bash
python main.py --poly-potential total_poly.json --N 2 --time 5
```

注意 poly 力场**时不变**（ secular 动力学，不含 RF micromotion），见 §8。

## 6. 面内声子

`--phonon` 在平衡位置线性化（复用 `equilibrium.phonon.solve_phonon_modes`，
`dof_indices` 取每离子 (3i, 3i+1) 共 2N 个面内自由度）。N=2 解析谱
（测试 T13，rtol 1e-6）：

| 模式 | 频率 |
|---|---|
| 轴向 stretch | √3·f_y |
| 径向 COM | f_x |
| 径向 stretch（横向） | √(f_x²−f_y²) |
| 轴向 COM | f_y |

（横向 stretch ω² = ω_x²−ω_y²，James 1998；库伦横向耦合在平衡处恰贡献 −mω_y²。）

**zigzag 提示**：默认参数 ω_x/ω_y ≈ 1.55，低于无限链 zigzag 阈值
（(ω_x/ω_y)² > 7ζ(3)/2 ≈ 4.21 → ω_x/ω_y > 2.05），故 N=10 默认演示的
基态是 zigzag 而非线性链——这是物理正确的基态（设计文档条件的意义所在），
非求解错误；提高 E（增强各向异性）可回到线性链。

## 7. 解析验证（tests/test_radial_trap.py）

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

## 8. v1 局限

- 单物种单参数点：不做多物种掺杂、不做参数扫描（--scan 留待后续）
- **micromotion 不能用 main.py 轨迹交叉验证**：poly 势力场时不变
  （无 RF 相位依赖），动力学只含 secular 运动。解析 micromotion 的数值
  验证依赖时变 RF 力扩展（后续）；当前以四极极限恒等式与 B≠0 场比恒等式
  作为解析验证
- 平衡求解为局部优化：随机初始化实测稳健（N≤20 多 seed 同能量），
  但极端参数（浅阱+大 N）可能落入局部极小，建议换 seed 复核
- z 方向无势场依赖（纯 2D 模型）：z 以窄 bounds 钉住，声子只取面内子空间
