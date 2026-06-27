# CLAUDE.md — ISM-Main 项目指南

## 项目概述

ISM-Main 是一个**离子阱动力学模拟**程序（MIT License，作者 Liangchen Yan）。在给定电势场分布下模拟离子晶格的动力学演化，支持 CPU 和 CUDA 加速的库仑力计算。

**语言**: Python ≥ 3.10，C++ (pybind11)  
**平台**: 仅 Linux / macOS（使用 `fork` 方式创建子进程，不支持 Windows）  
**构建**: CMake ≥ 3.18，依赖 Eigen 3.4 + pybind11（CMake 自动获取或 `externals/` 离线）

## 快速构建与运行

```bash
pip install -e .                  # 安装依赖
python build.py                   # 编译 C++ 扩展（默认尝试 CUDA，失败回退 CPU）
python build.py --no-cuda         # 强制 CPU-only
python main.py --N 50 --time 10 --plot   # 运行模拟
pytest                            # 运行测试
```

## 顶层架构

```
用户 CLI (main.py)
  └─ Interface/cli.py        解析参数 → ParsedRun
  └─ FieldConfiguration/     从 JSON + CSV 加载场配置 → Config (dt,dl,dV) + FieldSettings
  └─ FieldParser/            CSV 读取 → 电场插值 → 外力函数 force(r,v,t)
  └─ ComputeKernel/backend.py  启动子进程，调用 C++ ionsim 进行轨迹积分
  └─ Plotter/                实时 matplotlib 动画
```

**数据流**:
1. CLI 参数 → `ParsedRun`（包含 `Parameters`, `FieldSettings`, `Vision`, `Config` 等）
2. CSV + JSON → `FieldSettings` → `build_force()` → Python callable `force(r,v,t)`
3. `CalculationBackend` 在子进程中通过 `ionsim_calculate_trajectory()` 调用 C++ 核心积分
4. 子进程通过 `multiprocessing.Queue` 将 `Frame(r, v, timestamp)` 传回主进程
5. 主进程中 `DataPlotter` 或无头模式消费 Frame

## 模块详解

### `main.py` — 入口

- `_build_force()`: 从 CSV + FieldSettings 构建 Python 外力函数，可选 Savitzky-Golay 平滑
- `_create_backend_and_start()`: 创建队列 + 子进程 + `CalculationBackend`
- `run(parsed)`: 主模拟循环（有绘图 / 无头 / 连续采样三种模式）
- `main()`: 解析 CLI，支持逗号分隔的 `--N` 批量连跑

### `Interface/` — CLI 与参数

| 文件 | 关键内容 |
|------|---------|
| `cli.py` | `create_parser()`, `parse_and_build()`, `ParsedRun` 数据类 |
| `parameters.py` | `Parameters` 数据类（N, m, q, r0, v0, device, calc_method...），含同位素掺杂逻辑。默认 `calc_method="RK4"` |
| `bilayer_init.py` | `apply_bilayer_y_split()` — 将离子分为上下两层 |

### `FieldConfiguration/` — 常数与电压配置

| 文件 | 关键内容 |
|------|---------|
| `constants.py` | `Config` frozen dataclass (Omega, dt, dl, dV, freq_RF)；`init_from_config()` 从 JSON 计算无量纲单位 |
| `loader.py` | `build_voltage_list()`, `field_settings_from_config()` — 从 JSON 构建 `FieldSettings` |
| `field_settings.py` | `FieldSettings` 数据类（csv_filename, voltage_list, dissipation_mode, g）；`voltage_dc()`, `voltage_rf()` 工厂函数 |

**无量纲化**: 由 RF 频率导出 `dt`（Nyquist 时间）、`dl`（特征长度）、`dV`（特征电压），所有内部计算使用无量纲坐标。

### `FieldParser/` — 电场解析与力函数

| 文件 | 关键内容 |
|------|---------|
| `csv_reader.py` | `read()` — 读取 CSV 电场格点，自动检测单位，排序为 3D 网格 |
| `calc_field.py` | `calc_field()` — 对每个电极做 E = -grad(V)，返回 `RegularGridInterpolator` 列表；`calc_potential()` — 势插值 |
| `force.py` | `build_force()` → `make_force()` → 模块级 `force(r,v,t)` 函数（裁剪边界、加权求和、耗散项） |
| `potential_fit.py` | `fit_potential_1d()` — 1D 多项式拟合，`k2_to_trap_freq_MHz()` — 阱频计算 |

### `ComputeKernel/` — C++ 计算核心

| 文件 | 关键内容 |
|------|---------|
| `ionsim.cpp` | pybind11 绑定：`calculate_trajectory_impl()`，将 Python force 包装为 C++ 回调 |
| `numerical_integration.hpp` | `CalcTrajRK` (RK4，默认), `CalcTrajVV` (Velocity Verlet), `CalcTraj` 统一入口 |
| `types.hpp` | `DIM=3`, `data_t=double`, Eigen 数组类型 |
| `backend.py` | `CalculationBackend` — Python 侧：管理子进程、队列通信、批量帧输出 |
| `CMakeLists.txt` | 编译 `ionsim.cpp` + 数值积分 + 库仑力（可选 CUDA） |

**关键**: `ionsim_calculate_trajectory()` 接受 Python force callable，通过 pybind GIL 回调。`use_zero_force=True` 时跳过 Python 外力。

### `Plotter/` — 实时可视化

| 文件 | 关键内容 |
|------|---------|
| `dataplot.py` | `DataPlotter` — matplotlib 实时动画，1-2 个子图（zoy/zox/xoy），支持保存帧、rv 数据 |
| `blit.py` | `BlitManager` — 高效 blit 动画技术 |
| `vision.py` | `Vision` 数据类（显示范围、保存路径、颜色模式） |
| `color.py` | 颜色映射：y_pos / v2 / isotope（Ba133-138） |

### `ImgSimulation/` — 单帧离子成像

模拟 CCD/CMOS 传感器单帧图像，流程：
1. **动力学** — 调用 ionsim 积分离子轨迹
2. **曝光** — 将离子 2D 位置映射到像素网格，按高斯光束权重累积
3. **成像** — 高斯 PSF 模糊 + 可选散粒/读出噪声 + 可选归一化

| 文件 | 关键内容 |
|------|---------|
| `api.py` | 对外 API：`run_ion_image()`, `run_ion_image_from_json_file()`, `load_ion_image_json()` |
| `pipeline.py` | `render_single_frame()` — 完整 pipeline |
| `json_config.py` | `IonImageJsonBundle` frozen dataclass，从 JSON 加载全部参数 |
| `json_dynamics.py` | 解析 dynamics 节：初态（npz/µm/无量纲/随机）+ 力模型（zero/trap） |
| `types.py` | `CameraParams`, `BeamParams`, `NoiseParams`, `IntegrationParams` |
| `geometry.py` | `world_um_to_fractional_col_row()`, `bilinear_splat2d()` |
| `illumination.py` | `beam_intensity()` — 2D 高斯光束 |
| `integrate.py` | `integrate_exposure_xy_um()` — 梯形积分曝光 |
| `psf.py` | `apply_gaussian_psf()` — scipy 高斯卷积 |
| `noise_model.py` | `add_noise()` — Poisson + Gaussian + 背景偏移 |
| `normalize.py` | `normalize_image()` — max / minmax / none |
| `cli.py` | `python -m ImgSimulation` 入口 |

运行: `python -m ImgSimulation ImgSimulation/configs/example_ion_image.json --no-show -o out.png`

### `field_visualize/` — 电势场可视化

| 文件 | 关键内容 |
|------|---------|
| `core.py` | `compute_potentials()` (DC/RF赝势/总势), `apply_savgol_smooth()`, `build_grid_1d/2d()` |
| `trap_freq.py` | `compute_trap_freqs_at_point()`（除 `f_x/f_y/f_z` 外额外返回每轴纯二次模型 R² 谐性指标 `r2_quad_*`）, `quadratic_fit_r2()`, `compute_freq_scan_1d/2d()` |
| `symmetry.py` | `compute_symmetry_report()` — 势场对称性定量分析（镜面/旋转/多项式奇偶性/Hessian），支持 `which` 参数按需选择 |
| `plots.py` | `plot_1d()`, `plot_2d()`, `plot_bilayer()`, `plot_freq_scan_1d/2d()`, `print_symmetry_report()`, `plot_symmetry_radar()`, `plot_symmetry_deviation_heatmap()`, `plot_laplace_decomposition()` |
| `laplace_decompose.py` | `fit_laplace_2d()` — 2D Laplace 调和多项式基分解（四极 x²-y²、十六极 x⁴-6x²y²+y⁴ 等）；`eval_laplace_fit()`, `laplace_convergence()`, `print_laplace_report()` |
| `cli.py` | CLI 入口，参数解析 |

运行: `python field_visualize.py` 或 `python -m field_visualize`

阱频与谐性: `python -m field_visualize --csv <csv> --config <json> --freq [--x-range ...] [--freq-fit-degree 2|4]` — 输出 `f_x/f_y/f_z` 并附加每轴**纯二次模型 R²** 谐性指标（刻画势场在扫描范围内的整体谐性，弥补阱频仅反映原点局部曲率的不足；详见 `docs/harmonicity.md`）

对称性分析: `python field_visualize.py --csv <csv> --config <json> --symmetry m,r,p,h`（详见 `docs/symmetry_analysis.md`）

拉普拉斯分解: `python -m field_visualize --csv <csv> --config <json> --laplace [--laplace-max-degree 4] [--laplace-component dc|rf|pseudo|total] [--laplace-convergence]`

**注意**: 拉普拉斯分解严格适用于 DC 静电势和 RF 电势（满足 ∇²Φ = 0）。赝势 V_pseudo ∝ |∇Φ|² 不满足 Laplace 方程，分解仅作经验近似。

### `equilibrium/` — 平衡构型求解

| 文件 | 关键内容 |
|------|---------|
| `potential_fit_3d.py` | `fit_potential_3d_quartic()` — 3D 多项式拟合（默认 `quartic` 35 项；另有 125/27/10/4 项基组）；`grad_fit_3d()`, `hessian_fit_3d()` |
| `energy.py` | `trap_energy_and_grad()`, `coulomb_energy_and_grad()`, `total_energy_and_grad()` — 能量单位 eV |
| `phonon.py` | `solve_phonon_modes()` — Hessian + 动力学矩阵对角化 → 声子模；`PhononResult` |
| `find_equilibrium.py` | CLI：L-BFGS-B 最小化总势能，含 Hessian/声子谱/模式向量可视化 |

运行: `python -m equilibrium.find_equilibrium --N 40`

### `collision_pressure/` — 背景气压估算

模拟中性 H₂ 分子与俘获离子晶格的弹性碰撞，从重构概率反推背景气压。

| 文件 | 关键内容 |
|------|---------|
| `__main__.py` | CLI 入口：`python -m collision_pressure simulate` |
| `species.py` | `Species` 数据类 + 预置物种 (Ba135+, H2 等) |
| `collision.py` | 碰撞力学：散射角 (椭圆积分)、动量冲量 |
| `sampling.py` | 蒙特卡洛采样：速度、碰撞参数、方向 |
| `simulation.py` | `run_single_collision()` + `run_collision_scan()` |
| `reconfiguration.py` | `ZigzagFlipDetector` (排序+SSD) + `TopologyDetector` (Delaunay) |
| `topology.py` | Delaunay 三角剖分拓扑表征 |
| `pressure.py` | `estimate_pressure()` 从 P_flip 估算气压系数 |
| `config_scan.py` | 构型预扫描 + 平衡求解 |

运行: `python -m collision_pressure simulate --trap-freq 2 3 0.1 --n-ions 10 --workers 4`

**并行**: `--workers N` 使用 `multiprocessing.Pool.imap()` 并行运行碰撞模拟，按序返回结果（log 不乱序）。`--workers 1` 为串行（默认）。

### `field_optimize/` — 阱频反向设计

从目标阱频 (fx, fy, fz) MHz 反推电极电压，以势场对称性为正则化惩罚项。DC 电压对势场为线性叠加，优化 landscape 光滑、变量少（7-15），秒级收敛。详细文档见 `docs/field_optimize.md`。

| 文件 | 关键内容 |
|------|---------|
| `types.py` | `OptimizationConfig` (frozen: 目标频率、权重、边界、拟合参数)，`OptimizationResult` (优化前后电压/频率/对称性) |
| `objective.py` | `FastEvaluator` 预计算（1D 基函数矩阵 + 3D 网格），`compute_objective()` 频率误差 + 奇偶性/Hessian 惩罚 + NaN 保护 |
| `optimizer.py` | `optimize_voltages()` — scipy L-BFGS-B/Nelder-Mead 包装，含前后频率/对称性对比报告 |
| `cli.py` | CLI 入口；`python -m field_optimize --csv --config --target-freq fx fy fz` |

**数据流**: CSV+JSON → 插值器(一次) → `FastEvaluator` 预计算基函数矩阵 → 优化循环(仅矩阵-向量乘法) → 输出 JSON

**快速路径**: DC-only 模式下 RF 赝势恒定，每次 eval ~0.5ms；含对称性惩罚时增加 3D quartic 拟合。

运行: `python -m field_optimize --csv <csv> --config <json> --target-freq 2.0 3.0 0.1 [--optimize-rf-v0] [--out result.json]`

输出 JSON 与现有 config 格式兼容（含 `_optimization` 元数据），可直接用于其他模块。

### `benchmark/` — 性能测试

| 文件 | 关键内容 |
|------|---------|
| `common.py` | `run_simulation()` — 以子进程运行 main.py 并计时 |
| `plot_compare.py` | 有/无绘图性能对比（CUDA） |
| `device_compare.py` | CPU vs CUDA 性能对比 |

### `trap_stability/` — Mathieu 稳定性参数计算

从实际场几何计算 Mathieu a/q 稳定性参数、secular 频率、无量纲非谐常数（4/6 阶）、稳定性判断。

| 文件 | 关键内容 |
|------|---------|
| `stability.py` | `StabilityResult` frozen dataclass；`compute_stability_from_field()` 核心计算（6 阶多项式拟合 → a/q + 非谐常数）；`find_trap_center()` 自动检测陷阱中心；`check_stability_region()` 第一稳定区判断 |
| `cli.py` | CLI 入口 |

**数据流**: CSV+JSON → 插值器 → `compute_potentials()` 分离 DC/RF → 各轴 6 阶多项式拟合 → a/q + secular 频率 + 无量纲非谐常数 (anh4, anh6)

**非谐常数**: Taylor 系数 $c_{2k} = \Phi^{(2k)}/(2k)!$，通过 $dV$ 和 $dl$ 无量纲化: $\tilde{c}_{2k} = c_{2k}\cdot dl^{2k}/dV$。详细文档见 `docs/trap_stability.md`。

运行:
```bash
python -m trap_stability --csv <csv> --config <json> [--center 0,0,0] [--species Ca40+] [--out result.json]
```

### `motion_analysis/` — 动力学后分析（micromotion 数值测量）

基于 `continuous_sampling/` 的 `frame*.npz` 轨迹，逐离子数值测量 RF micromotion 幅度 β(t) 与有效调制深度 q_eff，并与 `trap_stability` 的理论 q 交叉验证。

物理：Paul 阱中离子运动为**乘性调制** x(t)≈X_sec(t)·[1+(q/2)cos(Ωt)]（非恒幅加性模型）。

| 文件 | 关键内容 |
|------|---------|
| `micromotion.py` | `load_continuous_sampling()`（RF 频率来自 config；含采样率/总时长断言）、`compute_micromotion()`（phase-folding 得时变 β(t) + FFT 低通提取 secular 包络并回归全局 q_eff）、`detect_warmup()`（secular 包络稳态检测瞬态收敛点 t*）、`analyze_run()` 多离子批处理（含 warmup 裁剪）、`cross_check_q()` 对接 trap_stability 理论 q |
| `plots.py` | `plot_ion_timeseries`/`plot_qeff_histogram`/`plot_qeff_vs_displacement`/`plot_beta_vs_secular`/`plot_lattice_micromotion`（zox 平面晶格**末端帧瞬时**位置 + 每离子 x 方向 micromotion 竖线，excess micromotion 成像；`show_theory` 叠理论比对竖线、`theory_z_offset` 挂钩中位离子间距、`equal_aspect=True` 默认等比、`axis_ranges` 按物理轴定尺度） |
| `__main__.py` | CLI 入口；`python -m motion_analysis` |
| `micromotion_analysis.ipynb` | 调用库的展示层 notebook |

**数据流**: `continuous_sampling/frame*.npz` → `load_continuous_sampling`（RF 频率从 config）→ `analyze_run` 逐 (ion,axis) 调 `detect_warmup` 裁瞬态 → `compute_micromotion`（FFT 低通提 secular 包络 + 回归 q_eff；phase-folding 滑窗得 β(t)）→ `cross_check_q`（trap_stability 理论 q 对比）

**瞬态裁剪（warmup）**: 默认逐 (ion,axis) 自动检测瞬态收敛点 t*（secular 包络滑窗稳态 + 3 窗中值滤波 + ≥3 连续 bad 段；混合 rel/abs 容差防冷离子过裁），裁掉弛豫段再分析。干净稳态/静止轴返回 no-op（`t*=t[0]`，结果与不裁一致）。CLI：`--no-auto-trim` 关闭、`--warmup-us`（相对）/`--trim-start-us`（绝对）手动 override、`--warmup-tol`(0.1)/`--warmup-periods`(3) 调参。结果记入 `IonAxisResult.t_star_us/dropped_frames/warmup_reason` 与 JSON `per_ion`。

**采样要求**: 每 RF 周期 ≥ 8 点（默认 `--interval 0.08 --step 10` 满足）；总时长 ≥ 3 个 secular 周期。不满足时 `load_continuous_sampling` 抛错并给出参数建议。

运行（先采集再分析）:
```bash
python main.py --N 3 --time 20 --continuous-sampling --continuous-sampling-frames 20000 --interval 0.08 --step 10
python -m motion_analysis continuous_sampling/t030.00_interval0.08_step10 \
    --csv monolithic20241118.csv --config default.json --out mm.json --plot-dir plots
```

说明文档: `docs/micromotion_analysis.md`（物理原理、测量方法、warmup 裁剪、CLI/输出格式）

规划文档: `docs/plan/micromotion_analysis.md`

## 核心类型 (`utils.py`)

- `CommandType` — START/PAUSE/RESUME/STOP（子进程控制）
- `Frame` — 单帧数据 (r, v, timestamp)
- `Message` — 控制消息（含初始条件、force callable）
- `Voltage` — 电势分量：`V0 * f(t) + V_bias`
- 时间函数工厂：`cos_time()`, `sin_time()`, `exp_decay()`, `exp_ramp()`, `constant()`, `cut_off()`

## 路径约定

- `setup_path.ensure_build_in_path()` — 将项目根和 build/（含 ionsim*.so）加入 `sys.path`
- `--csv` 支持仅传文件名，自动在 `data/` 下查找
- `--config` 支持仅传文件名，自动在 `FieldConfiguration/configs/` 下查找
- `--init-file` 的 `.npz` 需含 `r`（µm, shape (N,3)）和 `v`（m/s），可选 `t_us`
- `ImgSimulation` JSON 路径解析：先相对 JSON 文件目录，再相对仓库根

## 环境变量

- `ISM_DEFAULT_CONFIG`, `ISM_DEFAULT_CSV` — 覆盖默认配置路径
- `ISM_DEFAULT_SAVE_FIG_DIR` — 覆盖轨迹帧保存目录
- `ISM_LOG_LEVEL` — 日志级别 (DEBUG/INFO/WARNING/ERROR)

## 测试

```bash
pip install -e ".[dev]"
pytest                # 所有测试
```

| 测试文件 | 覆盖内容 |
|---------|---------|
| `tests/test_cli.py` | CLI 参数解析、save_times_us 语法 |
| `tests/test_force.py` | force 构建、_zero_force |
| `tests/test_parameters.py` | Parameters 构建、同位素掺杂、初态 |
| `tests/test_field_optimize.py` | FastEvaluator 预计算、目标函数、NaN 保护、优化收敛、CLI 解析、JSON 输出 |
| `tests/test_trap_stability.py` | a/q 教科书公式验证、物种质量标度、稳定性判断、secular 频率一致性、非谐常数（合成+场积分）、fit_degree=2/4/6、CLI 解析 |
| `tests/test_micromotion.py` | 合成数据回收已知 q（常位移/secular 调制）、q=0 退化、负 q、β(t) 跟踪 secular、加载/采样校验异常、多离子批处理 |
| `tests/test_cpu_cuda_error_accumulation.py` | 圆轨道 CPU/CUDA 误差对比 |

## 开发注意事项

1. **C++ 扩展必须先编译**: `ionsim` 模块由 CMake + pybind11 构建，产物在 `build/ionsim*.so`。未编译时多数功能不可用
2. **子进程 fork**: `CalculationBackend` 使用 `multiprocessing` with `fork`，不支持 Windows
3. **无量纲单位**: 所有内部计算使用 dt/dl/dV 无量纲坐标；与外部交互（CLI、npz、JSON）使用物理单位（µs、µm、m/s）
4. **force 函数签名**: `force(r: ndarray(N,3), v: ndarray(N,3), t: float) -> ndarray(N,3)`，位置会被裁剪到网格边界内
5. **同位素支持**: Ba133-Ba138 混合/单同位素/掺杂（alpha 比例），通过 `Parameters._apply_isotope_doping()` 处理
6. **JSON 配置版本**: `ImgSimulation` JSON 需要 `version: 1`
7. **`use_zero_force` 语义**: 为 True 时 ionsim 内部忽略 Python 外力（仅保留 C++ 库仑力），与 `dynamics.force: "trap"` 配合时应设为 False
8. **Savitzky-Golay 平滑**: `--smooth-axes` 默认沿 x,y,z 三轴对格点势场做平滑，影响力函数和场可视化
9. **能源单位**: `equilibrium` 模块使用 eV；外势做了零点平移（`V_shifted = V_true - V_min_grid`）
10. **externals/**: 可放置本地 Eigen/pybind11 以支持离线构建，此目录内容不应修改

## 关键数据格式

### npz 轨迹文件
- `r`: shape `(N, 3)`, 单位 µm
- `v`: shape `(N, 3)`, 单位 m/s
- `t_us`: 标量，当前时刻 (µs)

### 电场 CSV
- 首行为注释（含单位信息），随后为 x, y, z 坐标 + 各电极电势列
- `csv_reader.read()` 自动检测长度单位（mm 或 µm）并归一化

### 电压配置 JSON (`FieldConfiguration/configs/`)
- 含 RF 频率和各电极电压（DC 或 RF），`loader.py` 解析为 `Voltage` 对象列表
