# ISM-Main：离子阱动力学模拟

[English](README.md) | [中文](README.zh.md)

---

离子阱动力学模拟程序，采用模块化设计，在给定电势场分布下模拟离子晶格的动力学演化，支持 CPU 和 CUDA 加速的库仑力计算。

## 功能特点

- **模块化**：输入、场配置、力解析、计算核心、绘图等模块相互独立
- **积分算法**：支持 RK4 和 Velocity Verlet
- **GPU 加速**：库仑力可选用 CUDA 加速
- **实时绘图**：基于 matplotlib 的实时可视化
- **电势场可视化**：独立工具，可绘制静电势、RF 赝势、总电势的 1D 或 2D（热力图/三维曲面）分布
- **平衡构型求解**：拟合 3D 阱势并最小化总势能（外势 + 库仑）得到离子晶格平衡位置

## 环境要求

- Python ≥ 3.10
- CMake ≥ 3.18（用于编译 C++ 扩展）
- Eigen 3.4、pybind11（由 CMake 自动获取）
- 可选：CUDA Toolkit（用于 GPU 加速）

## 运行平台

**支持**：Linux、macOS  
**不支持**：Windows（程序使用 `fork` 方式创建子进程）

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

推荐使用 `pip install -e .` 安装，可统一依赖与路径管理；也可仅安装依赖：`pip install numpy scipy pandas matplotlib`。

### 2. 编译 C++ 扩展

```bash
python build.py
```

默认会尝试启用 CUDA（若已安装），不可用时自动回退为 CPU-only。构建后通过 `--device cpu` 或 `--device cuda` 选择计算设备。强制仅 CPU 构建：`python build.py --no-cuda`。

**离线构建**：将 Eigen 和 pybind11 放入 `externals/` 目录（如 `externals/eigen-4.3.0`、`externals/pybind11`），或使用 `python build.py --local /path/to/externals`，可避免联网下载。

更多构建选项见 [BUILD.md](BUILD.md)。

### 3. 运行

```bash
python main.py --N 50 --time 10 --plot
```

## 电势场可视化

`field_visualize` 用于可视化电场 CSV 与电压配置下的电势分布（静电势、RF 赝势、总电势），支持 1D（单坐标）与 2D（热力图或三维曲面）绘图，以及阱频计算与扫描。

```bash
python field_visualize.py [options]
# 或
python -m field_visualize [options]
```

### 使用示例

```bash
# 1D 沿 x 轴电势（默认）
python field_visualize.py

# 2D x-y 平面热力图
python field_visualize.py --vary x,y --mode heatmap

# 2D 三维曲面
python field_visualize.py --vary x,y --mode 3d

# 自定义范围并输出（负数参数需用 = 连接）
python field_visualize.py --vary x,y --x_range=-100,100 --y_range=-50,50 --out output.png

# 启用偏置与 RF 幅度图
python field_visualize.py --vary x,y --offset --show-rf-amp

# 1D 势场二次/四次拟合（叠加虚线，显示 R²、中心、k2）
python field_visualize.py --vary z --fit 2
python field_visualize.py --vary z --fit 4

# 计算阱频分布 f_x, f_y, f_z (MHz)
python field_visualize.py --freq
python field_visualize.py --freq --const 0,0,50 --freq-fit-degree 4

# 阱频沿轴扫描（不绘势场，仅绘阱频分布）
python field_visualize.py --freq-scan z --freq-scan-n 50
python field_visualize.py --freq-scan x,y --freq-scan-n 30,30 --mode heatmap
python field_visualize.py --freq-scan x,y --mode 3d --out freq_2d.png

# Savitzky-Golay 势场平滑（默认沿 z；--smooth-axes none 关闭）
python field_visualize.py --vary z
python field_visualize.py --freq --smooth-axes x,y,z --smooth-sg 15,3
python field_visualize.py --vary z --smooth-axes none
```

### 电势场可视化参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv` | data/monolithic20241118.csv | 电场 CSV 路径 |
| `--config` | FieldConfiguration/configs/default.json | 电压配置 JSON 路径 |
| `--vary` | x | 变化坐标：单坐标 (x/y/z) 为 1D；逗号分隔 (如 x,y) 为 2D |
| `--x_range` | -100,100 | 主变化方向范围 (μm)，逗号分隔 |
| `--y_range` | -100,100 | 2D 时第二坐标范围 (μm) |
| `--const` | 0,0,0 | 固定坐标 x,y,z (μm)，逗号分隔 |
| `--mode` | heatmap | 2D 模式：heatmap 或 3d |
| `--n_pts` | - | 1D：单个整数 (如 500)；2D：逗号分隔 (如 100,100) |
| `--out` | - | 输出图片路径 |
| `--offset` | - | 各电势减去最小值作为偏置（总电势 = 偏置后静电势 + 偏置后赝势） |
| `--show-rf-amp` | - | 显示 RF 幅度图（默认不显示） |
| `--fit` | - | 1D 时对势场做多项式拟合：`2`=二次，`4`=四次；叠加虚线并显示 R²、中心、k2 |
| `--freq` | - | 计算并输出阱频 f_x, f_y, f_z (MHz)，在 --const 点沿各轴拟合总势 |
| `--z_range` | -100,100 | --freq 时 z 轴拟合范围 (μm) |
| `--freq-fit-degree` | 2 | --freq 时拟合阶数：2 或 4 |
| `--freq-n-pts` | 200 | --freq 时每轴拟合采样点数 |
| `--freq-scan` | - | 阱频沿轴扫描：单轴 (x/y/z) 绘曲线；双轴 (如 x,y) 绘 heatmap/3d；指定后不绘势场 |
| `--freq-scan-n` | 50 | --freq-scan 扫描点数：1D 为整数；2D 为逗号分隔 (如 30,30) |
| `--smooth-axes` | z | 势场平滑方向 (如 x,y,z 或 z)；指定 none 关闭滤波 |
| `--smooth-sg` | 11,3 | Savitzky-Golay 参数：窗口长度,多项式阶数，逗号分隔 |

**注意**：传入负数时（如 `--x_range=-100,100`）需使用 `=` 将值与参数相连，否则解析器可能将 `-100` 识别为新选项。

## 平衡构型求解（Equilibrium Solver）

`equilibrium` 模块用于计算离子晶格平衡位置，流程为：

1. 对总外势进行 3D 四次多项式拟合（`fit_potential_3d_quartic`）
2. 构建总势能 `U_total = U_trap + U_coulomb`
3. 使用 L-BFGS-B 最小化 `U_total` 得到平衡位置

在平衡位置求解后，还可继续：

- 构建 Hessian 矩阵（`total`、`trap`、`coulomb`）
- 计算声子模式（质量加权动力学矩阵的本征值/本征向量）
- 使用 NumPy 风格 slice 在 Hessian 子空间上分析（支持逗号并集，如 `:`、`0:10`、`::3`、`5`、`0::3,2::3`）
- 绘制 Hessian 热力图与声子频谱

能量单位统一为 **eV**。同时外势采用统一零点平移（`V_shifted = V_true - V_min_grid`，其中 `V_min_grid` 为格点数据总势最小值），便于与库仑势能量级比较。

```bash
python -m equilibrium.find_equilibrium [options]
```

### 使用示例

```bash
# 默认场配置下求解 40 个离子的平衡构型
python -m equilibrium.find_equilibrium --N 40

# 自定义范围，并从已有 npz 初始化
python -m equilibrium.find_equilibrium --N 120 --x_range=-80,80 --y_range=-30,30 --z_range=-150,150 --init_file saves/rv/traj/cpu/300/t200.0us.npz

# 绘制平衡位置布局图（上 zoy，下 zox）
python -m equilibrium.find_equilibrium --N 80 --plot --color_mode y_pos --plot-out equilibrium/results/equi_pos/80.png

# 在 Hessian 子空间上计算声子并保存频谱/Hessian 结果
# （--*-out 不带路径时会保存到默认目录）
python -m equilibrium.find_equilibrium --N 120 --phonon --hessian-slice 0:90 --plot-phonon-spectrum --plot-phonon-spectrum-out --plot-hessian trap --plot-hessian-out --save-hessian-data

# 绘制指定声子模式的本征向量（zox 平面）
# （mode 序号按频率降序，默认 mode=0）
python -m equilibrium.find_equilibrium --N 120 --hessian-slice 0:90 --plot-mode-vector 3 --plot-mode-vector-arrow-scale 1.8 --plot-mode-vector-out

# 使用 Hessian 自由度并集切片（例如 x+z 子空间）
python -m equilibrium.find_equilibrium --N 120 --phonon --hessian-slice 0::3,2::3 --plot-phonon-spectrum index

# 交互式 mode-vector 查看（不指定输出路径时弹窗）
# 控件：滑条 / 文本框输入后回车 / 键盘左右键翻页
python -m equilibrium.find_equilibrium --N 120 --phonon --hessian-slice 0::3,2::3 --plot-mode-vector 0
```

### 平衡构型参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv` | data/monolithic20241118.csv | 电场 CSV 路径（也支持仅传文件名） |
| `--config` | FieldConfiguration/configs/default.json | 电压配置 JSON 路径（也支持仅传文件名） |
| `--N` | 20 | 离子数量 |
| `--charge` | 1.0 | 单离子电荷（单位 `e`） |
| `--center` | 0,0,0 | 拟合中心 `(x,y,z)`，单位 μm |
| `--x_range` | -50,50 | x 方向拟合/优化范围，单位 μm |
| `--y_range` | -20,20 | y 方向拟合/优化范围，单位 μm |
| `--z_range` | -150,150 | z 方向拟合/优化范围，单位 μm |
| `--fit-n-pts-x` | 100 | 3D 拟合 x 轴采样点数 |
| `--fit-n-pts-y` | 40 | 3D 拟合 y 轴采样点数 |
| `--fit-n-pts-z` | 300 | 3D 拟合 z 轴采样点数 |
| `--softening-um` | 0.001 | 库仑软化长度（μm） |
| `--phonon` | - | 在平衡位置处求解声子模式（对角化动力学矩阵） |
| `--mass-amu` | 135.0 | 声子求解质量（amu，默认 Ba135） |
| `--phonon-print-modes` | 10 | 打印前 N 个声子模（按频率降序） |
| `--hessian-slice` | : | Hessian 自由度子空间切片（支持逗号并集，如 `:`、`0:10`、`::3`、`5`、`0::3,2::3`） |
| `--plot-hessian` | - | 弹出 Hessian 热力图窗口；可选 `total`（默认）/`trap`/`coulomb` |
| `--plot-hessian-out` | - | 保存 Hessian 热力图；不带路径时默认 `equilibrium/results/hessian_plot/{N}_{slice}.png` |
| `--save-hessian-data` | - | 保存 Hessian 数据（`total/trap/coulomb`）为 npz |
| `--hessian-data-out` | equilibrium/results/hessian_data/{N}_{slice}.npz | Hessian 数据输出路径 |
| `--plot-phonon-spectrum` | - | 弹出声子频谱窗口；可选 `frequency`（默认）/`index` |
| `--plot-phonon-spectrum-out` | - | 保存声子频谱；不带路径时默认 `equilibrium/results/spectra/{N}_{slice}.png` |
| `--plot-mode-vector` | - | 弹出指定声子模式本征向量窗口（zox 平面）；可选模式序号（按频率降序），默认 `0`；窗口支持滑条 / 文本框 / 左右键 |
| `--plot-mode-vector-out` | - | 保存模式向量图；不带路径时默认 `equilibrium/results/mode_vector/{N}_{slice}_mode{k}.png` |
| `--plot-mode-vector-arrow-scale` | 1.0 | `--plot-mode-vector` 的箭头长度倍率（>0） |
| `--plot-point-size` | - | `--plot` 与 `--plot-mode-vector` 的散点大小；必须 >0；不指定时各图使用默认大小 |
| `--maxiter` | 500 | 优化最大迭代步数 |
| `--tol` | 1e-10 | 收敛阈值（`ftol`，无量纲） |
| `--seed` | 42 | 随机初始化种子 |
| `--init_file` | - | 可选初值 `.npz`，需含 `r`，形状 `(N,3)`，单位 μm |
| `--plot` | - | 绘制平衡位置布局（上 zoy，下 zox） |
| `--color_mode` | none | `none` / `y_pos` / `v2` / `isotope`（不适用模式会自动降级） |
| `--plot-out` | - | 布局图输出路径；不传则弹窗显示 |
| `--out` | equilibrium/results/equi_pos/{N}.npz | 结果 npz 输出路径（默认按离子数命名） |
| `--smooth-axes` | z | 势场平滑方向（`none` 关闭） |
| `--smooth-sg` | 11,3 | Savitzky-Golay 平滑参数 |

`--plot-*` 与 `--plot-*-out` 为独立开关：
- 指定 `--plot-*` 则弹窗显示
- 指定 `--plot-*-out` 则保存输出
- 同时指定则同一次运行中既显示也保存

**提供 `--*-out` 时的默认命名规则**：
- Hessian/频谱：`{N}_{slice}`（例如 `120_0:90.png`）
- 模式向量图：`{N}_{slice}_mode{k}`（例如 `120_0:90_mode3.png`）

## 命令行参数

```bash
python main.py [options]
```

### 模拟参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 50 | 离子数量；逗号分隔可连跑多场，如 `500,2000,10000`（上一场结束后自动进入下一场；与 `--init_file` 不可同时使用） |
| `--t0` | 0 | 起始时间 (μs) |
| `--time` | ∞ | 模拟终止时刻 (μs)，不传则无限运行 |
| `--alpha` | 0 | 同位素参杂比例；单同位素模式下为该同位素丰度 |
| `--isotope` | - | 单同位素模式：Ba133/Ba134/Ba135/Ba136/Ba137/Ba138；alpha 为该同位素丰度，其余为 Ba135；不指定则使用混合模式 |
| `--device` | cpu | 计算设备：cpu / cuda |
| `--calc_method` | VV | 积分算法：RK4 / VV |
| `--step` | 10 | 每帧积分步数 |
| `--interval` | 1.0 | 帧间隔 (dt 单位) |
| `--batch` | 50 | 每批帧数 |

### 输入 / 配置

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv` | data/monolithic20241118.csv | 电场 CSV；可仅传文件名(如 monolithic20241118.csv)，自动在 data/ 下查找 |
| `--config` | FieldConfiguration/configs/default.json | 电极电压 JSON；可仅传文件名(如 default.json)，自动在 FieldConfiguration/configs/ 下查找 |
| `--init_file` | - | 初始 r0/v0 的 .npz 文件路径，须含 'r'(μm)、'v'(m/s)，形状 (N,3)；t0 优先从 npz 内 t_us 读取（确保 RF 相位衔接），其次从文件名 t{时间}us.npz 解析，否则用 --t0 |
| `--smooth-axes` | z | 势场平滑方向 (如 x,y,z 或 z)；指定 none 关闭滤波；Savitzky-Golay |
| `--smooth-sg` | 11,3 | Savitzky-Golay 参数：窗口长度,多项式阶数，逗号分隔 |

### 绘图

| 参数 | 默认 | 说明 |
|------|------|------|
| `--plot` | - | 启用实时绘图 |
| `--plot_fig` | - | 子图视角，逗号分隔如 zoy,zox；默认 --plot 时为 zoy,zox |
| `--color_mode` | - | 着色：y_pos / v2 / isotope / none；alpha>0 或 --isotope 时默认 isotope |
| `--ion_size` | 5.0 | 散点大小 |
| `--x_range` | 100 | x 方向显示半宽 (μm) |
| `--y_range` | 20 | y 方向显示半宽 (μm) |
| `--z_range` | 200 | z 方向显示半宽 (μm) |
| `--save_final_image` | - | 最后一帧保存路径 |
| `--save_times_us` | - | 需保存轨迹图的时刻 (μs)，逗号分隔如 10,20,30；无窗口，仅保存图片 |
| `--save_fig_dir` | saves/images/traj | 轨迹帧保存根目录，结构为 `{dir}/{device}/{离子数}/t{时间}us.png` |
| `--save_rv_traj_dir` [DIR] | - | 指定时刻 r/v 保存根目录；指定但未传参时默认 saves/rv/traj；结构为 `{dir}/{device}/{离子数}/`；需 --save_times_us；不指定则不保存 |
| `--save_rv_status_dir` [DIR] | - | 最后一帧 r/v 保存根目录；指定但未传参时默认 saves/rv/status；结构为 `{dir}/{device}/{离子数}/`；以最后一帧时间命名；不指定则不保存 |

### 环境变量

- `ISM_DEFAULT_CONFIG`、`ISM_DEFAULT_CSV`：覆盖默认配置路径
- `ISM_DEFAULT_SAVE_FIG_DIR`：覆盖 save_fig_dir 默认值（saves/images/traj）
- `ISM_LOG_LEVEL`：日志级别，DEBUG / INFO / WARNING / ERROR

## 项目结构

```
ism-main/
├── Interface/          # 命令行设置、参数
├── FieldConfiguration/ # 通用常数、电压配置；configs/ 存放 JSON 配置
├── FieldParser/       # CSV 解析、场插值、外场力函数
├── ComputeKernel/     # C++ ionsim、Python 后端
├── Plotter/           # 实时绘图
├── benchmark/         # 性能测试
├── data/              # 电场格点 CSV（默认 data/monolithic20241118.csv）
├── externals/         # 本地 Eigen、pybind11（可选，用于离线构建）
├── field_visualize.py # 电势场可视化入口脚本
├── field_visualize/   # 电势场可视化包
│   ├── core.py        # 单位换算、电势计算、网格构建
│   ├── trap_freq.py   # 阱频计算
│   ├── plots.py       # 势场与阱频扫描绘图
│   └── cli.py         # 参数解析与主流程
├── equilibrium/       # 平衡构型求解模块
│   ├── potential_fit_3d.py  # 3D 四次势场拟合与梯度
│   ├── energy.py      # 外势/库仑/总势能（eV）
│   ├── phonon.py      # Hessian 构建与声子模式求解
│   ├── find_equilibrium.py  # CLI：最小化总势求平衡位置
│   └── results/       # 默认输出根目录
│       ├── equi_pos/      # 平衡构型 npz 输出
│       ├── hessian_data/  # Hessian 数据输出
│       ├── hessian_plot/  # Hessian 热力图输出
│       ├── spectra/       # 声子频谱输出
│       └── mode_vector/   # 声子模式向量图输出
├── main.py            # 入口
└── setup_path.py      # ionsim 路径配置
```

## 性能测试 (Benchmark)

性能测试测量每 10 μs 的模拟耗时（每组 100 μs 取平均）。结果保存至 `benchmark/benchmark_results/`（CSV 与 PNG）。

### Plot 对照（有/无 plot）

使用 CUDA。对比启用与不启用实时绘图时的运行时间。

```bash
python -m benchmark.plot_compare
```

### Device 对照（CPU vs CUDA）

不启用绘图。对比不同离子数下 CPU 与 CUDA 的性能。

```bash
python -m benchmark.device_compare
```

### 输出文件

| 脚本 | CSV | 图表 |
|------|-----|------|
| `plot_compare` | `benchmark/benchmark_results/benchmark_plot_performance.csv` | `benchmark/benchmark_results/benchmark_plot_performance.png` |
| `device_compare` | `benchmark/benchmark_results/benchmark_device_compare.csv` | `benchmark/benchmark_results/benchmark_device_compare.png` |

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## 许可证

见项目根目录。
