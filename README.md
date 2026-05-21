# ISM-Main

Ion trap dynamics simulation with modular architecture.

[English](#english) | [中文](#中文)

---

<a id="english"></a>

Ion trap dynamics simulation program. Simulates ion crystal evolution under given electric potentials or ideal harmonic traps. Supports RK4 / Velocity Verlet integration, CUDA-accelerated Coulomb force, real-time visualization, equilibrium solving, ion imaging simulation, and collision-pressure estimation.

## Quick Start

```bash
pip install -e .                    # Install dependencies
python build.py                     # Build C++ extension (tries CUDA, falls back to CPU)
python main.py --N 50 --time 10 --plot
```

**Platform**: Linux / macOS (uses `multiprocessing` with `fork`)

**Requirements**: Python ≥ 3.10, CMake ≥ 3.18, Eigen 3.4 + pybind11 (auto-fetched). Optional: CUDA Toolkit.

Offline build: place Eigen/pybind11 in `externals/` or `python build.py --local /path/to/externals`. See [BUILD.md](BUILD.md) for details.

## Potential Field Configuration

The external trap potential can be specified in two mutually exclusive ways:

### Option A: CSV + Config (real trap)

```bash
python main.py --N 50 --time 10 --csv data/monolithic20241118.csv --config FieldConfiguration/configs/default.json
```

Reads electrode potentials from CSV grid data and voltage configuration from JSON. `--csv` supports filename-only (auto-searches `data/`). Optional: `--smooth-axes z --smooth-sg 11,3` for Savitzky-Golay smoothing.

### Option B: Harmonic trap (--trap-freq)

```bash
python main.py --N 50 --time 10 --trap-freq 1.0 5.0 0.2
```

Uses an ideal quadratic potential with secular frequencies `fx fy fz` (MHz). `--config` still applies for the dimensionless system (Omega/dl/dt/dV). No CSV needed.

These two options are **mutually exclusive** (`--csv` xor `--trap-freq`).

## Modules

| Module | Description |
|--------|-------------|
| `main.py` | Dynamics simulation entry point |
| `field_visualize/` | Electric potential visualization (1D/2D, heatmap/3D, trap frequencies) |
| `equilibrium/` | Equilibrium position solver + phonon mode analysis |
| `ImgSimulation/` | Single-frame CCD/CMOS-style ion image simulation |
| `collision_pressure/` | H2 collision-pressure estimation via structural reconfiguration |
| `FieldParser/` | CSV reader, field interpolation, force functions |
| `FieldConfiguration/` | Constants, voltage config loader |
| `ComputeKernel/` | C++ ionsim integration kernel (RK4/VV, optional CUDA) |
| `Plotter/` | Real-time matplotlib visualization |
| `Interface/` | CLI argument parsing |

## field_visualize

Visualize electric potentials (static / RF pseudopotential / total) from CSV + config.

```bash
python field_visualize.py                          # 1D along x
python field_visualize.py --vary x,y --mode heatmap # 2D heatmap
python field_visualize.py --freq                    # Trap frequencies (MHz)
python field_visualize.py --freq-scan z             # Frequency scan along z
```

## equilibrium

Find ion crystal equilibrium positions via 3D polynomial fit + L-BFGS-B minimization. Supports phonon mode analysis, Hessian visualization, and mode-vector plots.

```bash
python -m equilibrium.find_equilibrium --N 40                # Basic equilibrium
python -m equilibrium.find_equilibrium --N 80 --phonon       # With phonon modes
python -m equilibrium.find_equilibrium --N 120 --phonon --hessian-slice 0:90 --plot-phonon-spectrum --plot-hessian trap
```

Both `--csv` and `--trap-freq` are supported (same as `main.py`). Run with `--help` for all options.

## ImgSimulation

Simulates a single integrated CCD/CMOS image: trajectory → Gaussian beam exposure → PSF blur → sensor noise.

```bash
python -m ImgSimulation ImgSimulation/configs/example_ion_image.json -o out.png
```

```python
from ImgSimulation.api import run_ion_image_from_json_file
img = run_ion_image_from_json_file("ImgSimulation/configs/example_ion_image.json")
```

## collision_pressure

Estimates cryogenic background pressure by simulating H2 elastic collisions with ion crystals and detecting structural reconfiguration.

### Configuration library

Builds a library of topologically distinct equilibrium configurations via multi-start optimization + Delaunay-based topology merging:

```bash
python -m collision_pressure --n-ions 54 --n-scans 200          # CSV + config (default)
python -m collision_pressure --n-ions 30 --trap-freq 1.0 5.0 0.2 # Harmonic trap
```

### Visualization

```bash
python -m collision_pressure.visualize_configs  # Plot top/bottom 5 configurations
```

Run with `--help` for all options.

## CLI Reference (main.py)

For complete option lists, run `python main.py --help`. Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--N` | 50 | Ion count; comma-separated for sequential runs |
| `--time` | ∞ | Simulation end time (μs) |
| `--csv` | *(default CSV)* | Electric field CSV (`--csv` and `--trap-freq` are mutually exclusive) |
| `--trap-freq FX FY FZ` | - | Harmonic trap frequencies (MHz) |
| `--config` | *(default JSON)* | Voltage/dimensionless system config |
| `--init_file` | - | Initial state `.npz` (r in μm, v in m/s) |
| `--device` | cpu | Compute device: cpu / cuda |
| `--calc_method` | VV | Integration: RK4 / VV |
| `--plot` | - | Enable real-time plotting |
| `--save_times_us` | - | Save trajectory frames at specified times |

### Environment Variables

`ISM_DEFAULT_CONFIG`, `ISM_DEFAULT_CSV`, `ISM_DEFAULT_SAVE_FIG_DIR`, `ISM_LOG_LEVEL`

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Project Structure

```
ism-main/
├── main.py                # Entry point
├── build.py               # C++ extension build script
├── Interface/             # CLI, parameters
├── FieldConfiguration/    # Constants, voltage config loader; configs/
├── FieldParser/           # CSV reader, field interpolation, force
├── ComputeKernel/         # C++ ionsim, Python backend
├── Plotter/               # Real-time visualization
├── field_visualize/       # Potential visualization
├── equilibrium/           # Equilibrium solver, phonon analysis
├── ImgSimulation/         # Single-frame ion image simulation
├── collision_pressure/    # H2 collision-pressure estimation
├── benchmark/             # Performance benchmarks
├── data/                  # Electric field CSV files
└── externals/             # Local Eigen/pybind11 (optional)
```

## License

MIT

---

<a id="中文"></a>

离子阱动力学模拟程序。在给定电势场或理想谐振势下模拟离子晶格演化，支持 RK4 / Velocity Verlet 积分、CUDA 加速库仑力、实时可视化、平衡构型求解、离子成像模拟及碰撞压强估算。

## 快速开始

```bash
pip install -e .                    # 安装依赖
python build.py                     # 编译 C++ 扩展（自动尝试 CUDA，失败回退 CPU）
python main.py --N 50 --time 10 --plot
```

**运行平台**：Linux / macOS（使用 `fork` 创建子进程）

**依赖**：Python ≥ 3.10、CMake ≥ 3.18、Eigen 3.4 + pybind11（CMake 自动获取）。可选：CUDA Toolkit。

离线构建：将 Eigen/pybind11 放入 `externals/` 或 `python build.py --local /path/to/externals`。详见 [BUILD.md](BUILD.md)。

## 势场指定方式

外场势有两种互斥的指定方式：

### 方式 A：CSV + Config（实际陷阱）

```bash
python main.py --N 50 --time 10 --csv data/monolithic20241118.csv --config FieldConfiguration/configs/default.json
```

从 CSV 格点数据和 JSON 电压配置读取电极势场。`--csv` 支持仅传文件名（自动在 `data/` 下查找）。可选 `--smooth-axes z --smooth-sg 11,3` 进行 Savitzky-Golay 平滑。

### 方式 B：谐振势（--trap-freq）

```bash
python main.py --N 50 --time 10 --trap-freq 1.0 5.0 0.2
```

使用理想二次势，指定三个方向的阱频 `fx fy fz`（MHz）。`--config` 仍用于无量纲系统参数（Omega/dl/dt/dV），无需 CSV。

两种方式**互斥**（`--csv` 与 `--trap-freq` 只能选其一）。

## 模块概览

| 模块 | 说明 |
|------|------|
| `main.py` | 动力学模拟入口 |
| `field_visualize/` | 电势场可视化（1D/2D、热力图/3D、阱频计算） |
| `equilibrium/` | 平衡构型求解 + 声子模式分析 |
| `ImgSimulation/` | 单帧类 CCD/CMOS 离子成像模拟 |
| `collision_pressure/` | H2 弹性碰撞压强估算（结构重构检测） |
| `FieldParser/` | CSV 解析、场插值、力函数 |
| `FieldConfiguration/` | 无量纲常数、电压配置加载 |
| `ComputeKernel/` | C++ ionsim 积分核心（RK4/VV，可选 CUDA） |
| `Plotter/` | matplotlib 实时可视化 |
| `Interface/` | 命令行参数解析 |

## field_visualize

可视化电场 CSV + 电压配置下的电势分布（静电势 / RF 赝势 / 总电势）。

```bash
python field_visualize.py                          # 1D 沿 x
python field_visualize.py --vary x,y --mode heatmap # 2D 热力图
python field_visualize.py --freq                    # 计算阱频 (MHz)
python field_visualize.py --freq-scan z             # 阱频沿 z 扫描
```

## equilibrium（平衡构型求解）

通过 3D 多项式拟合 + L-BFGS-B 最小化求解离子晶格平衡位置。支持声子模式分析、Hessian 可视化、模式向量图。

```bash
python -m equilibrium.find_equilibrium --N 40                # 基本求解
python -m equilibrium.find_equilibrium --N 80 --phonon       # 含声子
python -m equilibrium.find_equilibrium --N 120 --phonon --hessian-slice 0:90 --plot-phonon-spectrum --plot-hessian trap
```

支持 `--csv` 和 `--trap-freq`（与 `main.py` 一致）。运行 `--help` 查看全部参数。

## ImgSimulation（离子成像）

模拟单帧 CCD/CMOS 图像：轨迹 → 高斯光束曝光 → PSF 模糊 → 传感器噪声。

```bash
python -m ImgSimulation ImgSimulation/configs/example_ion_image.json -o out.png
```

```python
from ImgSimulation.api import run_ion_image_from_json_file
img = run_ion_image_from_json_file("ImgSimulation/configs/example_ion_image.json")
```

## collision_pressure（碰撞压强估算）

通过模拟 H2 弹性碰撞与离子晶格的相互作用，检测结构重构事件来估算低温区背景气压。

### 构型库构建

基于 Delaunay 拓扑表征的多起点优化，构建拓扑不等价的平衡构型库：

```bash
python -m collision_pressure --n-ions 54 --n-scans 200          # CSV + config（默认）
python -m collision_pressure --n-ions 30 --trap-freq 1.0 5.0 0.2 # 谐振势
```

### 构型可视化

```bash
python -m collision_pressure.visualize_configs  # 绘制能量最高/最低的 5 种构型
```

运行 `--help` 查看全部参数。

## 命令行参数（main.py）

完整参数列表请运行 `python main.py --help`。主要参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 50 | 离子数；逗号分隔可连跑多场 |
| `--time` | ∞ | 模拟终止时刻 (μs) |
| `--csv` | *(默认 CSV)* | 电场 CSV（与 `--trap-freq` 互斥） |
| `--trap-freq FX FY FZ` | - | 谐振势阱频 (MHz) |
| `--config` | *(默认 JSON)* | 电压/无量纲系统配置 |
| `--init_file` | - | 初态 `.npz`（r 单位 μm，v 单位 m/s） |
| `--device` | cpu | 计算设备：cpu / cuda |
| `--calc_method` | VV | 积分算法：RK4 / VV |
| `--plot` | - | 启用实时绘图 |
| `--save_times_us` | - | 指定保存轨迹帧的时刻 |

### 环境变量

`ISM_DEFAULT_CONFIG`、`ISM_DEFAULT_CSV`、`ISM_DEFAULT_SAVE_FIG_DIR`、`ISM_LOG_LEVEL`

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## 项目结构

```
ism-main/
├── main.py                # 入口
├── build.py               # C++ 扩展构建脚本
├── Interface/             # 命令行、参数
├── FieldConfiguration/    # 无量纲常数、电压配置；configs/
├── FieldParser/           # CSV 解析、场插值、力函数
├── ComputeKernel/         # C++ ionsim、Python 后端
├── Plotter/               # 实时可视化
├── field_visualize/       # 电势场可视化
├── equilibrium/           # 平衡构型求解、声子分析
├── ImgSimulation/         # 单帧离子成像模拟
├── collision_pressure/    # H2 碰撞压强估算
├── benchmark/             # 性能测试
├── data/                  # 电场 CSV 文件
└── externals/             # 本地 Eigen/pybind11（可选）
```

## 许可证

MIT
