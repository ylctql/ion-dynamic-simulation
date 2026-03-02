# ISM-Main：离子阱动力学模拟

[English](README.md) | [中文](README.zh.md)

---

离子阱动力学模拟程序，采用模块化设计，在给定电势场分布下模拟离子晶格的动力学演化，支持 CPU 和 CUDA 加速的库仑力计算。

## 功能特点

- **模块化**：输入、场配置、力解析、计算核心、绘图等模块相互独立
- **积分算法**：支持 RK4 和 Velocity Verlet
- **GPU 加速**：库仑力可选用 CUDA 加速
- **实时绘图**：基于 matplotlib 的实时可视化

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

更多构建选项见 [BUILD.md](BUILD.md)。

### 3. 运行

```bash
python main.py --N 50 --time 10 --plot
```

## 命令行参数

```bash
python main.py [options]
```

### 模拟参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 50 | 离子数量 |
| `--t0` | 0 | 起始时间 (μs) |
| `--time` | ∞ | 模拟时长 (μs)，不传则无限 |
| `--alpha` | 0 | 同位素参杂比例 |
| `--device` | cpu | 计算设备：cpu / cuda |
| `--calc_method` | VV | 积分算法：RK4 / VV |
| `--step` | 10 | 每帧积分步数 |
| `--interval` | 1.0 | 帧间隔 (dt 单位) |
| `--batch` | 50 | 每批帧数 |

### 输入 / 配置

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv` | data/monolithic20241118.csv | 电场格点 CSV 路径 |
| `--config` | FieldConfiguration/default.json | 电极电压配置 JSON |
| `--init_file` | - | 初始 r0/v0 的 .npz 文件路径，须含 'r'(μm)、'v'(m/s)，形状 (N,3) |

### 绘图

| 参数 | 默认 | 说明 |
|------|------|------|
| `--plot` | - | 启用实时绘图 |
| `--plot_fig` | - | 子图视角，逗号分隔如 zoy,zox；默认 --plot 时为 zoy,zox |
| `--color_mode` | - | 着色：y_pos / v2 / isotope / none；alpha>0 时默认 isotope |
| `--ion_size` | 5.0 | 散点大小 |
| `--x_range` | 100 | x 方向显示半宽 (μm) |
| `--y_range` | 20 | y 方向显示半宽 (μm) |
| `--z_range` | 200 | z 方向显示半宽 (μm) |
| `--save_final_image` | - | 最后一帧保存路径 |

### 环境变量

- `ISM_DEFAULT_CONFIG`、`ISM_DEFAULT_CSV`：覆盖默认配置路径
- `ISM_LOG_LEVEL`：日志级别，DEBUG / INFO / WARNING / ERROR

## 项目结构

```
ism-main-v1.0/
├── Interface/          # 命令行设置、参数
├── FieldConfiguration/ # 通用常数、电压配置
├── FieldParser/       # CSV 解析、场插值、外场力函数
├── ComputeKernel/     # C++ ionsim、Python 后端
├── Plotter/           # 实时绘图
├── data/              # 电场格点 CSV（默认 data/monolithic20241118.csv）
├── main.py            # 入口
└── setup_path.py      # ionsim 路径配置
```

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## 许可证

见项目根目录。
