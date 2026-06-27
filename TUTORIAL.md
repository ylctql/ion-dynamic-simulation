# 动力学模拟教程

> 适用于 `main.py` 模块 — 离子阱多体动力学模拟

## 目录

1. [快速上手](#1-快速上手)
2. [单位体系](#2-单位体系)
3. [设定模拟时间](#3-设定模拟时间)
4. [指定势场](#4-指定势场)
5. [初始条件](#5-初始条件)
6. [保存运动状态](#6-保存运动状态)
7. [三种运行模式](#7-三种运行模式)
8. [耗散与 Doppler 冷却](#8-耗散与-doppler-冷却)
9. [同位素掺杂](#9-同位素掺杂)
10. [断点续算](#10-断点续算)
11. [文件格式](#11-文件格式)
12. [完整示例](#12-完整示例)

---

## 1. 快速上手

### 最简运行

```bash
python main.py --N 50 --time 100 --plot
```

50 个 Ba-135+ 离子在默认势场中运行 100 μs，实时显示动画。

### 用谐振势模拟

```bash
python main.py --N 30 --time 50 --trap-freq 2.0 3.0 0.1 --plot
```

无需 CSV 文件，直接指定三轴阱频。

### 无头模式 + 保存轨迹

```bash
python main.py --N 100 --time 200 \
  --save_times_us 50,100,150,200 \
  --save_rv_traj_dir saves/rv/traj \
  --save_fig_dir saves/images
```

运行 200 μs，在 50/100/150/200 μs 处保存离子位置和图像。

---

## 2. 单位体系

所有内部计算使用**无量纲单位**，由 RF 驱动频率导出三个基准量：

| 基准 | 符号 | 定义 | 物理含义 |
|------|------|------|---------|
| 时间 | `dt` | `2 / Omega` | RF 周期的一半 |
| 长度 | `dl` | `(e²/(4πmε₀Ω²))^(1/3)` | 库仑长度尺度 |
| 电压 | `dV` | `(m/e)(dl/dt)²` | 能量等价电压 |

其中 `Omega = 2π × freq_RF`，`freq_RF` 从 JSON 配置文件中的 RF 电极频率自动提取。

### 内部量与物理量的转换

CLI 参数和输出文件统一使用**物理单位**：

| 量 | 内部（无量纲） | 外部（物理） | 转换 |
|----|--------------|-------------|------|
| 位置 | `r_dim` | `r_um` (μm) | `r_um = r_dim × dl × 10⁶` |
| 速度 | `v_dim` | `v_m_s` (m/s) | `v_m_s = v_dim × dl / dt` |
| 时间 | `t_dim` | `t_us` (μs) | `t_us = t_dim × dt × 10⁶` |

**用户不需要直接接触无量纲量**——CLI 输入和 npz 输出都是物理单位。

### dl/dt/dV 的查看

```python
from FieldConfiguration.constants import init_from_config
cfg = init_from_config("FieldConfiguration/configs/default.json")
print(f"dt = {cfg.dt*1e6:.6f} us")
print(f"dl = {cfg.dl*1e6:.6f} um")
print(f"dV = {cfg.dV:.6f} V")
```

对于 `--trap-freq` 模式，dt/dl/dV 由默认的 35.28 MHz RF 频率计算。

---

## 3. 设定模拟时间

### 时间参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--t0` | 0.0 | 起始时间（μs） |
| `--time` | ∞ | 结束时间（μs）。不指定则永远运行 |
| `--interval` | 1.0 | 帧间隔，以 dt 为单位 |
| `--step` | 10 | 每帧间的积分步数 |

### 时间精度

实际的积分时间步长为：

```
物理时间步长 = interval × dt / step
```

- `interval` 越小 → 输出越频繁，但总计算量不变
- `step` 越大 → 每帧之间积分更细，每帧计算时间更长
- 对于 RF 频率 35.28 MHz，`dt ≈ 9 ns`，默认 `interval=1, step=10` 给出每 0.9 ns 输出一帧

### 实际设置建议

| 场景 | `--interval` | `--step` | 说明 |
|------|-------------|---------|------|
| 快速可视化 | 1.0 | 10 | 默认，足够捕捉慢变演化 |
| 高精度轨迹 | 1.0 | 100 | 更多积分步，更精确 |
| 长时间模拟 | 10.0 | 100 | 减少帧数，节省存储 |

---

## 4. 指定势场

`--csv` 和 `--trap-freq` 互斥，二选一。

### 4.1 实验势场 (--csv)

```bash
python main.py --csv data/monolithic20241118.csv \
  --config FieldConfiguration/configs/default.json
```

流程：CSV 电场格点 → E = -∇V → 3D 插值 → `force(r, v, t)`

- `--csv`：电场数据文件。仅传文件名时自动在 `data/` 下查找
- `--config`：电压配置 JSON。仅传文件名时自动在 `FieldConfiguration/configs/` 下查找
- `--smooth-axes x,y,z`（默认）：沿 x,y,z 三轴做 Savitzky-Golay 平滑
- `--smooth-sg 11,3`（默认）：平滑窗口 11 点、3 阶多项式

### 4.2 理想谐振势 (--trap-freq)

```bash
python main.py --trap-freq 2.0 3.0 0.1   # fx fy fz, 单位 MHz
```

对应力函数 `F = -k·r - γ·v`，其中 `k = (ω·dt)²`。无需任何 CSV 或 JSON 文件。

- `fx, fy, fz`：三个轴向的阱频（MHz）
- `fz << fx, fy` 产生沿 z 轴延伸的线性链/zigzag
- 耗散 `γ` 默认 0.1（见[第 8 节](#8-耗散与-doppler-冷却)）

---

## 5. 初始条件

### 5.1 默认（随机位置 + 零速度）

```bash
python main.py --N 50
```

离子被随机放置在一个以原点为中心的区域内，初始速度为零。系统会自动在库仑力 + 势场作用下演化到稳定构型。

### 5.2 从文件加载 (--init_file)

```bash
python main.py --N 50 --init_file snapshot.npz
```

npz 文件要求：

| 字段 | Shape | 单位 | 必需 |
|------|-------|------|------|
| `r` | `(N, 3)` | μm | 是 |
| `v` | `(N, 3)` | m/s | 是 |
| `t_us` | 标量 | μs | 否（推荐） |

如果 npz 中包含 `t_us`（或文件名匹配 `t{X}us.npz`），模拟会从该时刻继续，保持 RF 相位连续。

**典型用途**：

1. 从一个已热化的状态开始（先用较长时间 + 耗散演化到稳态，保存快照，之后从该快照开始）
2. 断点续算
3. 从平衡构型开始（先用 `equilibrium` 模块求解，保存为 npz）

### 5.3 生成初始状态文件

```python
import numpy as np

# 手动构造
N = 50
r = np.zeros((N, 3))
r[:, 2] = np.linspace(-100, 100, N)  # 沿 z 轴均匀分布
v = np.zeros((N, 3))
np.savez("init.npz", r=r, v=v, t_us=0.0)

# 或从碰撞模块的平衡构型生成
from collision_pressure import setup_fit_harmonic, find_equilibrium
fit, _ = setup_fit_harmonic(2e6, 3e6, 0.1e6)
eq = find_equilibrium(fit, N, np.random.default_rng(42))
np.savez("init.npz", r=eq.r_um, v=np.zeros((N, 3)), t_us=0.0)
```

---

## 6. 保存运动状态

### 6.1 在指定时刻保存 (--save_times_us)

```bash
python main.py --N 50 --time 200 \
  --save_times_us 0,50,100,150,200 \
  --save_rv_traj_dir saves/rv/traj \
  --save_fig_dir saves/images
```

`--save_times_us` 支持灵活语法：

```bash
# 枚举
--save_times_us 10,20,30

# 范围（Python range 语义，不含终点）
--save_times_us 'range(0,1000,50)'

# 冒号语法（同 range）
--save_times_us 0:1000:50

# 混合
--save_times_us 50,100:500:100,999.9
```

每个目标时刻保存：
- PNG 图像到 `--save_fig_dir`（如果指定了 `--plot`）
- npz 数据到 `--save_rv_traj_dir`（如果指定）

### 6.2 仅保存最终状态

```bash
python main.py --N 50 --time 100 \
  --save_rv_status_dir saves/rv/status
```

模拟结束后保存最终帧到 `saves/rv/status/{device}/{N}/t{time}us.npz`。

### 6.3 逐帧连续保存 (--continuous-sampling)

```bash
python main.py --N 50 \
  --continuous-sampling \
  --continuous-sampling-frames 500
```

保存每一帧为 npz，输出到 `continuous_sampling/t0{t0}_interval{interval}_step{step}/frame{i}.npz`。

适合需要完整时间序列的场景（如声子谱分析、碰撞初态采样）。

### 6.4 保存最终图像

```bash
python main.py --N 50 --time 100 --save_final_image final.png
```

无头模式下也可以保存最后一帧的图像。

---

## 7. 三种运行模式

### 7.1 实时绘图模式 (--plot)

```bash
python main.py --N 50 --time 100 --plot
```

- 打开 matplotlib 窗口，实时显示离子位置
- 支持 1-2 个子图（`--plot_fig zoy,zox`）
- 按 Ctrl+C 安全停止
- 配合 `--save_times_us` 可同时保存轨迹

### 7.2 无头模式

```bash
python main.py --N 100 --time 500 \
  --save_times_us 100:500:100 \
  --save_rv_traj_dir saves/rv/traj
```

- 无 GUI 窗口，适合服务器/批量运行
- 配合 `--save_times_us` 和 `--save_rv_status_dir` 保存数据
- 可选 `--save_final_image` 保存最终快照图

### 7.3 连续采样模式 (--continuous-sampling)

```bash
python main.py --N 50 --continuous-sampling --continuous-sampling-frames 1000
```

- 忽略 `--plot` 和 `--save_times_us`
- 保存每一帧为 npz
- 达到指定帧数后自动停止
- 输出目录：`continuous_sampling/t0{t0}_interval{interval}_step{step}/`

---

## 8. 耗散与 Doppler 冷却

### 8.1 耗散项

力函数中包含速度阻尼项 `F_damp = -γ·v`：

- **CSV 模式**：γ 来自 JSON 配置文件中的 `g` 字段（默认 0.1）
- **谐振模式**：默认 γ = 0.1（FieldSettings 默认值）

### 8.2 调整耗散强度

在 JSON 配置文件中修改 `g`：

```json
{
  "g": 0.1,
  "dissipation_mode": "scalar"
}
```

- `"scalar"`：各向同性阻尼
- `"vector"`：`g` 为 `[γx, γy, γz]`，各轴独立

### 8.3 物理含义

γ = 0.1 表示中等阻尼——离子在约 10/γ ≈ 100 dt 时间内显著衰减。对于 `dt ≈ 9 ns`，这对应 ~0.9 μs 的弛豫时间。

**热化流程**：先以较大 γ 运行一段时间让离子晶格达到稳态，保存快照，再从该快照以较小 γ（或 γ = 0）运行感兴趣的动力学。

---

## 9. 同位素掺杂

### 单一同位素

```bash
python main.py --N 50 --isotope Ba138
```

所有离子设为指定同位素质量。

### 掺杂（混合同位素）

```bash
python main.py --N 50 --alpha 0.3
```

`--alpha 0.3` 表示掺杂比例为 0.3，离子按 Ba-133 到 Ba-138 六种同位素分配质量。用 `--color_mode isotope` 可在图中区分。

---

## 10. 断点续算

利用 `--init_file` 和 `t_us` 实现：

```bash
# 1. 运行前 100 μs，保存中间状态
python main.py --N 50 --time 100 \
  --save_rv_status_dir saves/rv/status

# 2. 从 100 μs 处继续运行到 200 μs
python main.py --N 50 --time 200 \
  --init_file saves/rv/status/cpu/50/t100.00us.npz
```

加载文件时，`t0` 按以下优先级确定：
1. npz 中的 `t_us` 字段
2. npz 中的 `t` 字段
3. 文件名匹配 `t{X}us.npz`
4. `--t0` 参数

RF 相位从 `t0` 时刻连续计算，确保断点续算的物理一致性。

---

## 11. 文件格式

### npz 输出文件

所有输出 npz 格式统一：

```python
np.savez(path, r=r_um, v=v_m_s, t_us=time_us)
```

| 字段 | Shape | 单位 | 说明 |
|------|-------|------|------|
| `r` | `(N, 3)` | μm | 离子位置 |
| `v` | `(N, 3)` | m/s | 离子速度 |
| `t_us` | 标量 | μs | 当前时刻 |

### 输出路径模板

| 保存方式 | 路径 |
|---------|------|
| `--save_rv_status_dir` | `{dir}/{device}/{N}/t{time}us.npz` |
| `--save_rv_traj_dir` | `{dir}/{device}/{N}/t{target}us.npz` |
| `--save_fig_dir` | `{dir}/{device}/{N}/t{target}us.png` |
| `--continuous-sampling` | `continuous_sampling/t0{t0}_interval{int}_step{step}/frame{i}.npz` |
| `--save_final_image` | 用户指定路径 |

### npz 输入文件 (--init_file)

与输出格式相同，必须含 `r` 和 `v`，推荐含 `t_us`。

---

## 12. 完整示例

### 示例 1：从随机态热化到稳态并保存

```bash
# 热化阶段：1000 μs，较大耗散
python main.py --N 50 --time 1000 \
  --trap-freq 2.0 3.0 0.1 \
  --save_rv_status_dir saves/rv/status

# 得到 saves/rv/status/cpu/50/t1000.00us.npz
```

### 示例 2：从稳态开始运行无耗散动力学

```bash
python main.py --N 50 --time 200 \
  --trap-freq 2.0 3.0 0.1 \
  --init_file saves/rv/status/cpu/50/t1000.00us.npz \
  --save_times_us 0:200:10 \
  --save_rv_traj_dir saves/rv/traj \
  --save_fig_dir saves/images
```

注：热化时 JSON 中 `g` 较大，这里要想无耗散需要把 `g` 改为 0 的配置文件。

### 示例 3：为碰撞模拟生成初态快照

先用连续采样收集热化后的状态序列，然后选取一帧作为碰撞模拟的输入：

```bash
# 热化并连续采样
python main.py --N 20 --time 500 \
  --trap-freq 0.2 3.0 0.1 \
  --continuous-sampling --continuous-sampling-frames 10000

# 从连续采样结果中选取一帧
python -c "
import numpy as np
data = np.load('continuous_sampling/t00.00_interval1.0_step10/frame9999.npz')
np.savez('zigzag_snapshot.npz', r=data['r'], v=data['v'], t_us=data['t_us'])
print(f'Saved snapshot at t={data[\"t_us\"]:.2f} us')
print(f'  r shape: {data[\"r\"].shape}')
print(f'  v range: [{data[\"v\"].min():.4f}, {data[\"v\"].max():.4f}] m/s')
"

# 在碰撞模拟中使用
python -m collision_pressure simulate \
  --trap-freq 0.2 3.0 0.1 \
  --init-file zigzag_snapshot.npz \
  --n-simulations 100 \
  --detector zigzag --flip-axis 0 \
  --gamma-damping 1e5
```

### 示例 4：大批量无头模拟

```bash
# 用实验势场，运行 500 μs，每 10 μs 保存一次
python main.py --N 200 --time 500 \
  --csv data/monolithic20241118.csv \
  --config FieldConfiguration/configs/default.json \
  --interval 5.0 --step 50 \
  --save_times_us 0:510:10 \
  --save_rv_traj_dir results/rv \
  --device cuda \
  --save_final_image results/final.png
```

### 示例 5：可视化展示

```bash
python main.py --N 50 --time 50 --plot \
  --trap-freq 2.0 3.0 0.1 \
  --plot_fig zoy,zox \
  --color_mode y_pos \
  --z_range 150 --y_range 50
```

---

## CLI 参数速查

### 基本参数

```
离子系统:
  --N N                     离子数量 (默认 50)
  --init_file PATH          初始状态 npz 文件
  --alpha FLOAT             同位素掺杂比例
  --isotope SPECIES         单一同位素: Ba133/Ba134/Ba135/Ba136/Ba137/Ba138

势场 (二选一):
  --csv PATH                电场 CSV 文件
  --trap-freq FX FY FZ      谐振阱频 (MHz)
  --config PATH             电压配置 JSON

时间:
  --t0 FLOAT                起始时间 (μs, 默认 0)
  --time FLOAT              结束时间 (μs, 不指定则无限)
  --interval FLOAT          帧间隔 (dt 单位, 默认 1.0)
  --step INT                每帧积分步数 (默认 10)
  --batch INT               每批帧数 (默认 50)
```

### 可视化

```
  --plot                    启用实时绘图
  --plot_fig VIEWS          子图视图, 如 zoy,zox
  --color_mode MODE         着色: y_pos / v2 / isotope / none
  --ion_size FLOAT          点大小 (默认 5.0)
  --x/y/z_range FLOAT       显示半宽 (μm)
  --bilayer                 双层模式
```

### 数据保存

```
  --save_times_us TIMES     保存时刻 (μs), 支持 range/冒号/混合语法
  --save_fig_dir DIR        图像保存目录
  --save_rv_traj_dir DIR    轨迹 r/v 保存目录
  --save_rv_status_dir DIR  最终状态保存目录
  --save_final_image PATH   最终帧图像路径
  --continuous-sampling      逐帧连续保存
  --continuous-sampling-frames N  连续保存帧数
```

### 计算

```
  --device DEVICE           cpu / cuda (默认 cpu)
  --calc_method METHOD      RK4 / VV (默认 VV)
  --smooth-axes AXES        场平滑方向 (默认 x,y,z)
  --smooth-sg WIN,POLY      Savitzky-Golay 参数 (默认 11,3)
```
