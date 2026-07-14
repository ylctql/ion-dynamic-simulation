# param_scan —— poly-potential x²/x⁴ 系数扫描 + σ_y 热力图

对 `softmode.json` 的 `x²`/`x⁴` 多项式系数做二维网格扫描：每组组合跑一次 N 离子动力学，
按 **发散 → 逃逸 → 塌缩 → 超时** 优先级判定结局，存结束位置 `.npy`、维护 `scan.log`，
最后由 log 画出 (x², x⁴, σ_y) 热力图。

> **相关**：势场来源见 [`docs/poly_potential.md`](poly_potential.md)；softmode.json 在
> `FieldConfiguration/configs/poly_potential/softmode.json`（当前 `x²=0.01`、`x⁴=0.005`，
> 其余 `y²=1.18`/`z²=0.002`/`z⁴=0.000917`/`scale_um=100` 固定）。

## 设计要点

- **固定其余系数**：每轮从 softmode.json 读基底系数，仅改 `x²`（`coeffs[2,0,0]`）与 `x⁴`（`coeffs[4,0,0]`），
  经 `fit_result_from_coeff_map` + `build_poly_potential_force` 重建力（不经 JSON I/O）。
- **复用动力学基础设施**：`main.py::_create_backend_and_start` 启动 backend 子进程，
  backend 在 `time=50µs` **自动 STOP**；发散/逃逸/σ_y 达标则由本模块消费者提前 STOP。`main.py`/`backend.py` 不改动。
- **结局判定**（优先级从高到低，见 `scan.py`）：
  - **发散 `diverged`** —— 后端子进程异常退出（C++ 抛错/段错误，`exitcode≠0`，`_get_from_queue` 抛
    `RuntimeError`），或某帧坐标出现 `NaN/inf`。势场囚禁能力不足、离子飞向无穷远时常引发。
  - **逃逸 `escaped`** —— 某帧任一离子坐标超出 **±600µm**（`ESCAPE_LIM_UM`，逐帧检测）。
    即便 50µs 内未触发数值发散，晶格已明显散开即判逃逸；log 记录逃逸方向（如 `+x(3),-y(1)`）。
  - **塌缩 `reached`** —— `σ_y < 0.0005µm`（提前 STOP，记录跨越时刻）。
  - **超时 `timeout`** —— 跑满 50µs 未达标且无逃逸。
- **σ_y 定义**：`np.std(r[:,1])`（y 方向总体标准差，µm），与 `Plotter/dataplot.py:513` 完全一致。
- **种子化初态**：每轮用 `np.random.default_rng(seed)` 生成均匀随机盒初态，seed 记入 log（可复现）；
  `--fixed-seed` 可让整网格共用一个初态做受控扫描。
- **动力学与绘图解耦**：`plot` 子命令仅读 `scan.log`，不依赖 ionsim/.npy。

## 用法

### 1) 扫描

```bash
python -m param_scan scan --x2 0.005,0.05,8 --x4 0.001,0.02,6
```

- `--x2`/`--x4`：扫描轴，**两种都支持**——`min,max,n`（→ linspace）或 `v1,v2,...`（显式列表）。
- `--out-dir`：输出目录（默认 `param_scan/results`）。`scan.log` 在该目录下，
  各轮 `.npy` 在其 `positions/` 子目录下（`positions/{idx:04d}.npy`）。
- log 默认**追加**：`scan.log` 已有内容则仅追加行（续跑同一网格时，已完成的序号会被
  跳过且不重复记 log）。`--new-log` 截断 log 重开一轮并强制重跑所有点（扫描**不同参数集**
  时用，避免同序号与旧数据冲突）。`--force` 仅覆盖已存在 `.npy`（log 仍追加）。
- `--base-seed N` / `--fixed-seed` / `--init-range X,Y,Z`。

易改的物理/计算常量在 `param_scan/scan.py` 顶部（`N_IONS`/`DEVICE`/`CALC_METHOD`/
`CONFIG_JSON`/`TIME_MAX_US`/`SIGMA_Y_THRESH_UM`/`ESCAPE_LIM_UM`/`INIT_RANGE_UM`/`GAMMA`/`STEP`/`INTERVAL`/`BATCH`）。

### 2) 热力图（与动力学解耦）

```bash
python -m param_scan plot --log param_scan/results/scan.log --out heat.png
# 颜色列可选 sigma_y_final_um / sim_time_us / reached_threshold / wall_time_s；σ_y 跨数量级时加 --log-color
python -m param_scan plot --log ... --value-col reached_threshold --out reach.png
```

## 输出格式

`scan.log`（**列以 TAB 分隔**，非逗号）：

| 列 | 含义 |
|----|------|
| `run_idx` | 运行序号（1 起，= `.npy` 文件名） |
| `x2_coeff`,`x4_coeff` | 当轮 x²/x⁴ 系数 (V) |
| `seed` | 当轮随机种子 |
| `reached_threshold` | 1/0，50µs 内是否 σ_y<阈值（escaped/diverged/error 恒为 0） |
| `status` | 结局状态码：`reached`/`timeout`/`escaped`/`diverged`/`error` |
| `sim_time_us` | reached→跨越时刻；escaped→逃逸时刻；timeout→50.0；diverged→最后帧时刻；error→0 |
| `sigma_y_final_um` | 对应时刻 σ_y (µm)；diverged/error 为 `nan` |
| `escape` | 逃逸方向描述（如 `+x(3),-y(1)`，方向后括号为离子数）；无逃逸为 `-` |
| `wall_time_s` | 该轮挂钟耗时 (s) |

> **分隔符/列集变更**：列间为 TAB。若旧版（逗号或旧列集）`scan.log` 已存在，
> `write_header` 会检测到表头不一致并抛错——用 `--new-log` 截断重开或删除旧 log 后重跑。

`positions/{idx:04d}.npy`：模拟结束位置 `(N,3)`，单位 µm（`positions/` 为 `out-dir` 子目录；
diverged 时为最后帧，可能含 `NaN`）。

## 验证

- **纯 Python 单测**（conda base，不依赖 ionsim）：`pytest tests/test_param_scan.py -q`
  覆盖 `parse_axis_spec`、`sigma_y_um`、`logio` 往返、`heatmap.load_grid`。
- **管线冒烟**（需 ionsim+CUDA）：临时改小 `N_IONS`/`TIME_MAX_US`，跑 1 点
  （`--x2 0.01 --x4 0.005`）确认 scan→`.npy`+log→plot 全链路。
- **真实单点计时**：`N_IONS=6000` 单点跑，据此估算整网格耗时。
