# 多图批处理（单卡）

## 策略概要

- **默认单进程**（`max_workers=1`）：单 GPU 上多进程 CUDA 易争用显存、上下文切换，吞吐未必更好。
- **快路径（默认）**：`share_dynamics` 为默认（`null`）或 `true` 时只算 **一次动力学** + **一次曝光积分**（`integrate_exposure_xy_um`），后面 **不再** 重复曝光。随后对每张图：若 **PSF 全员相同** 则 **只做一次** `apply_gaussian_psf`，再对每张图分别加噪与归一化；若 **PSF 不同** 则对共享的曝光图分别卷积。真正的大头通常是 **ionsim 轨迹**；曝光与 PSF 已在批内尽量只做必要次数。
- **跨多次脚本/会话复用曝光**：若希望把「动力学+曝光」存下来，只对 PSF/噪声做扫描，可用 `ImgSimulation.pipeline.compute_exposure_map` 得到曝光数组，再反复调用 `render_from_exposure`（见下文「高级」）。
- **慢路径**：`share_dynamics: false`（或 API 里 `share_dynamics=False`）时对每张图完整跑 `render_single_frame`（例如要与独立多次运行逐比特对齐、或调试）。
- **进度日志**：批处理默认 **忽略** `log_interval_sim_us`（等价于单次 ionsim 腿），避免分段积分重复创建 CUDA Coulomb 上下文。若需与单帧相同的分段打印，设 `allow_batch_progress_log=True`（JSON：`batch.allow_batch_progress_log`）。
- **多进程**（`max_workers>1`）：仅在 **`share_dynamics: false`** 的慢路径下会用到；且要求 **`force` 可被 pickle**。默认快路径下单进程完成；阱场力通常请保持 `max_workers=1`。
- **多卡**：为每个进程设置不同的 `CUDA_VISIBLE_DEVICES`，不在本模块内自动分配。

## Python API

- `ImgSimulation.api.render_batch` — 参数见源码 docstring。
- `ImgSimulation.json_config.IonImageJsonBundle.call_run_batch` — 由 JSON `batch` 块驱动。

## JSON

可选顶层 **`batch`**：

| 字段 | 含义 |
|------|------|
| `seeds` | 整数列表，每张图的默认 RNG seed（必填，非空）；可被该项 `noise_overrides[].seed` 覆盖 |
| `noise_overrides` | 可选；与 `seeds` 等长的对象数组；每项在根 `noise` 上合并字段，键可为 `shot_factor`、`shot_scale`、`readout_factor`、`readout_sigma`、`bg_offset`、`seed` |
| `psf_sigma_px` | 可选；与 `seeds` 等长的数组，每项为高斯 PSF 的 sigma（像素）；省略则全部使用根 `imaging.psf_sigma_px` |
| `figure_paths` | 与 `seeds` 等长的输出 PNG 路径列表（可选；相对路径相对 JSON 目录） |
| `max_workers` | 默认 `1` |
| `share_dynamics` | `null`/省略为自动；`true`/`false` 强制 |
| `allow_batch_progress_log` | 默认 `false` |
| `profile` | 若 `true`，运行期间设置 `IMG_SIM_PROFILE=1`（与 `simulation.log_interval_sim_us` 独立） |

## CLI

单份完整 JSON（与以前相同）：

```bash
python -m ImgSimulation path/to/config.json --no-show --profile
```

**动力学 + 成像 两份 JSON**（顺序：先动力学，后成像；与 `load_ion_image_merged` 一致）：

```bash
python -m ImgSimulation path/to/dynamics.json path/to/imaging.json --no-show
```

示例：`ImgSimulation/configs/example_dynamics.json` 与 `example_imaging.json`（由 `example_ion_image.json` 拆分，语义等价）。

若 JSON 含 `batch`，则执行 `call_run_batch`（写多张图）；否则仍为单帧 `call_run_ion_image`。`batch` 只能出现在 **imaging** JSON 中。

**仅导出成像平面轨迹 NPZ**（投影不依赖 `camera`/`beam`）：

```bash
# 单文件：完整 JSON，或仅 dynamics JSON
python -m ImgSimulation path/to/ion_image.json --export-plane-npz out/plane.npz --no-show
python -m ImgSimulation path/to/dynamics.json --export-plane-npz out/plane.npz --no-show

# 双文件：只需动力学 JSON 参与积分；imaging 用于校验（不得含 batch）
python -m ImgSimulation path/to/dynamics.json path/to/imaging.json --export-plane-npz out/plane.npz --no-show
```

相对路径相对于 **`ImgSimulation` 包目录**（与 `api.py` 同级的 `ImgSimulation/` 文件夹），不是相对于各 JSON 所在目录。`--export-plane-npz` 不可与 imaging JSON 中的 `batch` 同时使用。

**`pos_zx/` 平均位置 NPY**（与平面轨迹 NPZ 成对、默认打开）：凡写入平面轨迹 `*.npz`（经 `save_plane_trajectory_npz`），包内会同步写 `ImgSimulation/pos_zx/<与 NPZ 同 stem>.npy`：形状 `(N, 2)` float64，为曝光时间窗内 `xy_stack` 对时间的平均（z、x，µm，与 `xy_stack` 约定一致）。同 stem 的 NPZ 若写在不同目录，会覆盖同一 `pos_zx` 文件。在代码中可传 `write_mean_pos_zx=False` 关闭。

**从 `traj_zx` 批量导出 NPY + meta**（仅需成像 JSON；与 `--export-plane-npz` / `--export-plane-batch` 互斥）：

对每个平面轨迹 `*.npz`（默认扫描 `ImgSimulation/traj_zx/`），按成像 JSON 的 `batch.seeds`（无 `batch` 时仅输出 `_0001`）生成：

- `ImgSimulation/Imgs/<stem>/<stem>_0001.npy` …
- `ImgSimulation/meta/<stem>/<stem>_0001.json` …（`noise`、生效 `psf_sigma_px`、`imaging_json` / `traj_npz` 等追溯字段）

```bash
python -m ImgSimulation path/to/imaging.json --export-traj-ion-npy
```

可选：`--traj-dir`、`--imgs-root`、`--meta-root`（相对路径均相对于 **`ImgSimulation` 包目录**）、`--traj-pattern`（默认 `*.npz`）、`--dry-run`。

Python API：`ImgSimulation.api.export_ion_npy_from_traj_dir`、`default_traj_zx_paths`。

## 性能观测

- 环境变量 **`IMG_SIM_PROFILE=1`**（或 CLI `--profile`）：在 `render_single_frame` / 批处理快路径上打印各阶段累计墙钟时间（trajectory、exposure、psf、noise 等）。
- 对比脚本：`python tools/benchmark_img_batch.py`（噪声快路径 vs 循环单帧）。

## 流水线扩展（高级）

以下函数在 `ImgSimulation.pipeline` 中：

- `compute_exposure_trajectory` — 仅动力学，返回曝光窗 `r_list` 与 `dt_real_s`
- `r_list_to_r_plane_lists` — 维无量纲轨迹 → 成像平面 µm；返回 `(r_plane_list, mean_plane_px, xy_stack)`，其中 `xy_stack` 为 ``(T, N, 2)`` 连续数组，可直接传给 `integrate_exposure_xy_um` 以避免再 `stack`。
- `compute_exposure_map` — **一次**动力学 + 投影 + 曝光积分，得到 **PSF 之前** 的 `exposure` 数组（形状 `(h, l)`）
- `render_from_exposure` — 给定 `exposure`，只做 PSF + 可选传感器噪声 + 归一化（无动力学）
- `render_from_r_plane_lists` — 由平面轨迹列表完成曝光 + PSF + 噪声 + 归一化

示例（先付动力学成本，再扫噪声/PSF）：

```python
from ImgSimulation.pipeline import compute_exposure_map, render_from_exposure

exposure = compute_exposure_map(
    cfg, force, r0, v0, charge, mass, camera, beam, integ,
    use_cuda=True,
)
for sigma in (1.0, 1.5, 2.0):
    for noise in noise_list:
        img = render_from_exposure(exposure, sigma, noise, apply_sensor_noise=True)
```

## 成像平面轨迹 NPZ（动力学与成像解耦）

仅持久化曝光窗内、已投影到 **成像平面** 的二维轨迹（µm）与时间步长，成像阶段不再依赖 ionsim / 三维 `r_list`。

**契约**（`ImgSimulation.plane_trajectory_io`）：

| 键 / 字段 | 含义 |
|-----------|------|
| `xy_stack` | `float64`，形状 `(T+1, N, 2)`：第 0 维为时间步；最后一维为 **(仿真 z, 仿真 x)** 的 µm 坐标，即 **列=z、行=x**（约定名 `zox_col_z_row_x_um`，与 `integrate_exposure_xy_um` 一致）。 |
| `dt_real_s` | 标量 `float64`，相邻样本的统一物理时间步长（秒），与 `compute_exposure_trajectory` 返回值一致。 |
| `schema_version` | 整数，当前为 `1`。 |
| `convention` | 字节串 / meta 中为 `zox_col_z_row_x_um`。 |
| `meta_json` | 可选 UTF-8 JSON：可含 `source`、`n_ions`、`n_time` 等；加载时与固定键合并。 |
| `dynamics_provenance` | （推荐）由 `export_plane_trajectory_from_simulation(..., dynamics_json_path=...)` 或 CLI `--export-plane-npz` 自动写入：含 **`dynamics_json_content`**（JSON 全文快照）、**`field_setup.field_config_content`**（当场 `paths.field_config` 指向文件的解析内容或 UTF-8 原文）、**`field_setup.field_csv`**（`basename`、`path_as_in_json`、`resolved_path`）。 |

单文件 monolith 导出时 `dynamics_json_path` 指向该完整 JSON，故 `dynamics_json_content` 内也会包含成像段（与磁盘文件一致快照）。

**API 提要**：

- `save_plane_trajectory_npz` / `load_plane_trajectory_npz` — 写入 / 读取 `PlaneTrajectoryRecord`。
- `build_dynamics_provenance_meta` — 从给定动力学 JSON 路径生成 `dynamics_provenance` 字典（全文快照 + `field_config_content` + `field_csv` 信息）；也可由 `export_plane_trajectory_from_simulation(..., dynamics_json_path=...)` 自动合并进 `meta`。
- `export_plane_trajectory_from_simulation` — `compute_exposure_trajectory` → `r_list_to_r_plane_lists(..., return_mean_plane_px=False)` → 存 NPZ（投影不依赖真实 `camera`）；建议传入 `dynamics_json_path` 以写入完整溯源元数据。
- `render_from_plane_trajectory_file` — 加载 NPZ → `render_from_r_plane_lists`（曝光 + PSF + 噪声）。
- `stack_to_r_xy_list` / `r_xy_list_to_stack` — 与 `integrate_exposure_xy_um` 的列表格式互转。

平面轨迹与拆分 JSON 相关符号请从 **`ImgSimulation.api`** 导入（例如 `from ImgSimulation.api import load_plane_trajectory_npz`）。
