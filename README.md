# ISM-Main: Ion Trap Dynamics Simulation

[English](README.md) | [中文](README.zh.md)

---

Ion trap dynamics simulation with modular architecture. Simulates ion crystal dynamics under given electric field distributions, with support for CPU and CUDA-accelerated Coulomb force computation.

## Features

- **Modular design**: Input, field configuration, force parsing, compute kernel, and plotting are decoupled
- **RK4 & Velocity Verlet**: Configurable integration methods
- **CPU / CUDA**: Optional GPU acceleration for Coulomb force
- **Real-time plotting**: Live visualization with matplotlib
- **Field visualization**: Standalone tool for electric potential distribution (static, RF pseudopotential, total) in 1D or 2D (heatmap / 3D surface)

## Requirements

- Python ≥ 3.10
- CMake ≥ 3.18 (for building C++ extension)
- Eigen 3.4, pybind11 (auto-fetched via CMake)
- Optional: CUDA Toolkit (for `--device cuda`)

## Platform

**Supported**: Linux, macOS  
**Not supported**: Windows (uses `multiprocessing` with `fork`)

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

Recommended: `pip install -e .` for unified dependency and path management. Or install deps only: `pip install numpy scipy pandas matplotlib`.

### 2. Build C++ extension

```bash
python build.py
```

By default CUDA is enabled when available; otherwise CPU-only. Use `--device cpu` or `--device cuda` at runtime. Force CPU-only build: `python build.py --no-cuda`.

**Offline build**: Place Eigen and pybind11 in an `externals/` directory (e.g. `externals/eigen-4.3.0`, `externals/pybind11`), or use `python build.py --local /path/to/externals` to avoid network fetch.

See [BUILD.md](BUILD.md) for more build options.

### 3. Run simulation

```bash
python main.py --N 50 --time 10 --plot
```

## Field Visualization

The `field_visualize.py` script visualizes electric potential distributions (static potential, RF pseudopotential, total potential) from the field CSV and voltage config. Supports 1D (single-axis) and 2D (heatmap or 3D surface) plots.

```bash
python field_visualize.py [options]
```

### Usage examples

```bash
# 1D potential along x-axis (default)
python field_visualize.py

# 2D heatmap in x-y plane
python field_visualize.py --vary x,y --mode heatmap

# 2D 3D surface plot
python field_visualize.py --vary x,y --mode 3d

# Custom range and output (use = for negative values)
python field_visualize.py --vary x,y --x_range=-100,100 --y_range=-50,50 --out output.png

# With offset and RF amplitude
python field_visualize.py --vary x,y --offset --show-rf-amp

# 1D potential with quadratic/quartic fit (overlay dashed lines, show R², center, k2)
python field_visualize.py --vary z --fit 2
python field_visualize.py --vary z --fit 4

# Compute trap frequencies f_x, f_y, f_z (MHz)
python field_visualize.py --freq
python field_visualize.py --freq --const 0,0,50 --freq-fit-degree 4
```

### Field visualization options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | data/monolithic20241118.csv | Electric field CSV path |
| `--config` | FieldConfiguration/default.json | Voltage config JSON path |
| `--vary` | x | Varying axes: single (x/y/z) for 1D; comma-separated (e.g. x,y) for 2D |
| `--x_range` | -100,100 | Range for primary axis (μm), comma-separated |
| `--y_range` | -100,100 | Range for second axis in 2D (μm) |
| `--const` | 0,0,0 | Fixed coordinates x,y,z (μm), comma-separated |
| `--mode` | heatmap | 2D mode: heatmap or 3d |
| `--n_pts` | - | 1D: single integer (e.g. 500); 2D: comma-separated (e.g. 100,100) |
| `--out` | - | Output image path |
| `--offset` | - | Subtract min from each potential (total = offset DC + offset pseudopotential) |
| `--show-rf-amp` | - | Show RF amplitude plot (default: off) |
| `--fit` | - | 1D: polynomial fit for potential—`2`=quadratic, `4`=quartic; overlay dashed lines, show R², center, k2 |
| `--freq` | - | Compute and print trap frequencies f_x, f_y, f_z (MHz) at --const point |
| `--z_range` | -100,100 | z-axis fit range (μm) when using --freq |
| `--freq-fit-degree` | 2 | Fit degree (2 or 4) for --freq |
| `--freq-n-pts` | 200 | Sample points per axis for --freq |

**Note**: When passing negative values (e.g. `--x_range=-100,100`), use `=` to attach the value to the option; otherwise the parser may interpret `-100` as a new flag.

## Command-line Options

```bash
python main.py [options]
```

### Simulation

| Option | Default | Description |
|--------|---------|-------------|
| `--N` | 50 | Number of ions |
| `--t0` | 0 | Start time (μs) |
| `--time` | ∞ | Duration to run from t0 (μs); omit for infinite |
| `--alpha` | 0 | Isotope doping ratio |
| `--device` | cpu | Compute device: cpu / cuda |
| `--calc_method` | VV | Integration method: RK4 / VV |
| `--step` | 10 | Integration steps per frame |
| `--interval` | 1.0 | Frame interval (dt units) |
| `--batch` | 50 | Frames per batch |

### Input / Config

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | data/monolithic20241118.csv | Electric field CSV; pass filename only (e.g. monolithic20241118.csv) to look in data/ |
| `--config` | FieldConfiguration/default.json | Voltage config JSON; pass filename only (e.g. default.json) to look in FieldConfiguration/ |
| `--init_file` | - | Path to .npz with initial r0/v0; must contain 'r'(μm), 'v'(m/s), shape (N,3); if filename is t{time}us.npz, evolution starts from that time |

### Plotting

| Option | Default | Description |
|--------|---------|-------------|
| `--plot` | - | Enable real-time plotting |
| `--plot_fig` | - | Subplot views, comma-separated e.g. zoy,zox; default zoy,zox when --plot |
| `--color_mode` | - | Coloring: y_pos / v2 / isotope / none; default isotope when alpha>0 |
| `--ion_size` | 5.0 | Scatter point size |
| `--x_range` | 100 | x-axis display half-width (μm) |
| `--y_range` | 20 | y-axis display half-width (μm) |
| `--z_range` | 200 | z-axis display half-width (μm) |
| `--save_final_image` | - | Path to save the last frame |
| `--save_times_us` | - | Times (μs) to save trajectory frames, comma-separated e.g. 10,20,30; headless, no live window |
| `--save_fig_dir` | saves/images/traj | Root dir for trajectory frames; structure: `{dir}/{device}/{n_ions}/t{time}us.png` |
| `--save_rv_traj_dir` [DIR] | - | Save r/v at save_times_us to DIR; default saves/rv/traj when specified without value; structure: `{dir}/{device}/{n_ions}/`; requires --save_times_us |
| `--save_rv_status_dir` [DIR] | - | Save last-frame r/v to DIR; default saves/rv/status when specified without value; structure: `{dir}/{device}/{n_ions}/`; named by timestamp |

### Environment Variables

- `ISM_DEFAULT_CONFIG`, `ISM_DEFAULT_CSV`: Override default config paths
- `ISM_DEFAULT_SAVE_FIG_DIR`: Override default save_fig_dir (saves/images/traj)
- `ISM_LOG_LEVEL`: Log level (DEBUG / INFO / WARNING / ERROR)

## Project Structure

```
ism-main-v1.0/
├── Interface/          # CLI, parameters
├── FieldConfiguration/ # Constants, voltage config loader
├── FieldParser/       # CSV reader, field interpolation, force
├── ComputeKernel/     # C++ ionsim, Python backend
├── Plotter/           # Real-time visualization
├── benchmark/         # Performance benchmarks
├── data/              # Electric field grid CSV (default: data/monolithic20241118.csv)
├── externals/         # Local Eigen, pybind11 (optional, for offline build)
├── field_visualize.py # Electric potential field visualization (standalone)
├── main.py            # Entry point
└── setup_path.py      # Path setup for ionsim
```

## Benchmark

Performance benchmarks measure simulation time per 10 μs (averaged over 100 μs runs). Results are saved to `benchmark/benchmark_results/` (CSV and PNG).

### Plot comparison (with vs without plot)

Uses CUDA. Compares runtime with and without real-time plotting.

```bash
python -m benchmark.plot_compare
```

### Device comparison (CPU vs CUDA)

No plotting. Compares CPU and CUDA performance across ion counts.

```bash
python -m benchmark.device_compare
```

### Output

| Script | CSV | Figure |
|--------|-----|--------|
| `plot_compare` | `benchmark/benchmark_results/benchmark_plot_performance.csv` | `benchmark/benchmark_results/benchmark_plot_performance.png` |
| `device_compare` | `benchmark/benchmark_results/benchmark_device_compare.csv` | `benchmark/benchmark_results/benchmark_device_compare.png` |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

See project root for license information.
