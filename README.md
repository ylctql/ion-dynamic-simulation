# ISM-Main: Ion Trap Dynamics Simulation

[English](README.md) | [中文](README.zh.md)

---

Ion trap dynamics simulation with modular architecture. Simulates ion crystal dynamics under given electric field distributions, with support for CPU and CUDA-accelerated Coulomb force computation.

## Features

- **Modular design**: Input, field configuration, force parsing, compute kernel, and plotting are decoupled
- **RK4 & Velocity Verlet**: Configurable integration methods
- **CPU / CUDA**: Optional GPU acceleration for Coulomb force
- **Real-time plotting**: Live visualization with matplotlib

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

See [BUILD.md](BUILD.md) for more build options.

### 3. Run simulation

```bash
python main.py --N 50 --time 10 --plot
```

## Command-line Options

```bash
python main.py [options]
```

### Simulation

| Option | Default | Description |
|--------|---------|-------------|
| `--N` | 50 | Number of ions |
| `--t0` | 0 | Start time (μs) |
| `--time` | ∞ | Simulation duration (μs); omit for infinite |
| `--alpha` | 0 | Isotope doping ratio |
| `--device` | cpu | Compute device: cpu / cuda |
| `--calc_method` | VV | Integration method: RK4 / VV |
| `--step` | 10 | Integration steps per frame |
| `--interval` | 1.0 | Frame interval (dt units) |
| `--batch` | 50 | Frames per batch |

### Input / Config

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | data/monolithic20241118.csv | Electric field grid CSV path |
| `--config` | FieldConfiguration/default.json | Electrode voltage config JSON |
| `--init_file` | - | Path to .npz with initial r0/v0; must contain 'r'(μm), 'v'(m/s), shape (N,3) |

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

### Environment Variables

- `ISM_DEFAULT_CONFIG`, `ISM_DEFAULT_CSV`: Override default config paths
- `ISM_LOG_LEVEL`: Log level (DEBUG / INFO / WARNING / ERROR)

## Project Structure

```
ism-main-v1.0/
├── Interface/          # CLI, parameters
├── FieldConfiguration/ # Constants, voltage config loader
├── FieldParser/       # CSV reader, field interpolation, force
├── ComputeKernel/     # C++ ionsim, Python backend
├── Plotter/           # Real-time visualization
├── main.py            # Entry point
└── setup_path.py      # Path setup for ionsim
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

See project root for license information.
