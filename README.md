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
- **Equilibrium solver**: Fit 3D trap potential and minimize total energy (trap + Coulomb) to find ion crystal equilibrium positions

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

The `field_visualize` tool visualizes electric potential distributions (static potential, RF pseudopotential, total potential) from the field CSV and voltage config. Supports 1D (single-axis) and 2D (heatmap or 3D surface) plots, plus trap frequency computation and scanning.

```bash
python field_visualize.py [options]
# or
python -m field_visualize [options]
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

# Trap frequency scan along axis (no potential plot, freq distribution only)
python field_visualize.py --freq-scan z --freq-scan-n 50
python field_visualize.py --freq-scan x,y --freq-scan-n 30,30 --mode heatmap
python field_visualize.py --freq-scan x,y --mode 3d --out freq_2d.png

# Savitzky-Golay smoothing (default: along z; use --smooth-axes none to disable)
python field_visualize.py --vary z
python field_visualize.py --freq --smooth-axes x,y,z --smooth-sg 15,3
python field_visualize.py --vary z --smooth-axes none
```

### Field visualization options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | data/monolithic20241118.csv | Electric field CSV path |
| `--config` | FieldConfiguration/configs/default.json | Voltage config JSON path |
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
| `--freq-n-pts` | 200 | Fit sample points per axis for --freq |
| `--freq-scan` | - | Trap freq scan: single axis (x/y/z) for curve; two axes (e.g. x,y) for heatmap/3d; skips potential plot |
| `--freq-scan-n` | 50 | --freq-scan points: 1D=int, 2D=comma-separated (e.g. 30,30) |
| `--smooth-axes` | z | Smooth potential along axes (e.g. x,y,z or z); use `none` to disable |
| `--smooth-sg` | 11,3 | Savitzky-Golay params: window_length,polyorder (comma-separated) |

**Note**: When passing negative values (e.g. `--x_range=-100,100`), use `=` to attach the value to the option; otherwise the parser may interpret `-100` as a new flag.

## Equilibrium Solver

The `equilibrium` module computes ion crystal equilibrium positions by:

1. Fitting total trap potential with a 3D quartic polynomial (`fit_potential_3d_quartic`)
2. Building total energy `U_total = U_trap + U_coulomb`
3. Minimizing `U_total` with L-BFGS-B

After equilibrium is found, it can also:

- build Hessian matrices (`total`, `trap`, `coulomb`)
- solve phonon modes (eigenvalues/eigenvectors of mass-weighted dynamical matrix)
- work on a Hessian subspace selected by NumPy-like slice syntax (supports unions with commas, e.g. `:`, `0:10`, `::3`, `5`, `0::3,2::3`)
- visualize Hessian heatmaps and phonon spectra

Energy is reported in **eV**, and trap potential uses a unified shifted zero (`V_shifted = V_true - V_min_grid`, where `V_min_grid` is the minimum total potential on grid data) for clearer scale comparison with Coulomb energy.

```bash
python -m equilibrium.find_equilibrium [options]
```

### Usage examples

```bash
# Solve equilibrium for 40 ions (default field/config)
python -m equilibrium.find_equilibrium --N 40

# Custom trap ranges and initialization from file
python -m equilibrium.find_equilibrium --N 120 --x_range=-80,80 --y_range=-30,30 --z_range=-150,150 --init_file saves/rv/traj/cpu/300/t200.0us.npz

# Save figure with zoy (top) / zox (bottom) layout
python -m equilibrium.find_equilibrium --N 80 --plot --color_mode y_pos --plot-out equilibrium/results/equi_pos/80.png

# Solve phonon modes on a Hessian subspace and save spectrum/hessian outputs
# (use --*-out without a path to save to default directories)
python -m equilibrium.find_equilibrium --N 120 --phonon --hessian-slice 0:90 --plot-phonon-spectrum --plot-phonon-spectrum-out --plot-hessian trap --plot-hessian-out --save-hessian-data

# Plot eigenvector of a specific phonon mode on zox plane
# (mode index is in descending-frequency order; default mode 0)
python -m equilibrium.find_equilibrium --N 120 --hessian-slice 0:90 --plot-mode-vector 3 --plot-mode-vector-arrow-scale 1.8 --plot-mode-vector-out

# Use Hessian DOF union slices (e.g., x+z subspace)
python -m equilibrium.find_equilibrium --N 120 --phonon --hessian-slice 0::3,2::3 --plot-phonon-spectrum index

# Interactive mode-vector viewer (window only, no save path)
# Controls: slider / textbox+Enter / left-right arrow keys
python -m equilibrium.find_equilibrium --N 120 --phonon --hessian-slice 0::3,2::3 --plot-mode-vector 0
```

### Equilibrium solver options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | data/monolithic20241118.csv | Electric field CSV path (filename-only also supported) |
| `--config` | FieldConfiguration/configs/default.json | Voltage config JSON path (filename-only also supported) |
| `--N` | 20 | Number of ions |
| `--charge` | 1.0 | Charge per ion in elementary charge `e` |
| `--center` | 0,0,0 | Fit center `(x,y,z)` in μm |
| `--x_range` | -50,50 | x range for fit/optimization in μm |
| `--y_range` | -20,20 | y range for fit/optimization in μm |
| `--z_range` | -150,150 | z range for fit/optimization in μm |
| `--fit-n-pts-x` | 100 | Sample points along x-axis for 3D quartic fit |
| `--fit-n-pts-y` | 40 | Sample points along y-axis for 3D quartic fit |
| `--fit-n-pts-z` | 300 | Sample points along z-axis for 3D quartic fit |
| `--softening-um` | 0.001 | Coulomb softening length in μm |
| `--phonon` | - | Solve phonon modes at equilibrium (diagonalize dynamical matrix) |
| `--mass-amu` | 135.0 | Ion mass for phonon solver (amu, default Ba135) |
| `--phonon-print-modes` | 10 | Print first N phonon modes (descending by frequency) |
| `--hessian-slice` | : | Hessian DOF subspace slice (supports unions with commas), e.g. `:`, `0:10`, `::3`, `5`, `0::3,2::3` |
| `--plot-hessian` | - | Show Hessian heatmap window; optional kind: `total`(default) / `trap` / `coulomb` |
| `--plot-hessian-out` | - | Save Hessian heatmap; with no path uses default `equilibrium/results/hessian_plot/{N}_{slice}.png` |
| `--save-hessian-data` | - | Save Hessian matrices (`total/trap/coulomb`) as npz |
| `--hessian-data-out` | equilibrium/results/hessian_data/{N}_{slice}.npz | Hessian data npz output path |
| `--plot-phonon-spectrum` | - | Show phonon spectrum window; optional mode: `frequency`(default) / `index` |
| `--plot-phonon-spectrum-out` | - | Save phonon spectrum; with no path uses default `equilibrium/results/spectra/{N}_{slice}.png` |
| `--plot-mode-vector` | - | Show one phonon mode eigenvector on zox plane; optional mode index (descending by frequency), default `0`; window mode supports slider / textbox / left-right keys |
| `--plot-mode-vector-out` | - | Save mode-vector plot; with no path uses default `equilibrium/results/mode_vector/{N}_{slice}_mode{k}.png` |
| `--plot-mode-vector-arrow-scale` | 1.0 | Arrow length multiplier for `--plot-mode-vector` (>0) |
| `--plot-point-size` | - | Scatter point size for `--plot` and `--plot-mode-vector`; must be >0; if omitted, each plot uses its own default size |
| `--maxiter` | 500 | Max optimization iterations |
| `--tol` | 1e-10 | Relative convergence tolerance (`ftol`, dimensionless) |
| `--seed` | 42 | RNG seed when random initialization is used |
| `--init_file` | - | Optional `.npz` with key `r` (shape `(N,3)`, unit μm) |
| `--plot` | - | Plot equilibrium positions (`zoy` top, `zox` bottom) |
| `--color_mode` | none | `none` / `y_pos` / `v2` / `isotope` (unsupported modes degrade gracefully) |
| `--plot-out` | - | Output path for figure; if omitted, opens interactive window |
| `--out` | equilibrium/results/equi_pos/{N}.npz | Output path for equilibrium npz (default auto by ion count) |
| `--smooth-axes` | z | Potential smoothing axes (`none` to disable) |
| `--smooth-sg` | 11,3 | Savitzky-Golay smoothing parameters |

`--plot-*` and `--plot-*-out` are independent switches:
- set `--plot-*` to open a window
- set `--plot-*-out` to save output
- set both to show and save in one run

**Default naming rule for outputs with `--*-out`**:
- Hessian/spectrum: `{N}_{slice}` (for example `120_0:90.png`)
- mode-vector: `{N}_{slice}_mode{k}` (for example `120_0:90_mode3.png`)

## Command-line Options

```bash
python main.py [options]
```

### Simulation

| Option | Default | Description |
|--------|---------|-------------|
| `--N` | 50 | Number of ions; comma-separated runs multiple jobs in sequence (e.g. `500,2000,10000`); incompatible with `--init_file` |
| `--t0` | 0 | Start time (μs) |
| `--time` | ∞ | Simulation end time (μs); omit for infinite |
| `--alpha` | 0 | Isotope doping ratio; in single-isotope mode, abundance of that isotope |
| `--isotope` | - | Single-isotope mode: Ba133/Ba134/Ba135/Ba136/Ba137/Ba138; alpha = abundance of this isotope, rest = Ba135; omit for mixed mode |
| `--device` | cpu | Compute device: cpu / cuda |
| `--calc_method` | VV | Integration method: RK4 / VV |
| `--step` | 10 | Integration steps per frame |
| `--interval` | 1.0 | Frame interval (dt units) |
| `--batch` | 50 | Frames per batch |

### Input / Config

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | data/monolithic20241118.csv | Electric field CSV; pass filename only (e.g. monolithic20241118.csv) to look in data/ |
| `--config` | FieldConfiguration/configs/default.json | Voltage config JSON; pass filename only (e.g. default.json) to look in FieldConfiguration/configs/ |
| `--init_file` | - | Path to .npz with initial r0/v0; must contain 'r'(μm), 'v'(m/s), shape (N,3); t0 is taken from npz 't_us' first (for RF phase continuity), then from filename t{time}us.npz, else --t0 |
| `--smooth-axes` | z | Smooth potential along axes (e.g. x,y,z or z); use `none` to disable; Savitzky-Golay filter |
| `--smooth-sg` | 11,3 | Savitzky-Golay params: window_length,polyorder (comma-separated) |

### Plotting

| Option | Default | Description |
|--------|---------|-------------|
| `--plot` | - | Enable real-time plotting |
| `--plot_fig` | - | Subplot views, comma-separated e.g. zoy,zox; default zoy,zox when --plot |
| `--color_mode` | - | Coloring: y_pos / v2 / isotope / none; default isotope when alpha>0 or --isotope |
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
ism-main/
├── Interface/          # CLI, parameters
├── FieldConfiguration/ # Constants, voltage config loader; configs/ for JSON configs
├── FieldParser/       # CSV reader, field interpolation, force
├── ComputeKernel/     # C++ ionsim, Python backend
├── Plotter/           # Real-time visualization
├── benchmark/         # Performance benchmarks
├── data/              # Electric field grid CSV (default: data/monolithic20241118.csv)
├── externals/         # Local Eigen, pybind11 (optional, for offline build)
├── field_visualize.py # Field visualization entry script
├── field_visualize/   # Field visualization package
│   ├── core.py        # Unit conversion, potential computation, grid building
│   ├── trap_freq.py   # Trap frequency computation
│   ├── plots.py       # Potential and freq-scan plotting
│   └── cli.py         # Argument parsing and main flow
├── equilibrium/       # Equilibrium-position solver
│   ├── potential_fit_3d.py  # 3D quartic potential fit and gradient
│   ├── energy.py      # Trap/Coulomb/total energy in eV
│   ├── phonon.py      # Hessian construction and phonon mode solver
│   ├── find_equilibrium.py  # CLI: minimize total energy for equilibrium
│   └── results/       # Default output root directory
│       ├── equi_pos/      # Equilibrium npz outputs
│       ├── hessian_data/  # Hessian npz data outputs
│       ├── hessian_plot/  # Hessian heatmap outputs
│       ├── spectra/       # Phonon spectrum outputs
│       └── mode_vector/   # Phonon mode-vector plot outputs
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
