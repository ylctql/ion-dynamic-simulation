# 构建说明

## 一键构建（推荐）

```bash
# 默认尝试 CUDA，不可用时回退 CPU-only
python build.py
# 或
./build.sh
```

## 选项

| 方式 | 命令 |
|------|------|
| 强制仅 CPU | `python build.py --no-cuda` |
| 使用本地依赖（无网络） | `python build.py --local ../ism-hybrid` 或 `./build.sh --local ../ism-hybrid` |
| 清理后重建 | `python build.py --clean` 或 `./build.sh --clean` |
| 并行编译 | `python build.py -j 8` |

构建后通过 `main.py --device cpu` 或 `--device cuda` 选择计算设备。

## 手动构建

若需手动控制 CMake 参数：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

无网络时指定本地 Eigen、pybind11（如 ism-hybrid 的 externals）：

```bash
cmake -B build \
  -DEIGEN_LOCAL_PATH=/path/to/ism-hybrid/externals/eigen-4.3.0 \
  -DPYBIND11_LOCAL_PATH=/path/to/ism-hybrid/externals/pybind11
cmake --build build
```

CUDA 加速需已安装 CUDA Toolkit。

## 使用

```python
import sys
sys.path.insert(0, 'build')
import ionsim

# init_r, init_v: (N, 3) Fortran-contiguous
# charge, mass: (N, 1) 或 (N,) 列向量
r_list, v_list = ionsim.calculate_trajectory(
    init_r, init_v, charge, mass,
    step=100, time_start=0, time_end=1,
    force=force_fn, use_cuda=False, calc_method="RK4"
)
```
