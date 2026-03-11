# CUDA 库仑力 Kernel 多 Block 错误修复总结

## 一、问题现象

- **N ≤ 256**（单 block）：CPU 与 CUDA 轨迹吻合良好，偏差在 1e-9 μm 以下
- **N > 256**（多 block）：CPU 与 CUDA 轨迹偏差达 μm ~ 10 μm 量级

即使从完全相同的初始状态开始演化，两种模式下的同一离子轨迹也会出现显著差异。

---

## 二、算法结构：Tiled N-body 力计算

每个离子 i 受到的库仑力为：

$$F_i = \sum_{j \neq i} q_i q_j \frac{\vec{r}_i - \vec{r}_j}{|\vec{r}_i - \vec{r}_j|^3}$$

需要遍历所有 j ≠ i，共 N-1 对相互作用。

### 2.1 线程与离子的对应关系

```
blockIdx.x = 0:  线程 tid 0..255  →  离子 i = 0..255
blockIdx.x = 1:  线程 tid 0..255  →  离子 i = 256..511
...
```

- 每个 block 固定 256 个线程（`threadsPerBlock = 256`）
- 离子索引：`i = blockIdx.x * 256 + tid`
- 当 N > 256 时，`blocksPerGrid ≥ 2`

### 2.2 Tile 循环与 Shared Memory

将 N 个离子分成若干 tile，每个 tile 大小为 256：

```
tile 0: j = 0..255
tile 1: j = 256..511
...
```

对每个 tile 的两阶段：

1. **加载阶段**：线程 tid 负责加载 `j = tile_start + tid` 的位置和电荷到 shared memory
   - `shared_r[d * 256 + tid]` ← `r[d*N + (tile_start+tid)]`
   - `shared_charge[tid]` ← `charge[tile_start + tid]`

2. **计算阶段**：每个 active 线程 i 遍历 `j_in_tile = 0..255`，用 `shared_r[..., j_in_tile]` 和 `shared_charge[j_in_tile]` 计算力贡献并累加

---

## 三、核心：加载者与读取者的对应关系

| 角色 | 行为 |
|------|------|
| **加载者** | 线程 tid 写入 `shared_r[..., tid]`、`shared_charge[tid]` |
| **读取者** | active 线程在 `j_in_tile = 0..255` 时读取 `shared_r[..., j_in_tile]`、`shared_charge[j_in_tile]` |

**关键结论**：`shared_r[j_in_tile]` 必须由 **tid = j_in_tile** 的线程写入。

---

## 四、Bug 根源：Early Return 导致加载者不足

原代码在 kernel 开头有：

```cuda
if (i >= N) return;   // 离子 i 超出范围 → 线程直接退出
```

含义：`i >= N` 的线程在进入 tile 循环前就退出，不再参与 shared memory 加载和 `__syncthreads()`。

### 4.1 单 Block（N ≤ 256）时为何正确

此时 `blocksPerGrid = 1`，只有 block 0。

- 参与计算的线程：tid = 0..N-1（对应 i < N）
- 提前退出的线程：tid = N..255（对应 i ≥ N）

对每个 tile，我们只会在 `j < N` 时使用 shared 数据，即只用到：

- `j_in_tile` 满足 `tile_start + j_in_tile < N`
- 即 `j_in_tile < N - tile_start`

需要加载的索引：`j_in_tile = 0 .. min(255, N - tile_start - 1)`。

负责加载的线程：tid = 0 .. min(255, N - tile_start - 1)。

这些 tid 都满足 tid < N，因此都在「参与计算的线程」集合里，**不会 early return**，所以这些索引都能被正确加载。

**结论**：单 block 时，所有需要读取的 shared 索引，都由参与计算的线程正确写入。

### 4.2 多 Block（N > 256）时为何出错

以 N = 257 为例：

- **Block 0**：i = 0..255，全部 256 个线程参与 ✓
- **Block 1**：i = 256..511，但只有 i = 256 有效
  - tid 0：i = 256，参与
  - tid 1..255：i ≥ 257，**early return**，不进入 tile 循环

在 Block 1 中，对 tile 0（j = 0..255）：

- 需要读取：`shared_r[0..255]`、`shared_charge[0..255]`
- 实际加载：只有 tid 0 执行加载，只写了 `shared_r[0]`、`shared_charge[0]`
- `shared_r[1..255]`、`shared_charge[1..255]` 从未被写入，是未初始化/垃圾数据

内层循环会遍历 `j_in_tile = 0..255`，对 j = 1..255 会读到这些垃圾值，导致力计算错误。

---

## 五、总结表

| 场景 | 参与线程 | tile 0 需要读取的索引 | 负责加载的线程 | 是否足够 |
|------|----------|------------------------|----------------|----------|
| N=10, 单 block | tid 0..9 | 0..9 | tid 0..9 | ✓ |
| N=257, block 0 | tid 0..255 | 0..255 | tid 0..255 | ✓ |
| N=257, block 1 | 仅 tid 0 | 0..255 | 需要 tid 0..255 | ✗ 只有 tid 0 在加载 |

---

## 六、本质原因

- **单 block**：参与计算的线程集合 = {tid : i < N} = {0..N-1}，而每个 tile 需要读取的索引集合 ⊆ {0..N-1}，因此「加载者」足够。
- **多 block 最后一个 block**：参与计算的线程数 = N - 256⌊N/256⌋，可能远小于 256；但每个 tile 仍需要 256 个加载者才能填满 shared memory。当参与线程数 < 256 时，部分 shared 索引无人写入，导致读到垃圾数据。

---

## 七、修复方案

**去掉 early return**，让 block 内所有 256 个线程都参与 tile 加载和 `__syncthreads()`，保证每个 tile 的 shared 内存都被完整、正确填充。只有 `i < N` 的线程做力累加和写回。

### 修改要点

1. 用 `bool active = (i < N)` 替代 `if (i >= N) return;`
2. 所有线程参与 shared memory 加载（无 early return）
3. 仅 `active` 的线程执行力累加和 `result` 写回

### 验证结果

| N | 修复前 | 修复后 |
|---|--------|--------|
| 258 | μm ~ 10 μm | 0.000000e+00 μm |
| 300 | μm ~ 10 μm | 0.000000e+00 μm |

---

## 八、相关文件

- `ComputeKernel/Coulomb_cuda.cu`：CUDA kernel 实现
- `ComputeKernel/Coulomb.cpp`：调用入口与 CPU 实现
- `tests/test_cpu_cuda_trajectory_distance.py`：CPU-CUDA 轨迹一致性测试
