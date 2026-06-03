# 物理审阅报告 — 2026-06-02

对 ISM-Main 离子阱动力学模拟程序的物理正确性审阅。

---

## 🔴 严重问题

### P0-1. C++ 库仑力缺少无量纲常数因子 4 [已修复]

**文件**: `ComputeKernel/Coulomb.cpp:29-65`, `Coulomb_cuda.cu:56-61`

**现状**: C++ 库仑力返回 `q_i * q_j * Δr / |Δr|³`（常数 C = 1）。

**正确值**: C = 4，即 `4 * q_i * q_j * Δr / |Δr|³`。

**推导**:
- `dl³ = e² / (4πε₀ m₀ Ω²)`, `dt = 2/Ω`
- 无量纲库仑常数 `C = k_e·e²·dt² / (m₀·dl³) = Ω²·(2/Ω)² = 4`

**验证**: 圆轨道测试 (`tests/test_cpu_cuda_error_accumulation.py:51`) 使用 `v = 0.5/√R`，对应 C=1。若 C=4（正确值）应为 `v = 1/√R`。两离子平衡间距偏小为正确值的 `(1/4)^(1/3) ≈ 0.63` 倍。

**影响范围**:
- ✅ `main.py` 主模拟中所有使用 C++ ionsim 积分器的动力学模拟（CSV 外场和简谐阱模式）
- ❌ 不影响 `equilibrium/` 和 `collision_pressure/`（它们使用 Python 端 `energy.py` 中含正确 `k_e` 的库仑力计算）

**修复**: 在 `Coulomb.cpp` 和 `Coulomb_cuda.cu` 的 `factor` 计算中乘以 4.0：
```cpp
// Coulomb.cpp CPU
data_t factor = 4.0 * charge_i * charge_j * inv_dist3;

// Coulomb_cuda.cu
data_t factor = 4.0 * charge_i * shared_charge[j_in_tile] * inv_dist3;
```

**注意**: 修复后 `test_cpu_cuda_error_accumulation.py` 中圆轨道理论速度需同步修改为 `v = 1.0 / np.sqrt(R)`。

---

### P0-2. 碰撞散射角公式中椭圆积分参数错误 [已修复]

**文件**: `collision_pressure/collision.py:46-48`

**现状**:
```python
m_ellip = x / (1.0 + x)
theta = np.pi - 2.0 * np.sqrt(2.0) / np.sqrt(1.0 + x) * ellipk(m_ellip)
```

**正确公式**: 对 -C₄/r⁴ 偏振势，散射角积分为
```
χ = π - 2√2/√(1+x) · K((1-x)/(1+x))
```
其中 K 是完全椭圆积分（参数约定），`x = √(1 - b_c⁴/b⁴)`。

**错误**: `m_ellip` 应为 `(1-x)/(1+x)` 而非 `x/(1+x)`。

**数值验证**:

| 碰撞参数 | 代码结果 x/(1+x) | 正确结果 (1-x)/(1+x) |
|----------|--------------------|-----------------------|
| b → ∞    | ~33° (不趋于 0!)    | → 0° ✓               |
| b = 2b_c | ~30°               | ~2.2°                 |
| b → b_c  | ~74°               | → π (轨道化)          |

**影响**: 大幅高估远碰撞参数的散射效应，导致重构概率偏高，反推气压偏低。

**修复**:
```python
m_ellip = (1.0 - x) / (1.0 + x)  # 修改此行
```

---

## 🟡 中等问题

### P1-1. 碰撞垂直冲量方向未随机化方位角

**文件**: `collision_pressure/collision.py:79-85`

**现状**: 垂直方向始终取 `direction × ẑ`，未对碰撞平面方位角 φ 随机采样。

**物理要求**: 弹性散射的垂直分量应在以入射方向为轴的锥面上各向同性分布。

**建议修复**:
```python
def post_collision_kick(ion, mol, v0, theta, direction, *, rng=None):
    ...
    z = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(direction, z)
    norm = np.linalg.norm(e1)
    if norm < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 /= norm
    e2 = np.cross(direction, e1)
    phi = rng.uniform(0.0, 2.0 * np.pi) if rng is not None else 0.0
    perp = np.cos(phi) * e1 + np.sin(phi) * e2
    ...
```

需同步修改 `post_collision_kick` 签名以接收 `rng` 参数。

---

### P1-2. Velocity Verlet 积分器对速度相关力（耗散项）处理不当

**文件**: `ComputeKernel/numerical_integration.cpp:122-138`

**问题 a — 第一步使用 Euler**:
```cpp
// i=0 时：
a = acceleration(r, v, ...);
r += v * dt + a * dt²/2;
v += a * dt;           // ← 纯 Euler，精度一阶
```

**问题 b — 后续步用旧速度评估加速度**:
```cpp
// i≥1 时：
r += v * dt + a_last * dt²/2;
a = acceleration(r, v, t+dt);  // ← v 是 v_n，不是 v_{n+1}
v += (a + a_last) * dt/2;
```

当加速度含耗散项 `-γv` 时，应在更新后的速度处评估加速度，否则耗散精度降为一阶。

**影响**: 小阻尼下影响微弱；大 Doppler 冷却阻尼下能量耗散速率有偏差。

**建议**: 对含耗散的系统改用 Predictor-Corrector Verlet 或统一使用 RK4（现为默认）。

---

### P1-3. 碰撞参数采样不按截面积加权

**文件**: `collision_pressure/sampling.py:21-23`

**现状**: `b ~ Uniform(0, b_max)` — 均匀 in b。

**物理要求**: 微分截面 `dσ/db = 2πb`，应按面积加权采样：
```python
b = b_max * np.sqrt(rng.uniform(0.0, 1.0))  # 正确
```

**与 Langevin 速率的一致性问题**: `pressure.py` 的 `k_L` 仅覆盖 b < b_c 的轨道碰撞截面，而模拟采样 b ∈ [0, 3b_c]，二者不匹配。修复采样方式后需同步审查压力估算公式。

---

## 🟢 已验证正确的部分

| 模块 | 文件 | 状态 |
|------|------|------|
| 无量纲化系统 | `FieldConfiguration/constants.py` | ✅ dt, dl, dV 定义正确 |
| CSV 归一化 | `FieldParser/csv_reader.py` | ✅ r×unit_l/dl, V/dV 正确 |
| 外力函数 | `FieldParser/force.py` | ✅ 量纲转换与无量纲系统一致（除库仑力常数） |
| 电场计算 | `FieldParser/calc_field.py` | ✅ E = -grad(V) 正确 |
| 简谐阱力 | `FieldParser/force.py:build_harmonic_force` | ✅ k = (ω·dt)² 正确 |
| RK4 积分器 | `ComputeKernel/numerical_integration.cpp` | ✅ 标准四阶 Runge-Kutta |
| 库仑能量/梯度 | `equilibrium/energy.py` | ✅ 含正确 k_e 常数 |
| 库仑 Hessian | `equilibrium/phonon.py` | ✅ -I/s³ + 3rr'/s⁵ 正确 |
| 声子模式 | `equilibrium/phonon.py` | ✅ 动力学矩阵对角化、频率提取正确 |
| Langevin 碰撞速率 | `collision_pressure/collision.py` | ✅ |
| 临界碰撞参数 | `collision_pressure/collision.py` | ✅ (8C₄/(μv₀²))^(1/4) |
| 偏振系数 C₄ | `collision_pressure/species.py` | ✅ αq²/(2(4πε₀)²) |
| 质心速度转移 | `collision_pressure/collision.py` | ✅ ratio·v₀·[(1-cosθ)d̂ + sinθ·p̂] |
| 压力估算公式 | `collision_pressure/pressure.py` | ✅ P = R_obs·kB·T/(P_flip·k_L)（前提是 P_flip 采样一致） |
