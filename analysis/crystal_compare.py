# 比较Comsol网格和多项式拟合出来的晶格
import numpy as np
import matplotlib.pyplot as plt

# 加载两组平衡构型
d1 = np.load("../outputs/300+50Trap/1000.npz")
d2 = np.load("../outputs/300+50Trap/1000_poly.npz")
r1, r2 = d1["r"], d2["r"]  # (N, 3): x, y, z (μm)

# zox 面投影：z 为横轴，x 为纵轴
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(r1[:, 2], r1[:, 0], s=20, c="red", label="Comsol", zorder=3)
ax.scatter(r2[:, 2], r2[:, 0], s=20, c="blue", label="Poly", zorder=3)
ax.set_xlabel("z (μm)", fontsize=13)
ax.set_ylabel("x (μm)", fontsize=13)
ax.set_aspect("equal")
ax.legend(fontsize=11)
ax.set_title("zox projection — crystal comparison", fontsize=14)
plt.tight_layout()
plt.show()
print("s")
