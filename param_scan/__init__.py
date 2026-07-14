"""
param_scan —— poly-potential x²/x⁴ 系数扫描 + σ_y 热力图。

两部分严格解耦：
  * ``scan``  —— 动力学扫描：遍历 (x², x⁴) 组合跑 N 离子动力学，终止于
                 σ_y < 阈值 或 模拟达 TIME_MAX_US，存结束位置 .npy + 维护 scan.log。
  * ``plot``  —— 仅读 scan.log 画 (x², x⁴, σ_y) 热力图，不依赖动力学或 .npy。

用法::

    python -m param_scan scan --x2 0.005,0.05,8 --x4 0.001,0.02,6
    python -m param_scan plot --log param_scan/results/scan.log --out heat.png
"""
