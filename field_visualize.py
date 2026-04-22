"""
电场可视化：静电势、RF 赝势、总电势的空间分布
支持 1D（单坐标）与 2D（热力图/三维）绘图，可指定坐标变量与绘图区间。
命令行可加 --mark-potential-min：标出总势全局最小（品红），并用另一颜色标出经抑噪后的其它局部极小（青绿）。
"""
from __future__ import annotations

from pathlib import Path

# 需在 import 包前设置路径
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from field_visualize import main

if __name__ == "__main__":
    main()
