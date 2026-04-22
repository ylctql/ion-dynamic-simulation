"""
命令行：从 JSON 配置运行单帧离子成像模拟（见 configs/example_ion_image.json）。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(
        description="单帧离子 CCD/CMOS 风格图像模拟（由 JSON 驱动）",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="成像参数 JSON 路径（如 ImgSimulation/configs/example_ion_image.json）",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="仓库根目录；默认自动推断为 ImgSimulation 的上一级",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--show",
        action="store_true",
        help="显示图像窗口（覆盖 JSON 中 display.show）",
    )
    g.add_argument(
        "--no-show",
        action="store_true",
        help="不显示窗口（覆盖 JSON 中 display.show）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="保存 PNG 路径（覆盖 JSON 中 display.figure_path；可为相对路径）",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="显示窗口时不阻塞（matplotlib non-blocking；仅当实际 show 时有效）",
    )
    args = parser.parse_args()

    from ImgSimulation.json_config import load_ion_image_json

    bundle = load_ion_image_json(args.config, project_root=args.project_root)
    overrides: dict = {}
    if args.show:
        overrides["show"] = True
    elif args.no_show:
        overrides["show"] = False
    if args.output is not None:
        overrides["figure_path"] = args.output.expanduser().resolve()
    if args.no_block:
        overrides["show_block"] = False
    if overrides:
        bundle = replace(bundle, **overrides)

    bundle.call_run_ion_image()


if __name__ == "__main__":
    main()
