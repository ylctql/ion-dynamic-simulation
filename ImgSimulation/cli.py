"""
命令行：从 JSON 配置运行单帧或 batch 离子成像模拟（见 configs/example_ion_image.json）。
"""
from __future__ import annotations

import argparse
import json
import os
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
        nargs="+",
        metavar="CONFIG",
        help=(
            "一份完整成像 JSON；或两份「动力学 JSON」「成像 JSON」（顺序固定）。示例见 "
            "configs/example_dynamics.json + configs/example_imaging.json"
        ),
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="启用 IMG_SIM_PROFILE=1（各阶段墙钟计时打印到 stderr）",
    )
    exp = parser.add_mutually_exclusive_group()
    exp.add_argument(
        "--export-plane-npz",
        type=Path,
        default=None,
        metavar="OUT.npz",
        help=(
            "仅运行曝光窗动力学并将成像平面轨迹写入该 NPZ（与 integrate_exposure_xy_um 契约一致）；"
            "不写 PNG。相对路径相对于 ImgSimulation 包目录。不可与 imaging JSON 中的 batch 块联用；"
            "动力学 JSON 若含 batch.dynamics_overrides 则不可用，请改用 --export-plane-batch。"
        ),
    )
    exp.add_argument(
        "--export-plane-batch",
        action="store_true",
        help=(
            "仅动力学：按动力学 JSON 的 batch.dynamics_overrides 逐条积分，"
            "将轨迹写入 batch.plane_npz_paths（与 overrides 等长；路径相对动力学 JSON 目录）。"
            "不需要成像 JSON。与 --export-plane-npz 互斥。"
        ),
    )
    args = parser.parse_args()

    from ImgSimulation.json_config import (
        export_dynamics_batch_plane_npz,
        load_dynamics_json,
        load_imaging_json,
        load_ion_image_json,
        load_ion_image_merged,
    )

    if args.profile:
        os.environ["IMG_SIM_PROFILE"] = "1"

    cfg_paths = [p.expanduser().resolve() for p in args.config]
    if len(cfg_paths) > 2:
        parser.error("至多 2 个 JSON 路径：先动力学，后成像")

    project_root = args.project_root

    if args.export_plane_batch:
        if len(cfg_paths) != 1:
            parser.error("--export-plane-batch 仅支持单个动力学 JSON")
        root = json.loads(cfg_paths[0].read_text(encoding="utf-8"))
        im = root.get("imaging")
        if isinstance(im, dict) and "psf_sigma_px" in im:
            parser.error("--export-plane-batch 仅用于动力学拆分 JSON（无顶层 imaging.psf_sigma_px）")
        dyn = load_dynamics_json(cfg_paths[0], project_root=project_root)
        recs = export_dynamics_batch_plane_npz(dyn)
        for i, rec in enumerate(recs):
            print(f"plane trajectory [{i}]: xy_stack shape (T+1, N, 2) = {rec.xy_stack.shape}", flush=True)
            print(f"  dt_real_s = {rec.dt_real_s}", flush=True)
        return

    if args.export_plane_npz is not None:
        from ImgSimulation.plane_trajectory_io import export_plane_trajectory_from_simulation

        # Relative --export-plane-npz paths are resolved against the ImgSimulation package directory
        # (the folder containing this file), not against the config JSON directory.
        img_sim_package_dir = Path(__file__).resolve().parent

        if len(cfg_paths) == 2:
            img_only = load_imaging_json(cfg_paths[1], project_root=project_root)
            if img_only.batch is not None:
                parser.error("--export-plane-npz 不能与 imaging JSON 中的 batch 同时使用")
            dyn = load_dynamics_json(cfg_paths[0], project_root=project_root)
            if dyn.dynamics_batch_overrides is not None:
                parser.error(
                    "动力学 JSON 含 batch.dynamics_overrides 时不能使用 --export-plane-npz；"
                    "请改用 --export-plane-batch（输出路径由 batch.plane_npz_paths 指定）"
                )
            meta = {
                "dynamics_json": str(cfg_paths[0]),
                "imaging_json": str(cfg_paths[1]),
                "cli": "ImgSimulation --export-plane-npz",
            }
            export_src = dyn
        else:
            root = json.loads(cfg_paths[0].read_text(encoding="utf-8"))
            im = root.get("imaging")
            if isinstance(im, dict) and "psf_sigma_px" in im:
                bundle_one = load_ion_image_json(cfg_paths[0], project_root=project_root)
                if bundle_one.batch is not None:
                    parser.error("--export-plane-npz 与 JSON 中的 batch 块不能同时使用")
                meta = {"source_json": str(bundle_one.source_json), "cli": "ImgSimulation --export-plane-npz"}
                export_src = bundle_one
            else:
                dyn = load_dynamics_json(cfg_paths[0], project_root=project_root)
                if dyn.dynamics_batch_overrides is not None:
                    parser.error(
                        "动力学 JSON 含 batch.dynamics_overrides 时不能使用 --export-plane-npz；"
                        "请改用 --export-plane-batch（输出路径由 batch.plane_npz_paths 指定）"
                    )
                meta = {"source_json": str(dyn.source_json), "cli": "ImgSimulation --export-plane-npz"}
                export_src = dyn

        out = args.export_plane_npz.expanduser()
        if not out.is_absolute():
            out = (img_sim_package_dir / out).resolve()
        rec = export_plane_trajectory_from_simulation(
            out,
            export_src.config,
            export_src.force,
            export_src.r0,
            export_src.v0,
            export_src.charge,
            export_src.mass,
            export_src.integ,
            use_cuda=export_src.use_cuda,
            calc_method=export_src.calc_method,
            use_zero_force=export_src.use_zero_force,
            log_interval_sim_us=export_src.log_interval_sim_us,
            meta=meta,
            dynamics_json_path=cfg_paths[0],
            project_root=project_root,
        )
        print(f"plane trajectory: {out}", flush=True)
        print(f"xy_stack shape (T+1, N, 2) = {rec.xy_stack.shape}", flush=True)
        print(f"dt_real_s = {rec.dt_real_s}", flush=True)
        return

    if len(cfg_paths) == 2:
        bundle = load_ion_image_merged(
            cfg_paths[0],
            cfg_paths[1],
            project_root=project_root,
        )
    else:
        root_data = json.loads(cfg_paths[0].read_text(encoding="utf-8"))
        im = root_data.get("imaging")
        if isinstance(im, dict) and "psf_sigma_px" in im:
            bundle = load_ion_image_json(cfg_paths[0], project_root=project_root)
        else:
            img_sibling = (cfg_paths[0].parent / "example_imaging.json").resolve()
            parser.error(
                "该 JSON 为动力学拆分配置（无 imaging.psf_sigma_px），不能单独跑成像流水线。\n"
                "请同时传入「动力学 JSON」「成像 JSON」，例如:\n"
                f"  python -m ImgSimulation {cfg_paths[0]} {img_sibling}"
            )
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

    if bundle.batch is not None:
        bundle.call_run_batch()
    else:
        bundle.call_run_ion_image()


if __name__ == "__main__":
    main()
