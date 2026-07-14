"""
param_scan 共享 log 读写（**列以 TAB 分隔**，非逗号）。

列定义在扫描与热力图间共用，保证"绘图仅依赖 log"的解耦约定：

    run_idx  x2_coeff  x4_coeff  seed  reached_threshold  status  sim_time_us  sigma_y_final_um  escape  wall_time_s

  * run_idx            —— 运行序号（1 起，与 .npy 文件名一致）
  * x2_coeff, x4_coeff —— 当轮 x²/x⁴ 多项式系数 (V)
  * seed               —— 当轮随机种子（初态可复现）
  * reached_threshold  —— 1/0，50µs 内是否达到 σ_y < 阈值（escaped/diverged/error 恒为 0）
  * status             —— 当轮结局状态码：reached / timeout / escaped / diverged / error
  * sim_time_us        —— reached→跨越时刻；escaped→逃逸时刻；timeout→TIME_MAX_US；
                          diverged→最后有效帧时刻；error→0
  * sigma_y_final_um   —— 对应时刻 σ_y (µm)；diverged/error 为 nan
  * escape             —— 逃逸方向描述（如 ``+x(3),-y(1)``）；无逃逸为 ``-``
  * wall_time_s        —— 该轮挂钟耗时 (s)

**分隔符**：TAB（``\t``）。escape 字段内部用逗号，因列分隔符为 tab 不冲突。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

DELIMITER = "\t"

COLUMNS = [
    "run_idx",
    "x2_coeff",
    "x4_coeff",
    "seed",
    "reached_threshold",
    "status",
    "sim_time_us",
    "sigma_y_final_um",
    "escape",
    "wall_time_s",
]

_INT_COLS = {"run_idx", "seed", "reached_threshold"}
_FLOAT_COLS = {"x2_coeff", "x4_coeff", "sim_time_us", "sigma_y_final_um", "wall_time_s"}


def write_header(path: str | Path) -> None:
    """
    log 不存在或为空时写表头（幂等）。已有内容时校验表头与当前列定义一致——
    若不一致（旧版逗号格式或列集变更）抛 RuntimeError，提示用 ``--new-log`` 重开，
    避免旧表头 + 新行错位污染数据。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    expected = DELIMITER.join(COLUMNS)
    if not p.exists() or p.stat().st_size == 0:
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write(expected + "\n")
        return
    with p.open("r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n").rstrip("\r")
    if first != expected:
        raise RuntimeError(
            "已有 log 的表头与当前列定义不一致（分隔符或列集已变更）：\n"
            f"  文件表头: {first!r}\n"
            f"  当前定义: {expected!r}\n"
            f"  → 请用 --new-log 截断重开，或删除该 log 后重跑。"
        )
    # 表头一致：幂等 no-op


def append_row(
    path: str | Path,
    *,
    run_idx: int,
    x2_coeff: float,
    x4_coeff: float,
    seed: int,
    reached_threshold: bool,
    status: str,
    sim_time_us: float,
    sigma_y_final_um: float,
    escape: str,
    wall_time_s: float,
) -> None:
    """追加一行（TAB 分隔）并立即 flush（流式落盘，抗崩溃 / 支持断点续跑）。"""
    fields = {
        "run_idx": str(int(run_idx)),
        "x2_coeff": f"{float(x2_coeff):.12g}",
        "x4_coeff": f"{float(x4_coeff):.12g}",
        "seed": str(int(seed)),
        "reached_threshold": "1" if reached_threshold else "0",
        "status": str(status),
        "sim_time_us": f"{float(sim_time_us):.6g}",
        "sigma_y_final_um": f"{float(sigma_y_final_um):.6g}",
        "escape": str(escape) if escape else "-",
        "wall_time_s": f"{float(wall_time_s):.3f}",
    }
    row = DELIMITER.join(fields[c] for c in COLUMNS)
    with Path(path).open("a", encoding="utf-8", newline="") as f:
        f.write(row + "\n")
        f.flush()


def read_log(path: str | Path) -> list[dict[str, Any]]:
    """
    读 log（TAB 分隔）。数值列按列名转 int/float，其余（status/escape）保留字符串。
    文件缺失返回 []。空字段（``-`` / 空）→ int 列 0、float 列 nan。
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return []
        header = header_line.rstrip("\n").rstrip("\r").split(DELIMITER)
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            vals = line.split(DELIMITER)
            row: dict[str, Any] = {}
            for i, name in enumerate(header):
                raw = vals[i] if i < len(vals) else ""
                if name in _INT_COLS:
                    row[name] = int(raw) if raw not in ("", "-") else 0
                elif name in _FLOAT_COLS:
                    row[name] = float(raw) if raw not in ("", "-") else float("nan")
                else:
                    row[name] = raw
            rows.append(row)
    return rows
