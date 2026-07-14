"""
param_scan 纯 Python 单测（不依赖 ionsim / 动力学，可在 conda base 跑）。

覆盖：
  * parse_axis_spec —— linspace / 显式列表两种解析
  * sigma_y_um      —— 与 np.std 一致（µm）
  * detect_escape   —— ±600µm 逃逸方向检测（含多方向/无逃逸/NaN 不计入）
  * logio 往返       —— TAB 分隔、write_header/append_row/read_log 列与数值保真
  * logio 旧表头拒绝 —— 逗号/旧列表头混入时 write_header 抛错防错位
  * heatmap.load_grid —— 由 log 重建 (x2s, x4s, Z)，缺失点 NaN
scan.py 的动力学路径（run_until_sigma_or_time / run_scan）需 ionsim+CUDA，属集成测试，
见 docs/param_scan.md 的冒烟步骤，不在此覆盖。
"""
import math

import pytest
import numpy as np

from param_scan.heatmap import load_grid
from param_scan.logio import COLUMNS, DELIMITER, append_row, read_log, write_header
from param_scan.scan import detect_escape, parse_axis_spec, sigma_y_um


# ---------------------------------------------------------------- scan.parse_axis_spec
def test_parse_axis_spec_linspace():
    v = parse_axis_spec("0.005,0.025,5")
    assert len(v) == 5
    assert math.isclose(v[0], 0.005)
    assert math.isclose(v[-1], 0.025)
    assert math.isclose(v[2], 0.015)  # 中点


def test_parse_axis_spec_single_point():
    assert parse_axis_spec("0.01,0.01,1") == [0.01]


def test_parse_axis_spec_explicit():
    assert parse_axis_spec("0.01,0.02,0.03,0.05") == [0.01, 0.02, 0.03, 0.05]


def test_parse_axis_spec_three_noninteger_is_explicit():
    # 3 个值但第 3 个非整数 → 视为显式 3 元素列表（不误判为 linspace）
    assert parse_axis_spec("0.01,0.02,0.03") == [0.01, 0.02, 0.03]


def test_parse_axis_spec_single_value():
    # 1 个数：既非 linspace 也非多点 → 显式单值列表
    assert parse_axis_spec("0.042") == [0.042]


# ---------------------------------------------------------------- scan.sigma_y_um
def test_sigma_y_matches_std():
    dl = 1.5e-7  # 0.15 µm 特征长度
    rng = np.random.default_rng(0)
    r_norm = rng.normal(0.0, 1.0, (200, 3))
    r_um = r_norm * dl * 1e6
    assert math.isclose(sigma_y_um(r_norm, dl), float(np.std(r_um[:, 1])), rel_tol=1e-12)


def test_sigma_y_population_std_ddof0():
    # 与 dataplot.py:513 同定义：总体标准差（ddof=0），非样本（ddof=1）
    r_norm = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert math.isclose(sigma_y_um(r_norm, 1e-6), float(np.std([-1.0, 0.0, 1.0])), rel_tol=1e-12)


# ---------------------------------------------------------------- scan.detect_escape
def test_detect_escape_none():
    # 所有离子在 ±600µm 内 → 无逃逸
    r_um = np.array([[0, 0, 0], [599, -599, 300], [-300, 300, -599.999]])
    assert detect_escape(r_um) == ""


def test_detect_escape_single_axis():
    # 1 个离子 +x 超出，1 个 -y 超出
    r_um = np.array([[0, 0, 0], [601, 0, 0], [0, -700, 0]])
    assert detect_escape(r_um) == "+x(1),-y(1)"


def test_detect_escape_multi_direction_one_ion():
    # 单个离子斜向超出 +x/+y → 两个方向各计 1
    r_um = np.array([[800, 900, 0]])
    assert detect_escape(r_um) == "+x(1),+y(1)"


def test_detect_escape_counts():
    # +x 方向 3 个超出
    r_um = np.array([[601, 0, 0], [700, 0, 0], [1000, 0, 0], [0, 0, 0]])
    assert detect_escape(r_um) == "+x(3)"


def test_detect_escape_nan_not_counted():
    # NaN 坐标不计入逃逸（调用方应先用有限性检查）；超大有限值算逃逸
    r_um = np.array([[np.nan, np.nan, np.nan], [1e4, 0, 0]])
    assert detect_escape(r_um) == "+x(1)"


def test_detect_escape_custom_limit():
    r_um = np.array([[15.0, 0, 0]])
    assert detect_escape(r_um, limit_um=10.0) == "+x(1)"
    assert detect_escape(r_um, limit_um=20.0) == ""


# ---------------------------------------------------------------- logio 往返
def test_logio_roundtrip(tmp_path):
    log = tmp_path / "scan.log"
    write_header(log)
    append_row(log, run_idx=1, x2_coeff=0.01, x4_coeff=0.005, seed=1,
               reached_threshold=True, status="reached", sim_time_us=12.3,
               sigma_y_final_um=0.0004, escape="-", wall_time_s=3.5)
    append_row(log, run_idx=2, x2_coeff=0.02, x4_coeff=0.005, seed=2,
               reached_threshold=False, status="escaped", sim_time_us=50.0,
               sigma_y_final_um=0.012, escape="+x(3),-y(1)", wall_time_s=4.25)

    rows = read_log(log)
    assert len(rows) == 2
    assert list(rows[0].keys()) == COLUMNS
    assert rows[0]["run_idx"] == 1
    assert math.isclose(rows[0]["x2_coeff"], 0.01, rel_tol=1e-12)
    assert rows[0]["seed"] == 1
    assert rows[0]["reached_threshold"] == 1
    assert rows[0]["status"] == "reached"
    assert rows[0]["escape"] == "-"
    assert math.isclose(rows[0]["sim_time_us"], 12.3, rel_tol=1e-9)
    assert math.isclose(rows[0]["sigma_y_final_um"], 0.0004, rel_tol=1e-9)
    assert math.isclose(rows[0]["wall_time_s"], 3.5, rel_tol=1e-9)
    assert rows[1]["reached_threshold"] == 0
    assert rows[1]["status"] == "escaped"
    assert rows[1]["escape"] == "+x(3),-y(1)"
    assert math.isclose(rows[1]["sim_time_us"], 50.0)
    assert math.isclose(rows[1]["wall_time_s"], 4.25)


def test_logio_tab_separator(tmp_path):
    """列间必须为 TAB（非逗号）；escape 字段内的逗号不破坏列解析。"""
    log = tmp_path / "scan.log"
    write_header(log)
    append_row(log, run_idx=1, x2_coeff=0.01, x4_coeff=0.005, seed=1,
               reached_threshold=True, status="escaped", sim_time_us=12.3,
               sigma_y_final_um=0.0004, escape="+x(3),-y(1)", wall_time_s=3.5)
    text = log.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    assert len(lines) == 2
    header, row = lines
    assert header.split(DELIMITER) == COLUMNS  # 表头 TAB 分隔
    # 数据行恰好 10 个 TAB 分隔字段；escape 内的逗号不分裂
    assert len(row.split(DELIMITER)) == len(COLUMNS)
    assert "," not in DELIMITER  # 防御：分隔符确为 tab


def test_write_header_idempotent(tmp_path):
    log = tmp_path / "scan.log"
    write_header(log)
    write_header(log)  # 已有内容，不应重复写表头
    rows = read_log(log)
    assert rows == []  # 只有表头，无数据行


def test_write_header_rejects_old_comma_format(tmp_path):
    """旧版逗号格式表头混入时应抛错，提示 --new-log，而非静默追加错位行。"""
    log = tmp_path / "scan.log"
    log.write_text("run_idx,x2_coeff,x4_coeff\n1,0.01,0.005\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="--new-log"):
        write_header(log)


def test_write_header_rejects_old_columns(tmp_path):
    """表头为 TAB 但列集为旧版（无 status/escape/wall_time_s）时也应抛错。"""
    log = tmp_path / "scan.log"
    old = "\t".join(
        ["run_idx", "x2_coeff", "x4_coeff", "seed",
         "reached_threshold", "sim_time_us", "sigma_y_final_um"]
    )
    log.write_text(old + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="--new-log"):
        write_header(log)


# ---------------------------------------------------------------- heatmap.load_grid
def _row(log, i, x2, x4, s, *, status="reached", reached=True, escape="-"):
    append_row(log, run_idx=i, x2_coeff=x2, x4_coeff=x4, seed=i,
               reached_threshold=reached, status=status, sim_time_us=1.0,
               sigma_y_final_um=s, escape=escape, wall_time_s=2.0)


def test_heatmap_load_grid(tmp_path):
    log = tmp_path / "scan.log"
    write_header(log)
    # 2×2 网格；故意乱序写入，验证按 (x²,x⁴) 正确归位
    pts = [(0.01, 0.010, 0.002), (0.02, 0.005, 0.003),
           (0.01, 0.005, 0.001), (0.02, 0.010, 0.004)]
    for i, (x2, x4, s) in enumerate(pts, 1):
        _row(log, i, x2, x4, s)

    x2s, x4s, Z = load_grid(log)
    assert list(x2s) == [0.01, 0.02]
    assert list(x4s) == [0.005, 0.010]
    assert Z.shape == (2, 2)
    assert math.isclose(Z[0, 0], 0.001, rel_tol=1e-9)  # x2=0.01, x4=0.005
    assert math.isclose(Z[0, 1], 0.002, rel_tol=1e-9)  # x2=0.01, x4=0.010
    assert math.isclose(Z[1, 1], 0.004, rel_tol=1e-9)  # x2=0.02, x4=0.010


def test_heatmap_load_grid_missing_nan(tmp_path):
    log = tmp_path / "scan.log"
    write_header(log)
    # 只填 3/4 个点
    for i, (x2, x4, s) in enumerate(
        [(0.01, 0.005, 0.001), (0.01, 0.010, 0.002), (0.02, 0.010, 0.004)], 1
    ):
        _row(log, i, x2, x4, s)

    x2s, x4s, Z = load_grid(log)
    assert Z.shape == (2, 2)
    assert math.isnan(Z[1, 0])  # x2=0.02, x4=0.005 缺失
    assert math.isclose(Z[0, 0], 0.001, rel_tol=1e-9)
