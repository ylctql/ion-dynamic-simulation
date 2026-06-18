"""CLI 模块测试"""
from pathlib import Path

from Interface.cli import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CSV_PATH,
    create_parser,
    parse_save_times_us,
)


def test_create_parser_returns_parser():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.N == [50]
    assert args.config == ""
    assert args.csv == ""


def test_default_paths():
    assert "default.json" in DEFAULT_CONFIG_PATH
    assert "FieldConfiguration" in DEFAULT_CONFIG_PATH
    assert "data" in DEFAULT_CSV_PATH or "monolithic" in DEFAULT_CSV_PATH


def test_parse_args_with_options():
    parser = create_parser()
    args = parser.parse_args(["--N", "100", "--config", "my.json", "--csv", "my.csv"])
    assert args.N == [100]
    assert args.config == "my.json"
    assert args.csv == "my.csv"


def test_parse_n_comma_separated():
    parser = create_parser()
    args = parser.parse_args(["--N", "500,2000,10000"])
    assert args.N == [500, 2000, 10000]


def test_parse_save_times_us_comma():
    assert parse_save_times_us("10, 20, 30") == [10.0, 20.0, 30.0]


def test_parse_save_times_us_range():
    assert parse_save_times_us("range(100,1100,100)") == [
        float(x) for x in range(100, 1100, 100)
    ]


def test_parse_save_times_us_colon_range():
    """start:stop:step — no parentheses, shell-safe"""
    assert parse_save_times_us("100:1100:100") == [
        float(x) for x in range(100, 1100, 100)
    ]


def test_parse_save_times_us_mixed():
    assert parse_save_times_us("50,range(200,500,100),600") == [
        50.0,
        200.0,
        300.0,
        400.0,
        600.0,
    ]


def test_parse_save_times_us_range_step_zero_raises():
    import pytest

    with pytest.raises(ValueError, match="step"):
        parse_save_times_us("range(0,10,0)")


# ============== --continuous-sampling-plot ==============

_ROOT = Path(__file__).resolve().parent.parent
_CSV = "60electrodes_x100y100z600.csv"
_CFG = "default.json"


def test_continuous_sampling_plot_default_false():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.continuous_sampling_plot is False


def test_continuous_sampling_plot_flag_parsed():
    parser = create_parser()
    args = parser.parse_args(["--continuous-sampling", "--continuous-sampling-plot"])
    assert args.continuous_sampling_plot is True


def _try_build(args_list):
    """parse_and_build（用项目根 + 真实 csv/config）；csv/config 缺失则 skip。"""
    import pytest
    from Interface.cli import parse_and_build
    parser = create_parser()
    args = parser.parse_args(args_list)
    try:
        return parse_and_build(args, _ROOT)
    except FileNotFoundError:
        pytest.skip("csv/config 文件缺失，跳过 parse_and_build 测试")


def test_continuous_sampling_plot_implies_show_plot():
    parsed = _try_build([
        "--N", "2", "--csv", _CSV, "--config", _CFG,
        "--continuous-sampling", "--continuous-sampling-frames", "5",
        "--continuous-sampling-plot",
    ])
    assert parsed.continuous_sampling_plot is True
    assert parsed.vision.plot_fig is not None      # 不再被强制 None
    assert parsed.vision.show_plot is True         # imply 弹窗
    assert parsed.vision.save_times_us is None     # 聚焦逐帧 npz


def test_continuous_sampling_plot_requires_continuous_sampling():
    import pytest
    from Interface.cli import parse_and_build
    parser = create_parser()
    args = parser.parse_args([
        "--N", "2", "--csv", _CSV, "--config", _CFG, "--continuous-sampling-plot",
    ])
    with pytest.raises(ValueError, match="须配合"):
        parse_and_build(args, _ROOT)


def test_pure_continuous_sampling_unchanged():
    """回归：纯 --continuous-sampling 仍强制 plot_fig=None / show_plot=None。"""
    parsed = _try_build([
        "--N", "2", "--csv", _CSV, "--config", _CFG,
        "--continuous-sampling", "--continuous-sampling-frames", "5",
    ])
    assert parsed.continuous_sampling is True
    assert parsed.continuous_sampling_plot is False
    assert parsed.vision.plot_fig is None
    assert parsed.vision.show_plot is None
