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
