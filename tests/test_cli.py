"""CLI 模块测试"""
from pathlib import Path

from Interface.cli import DEFAULT_CONFIG_PATH, DEFAULT_CSV_PATH, create_parser


def test_create_parser_returns_parser():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.N == 50
    assert args.config == ""
    assert args.csv == ""


def test_default_paths():
    assert "default.json" in DEFAULT_CONFIG_PATH
    assert "FieldConfiguration" in DEFAULT_CONFIG_PATH
    assert "data" in DEFAULT_CSV_PATH or "monolithic" in DEFAULT_CSV_PATH


def test_parse_args_with_options():
    parser = create_parser()
    args = parser.parse_args(["--N", "100", "--config", "my.json", "--csv", "my.csv"])
    assert args.N == 100
    assert args.config == "my.json"
    assert args.csv == "my.csv"
