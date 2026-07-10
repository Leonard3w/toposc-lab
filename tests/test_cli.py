from __future__ import annotations

import pytest

from toposc_lab.cli import build_parser, main


def test_kitaev_scan_parser_uses_defaults() -> None:
    args = build_parser().parse_args(["kitaev-scan"])

    assert args.L == 60
    assert args.mu_min == -4.0
    assert args.mu_max == 4.0
    assert args.num_points == 161
    assert not args.periodic


def test_kitaev_scan_parser_accepts_custom_values() -> None:
    args = build_parser().parse_args(
        [
            "kitaev-scan",
            "--L",
            "20",
            "--mu-min",
            "-3",
            "--mu-max",
            "3",
            "--num-points",
            "21",
            "--t",
            "2",
            "--delta",
            "0.5",
            "--periodic",
        ],
    )

    assert args.L == 20
    assert args.mu_min == -3.0
    assert args.mu_max == 3.0
    assert args.num_points == 21
    assert args.t == 2.0
    assert args.delta == 0.5
    assert args.periodic


def test_main_rejects_invalid_mu_range() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["kitaev-scan", "--mu-min", "1", "--mu-max", "1"])
