from __future__ import annotations

from pathlib import Path

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


def test_phase_9_8_parser_requires_an_explicit_mode() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["phase-9-8"])


def test_phase_9_8_parser_accepts_dry_and_full_modes() -> None:
    dry = build_parser().parse_args(["phase-9-8", "--dry-run"])
    full = build_parser().parse_args(
        ["phase-9-8", "--full", "--output", "results/custom"],
    )

    assert dry.dry_run
    assert not dry.full_run
    assert dry.output is None
    assert full.full_run
    assert not full.dry_run
    assert full.output == Path("results/custom")


def test_main_dispatches_phase_9_8_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from toposc_lab import phase_9_8_cli

    observed: list[bool] = []

    def fake_command(args: object) -> int:
        observed.append(bool(getattr(args, "dry_run")))
        return 17

    monkeypatch.setattr(phase_9_8_cli, "run_phase_9_8_command", fake_command)

    assert main(["phase-9-8", "--dry-run"]) == 17
    assert observed == [True]
