"""Command-line interface for TopoSC Lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from toposc_lab.scans.kitaev_scan import scan_kitaev_mu
from toposc_lab.visualization.plots import (
    plot_gap_vs_parameter,
    plot_spectrum_vs_parameter,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the TopoSC Lab command-line parser."""
    parser = argparse.ArgumentParser(
        prog="toposc",
        description="Tools for simulating topological superconducting systems.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    kitaev_scan_parser = subparsers.add_parser(
        "kitaev-scan",
        help="Scan the Kitaev-chain spectrum over chemical potential.",
    )
    kitaev_scan_parser.add_argument("--L", type=int, default=60, help="Number of lattice sites.")
    kitaev_scan_parser.add_argument(
        "--mu-min",
        type=float,
        default=-4.0,
        help="Minimum chemical potential.",
    )
    kitaev_scan_parser.add_argument(
        "--mu-max",
        type=float,
        default=4.0,
        help="Maximum chemical potential.",
    )
    kitaev_scan_parser.add_argument(
        "--num-points",
        type=int,
        default=161,
        help="Number of chemical-potential values.",
    )
    kitaev_scan_parser.add_argument("--t", type=float, default=1.0, help="Hopping amplitude.")
    kitaev_scan_parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="p-wave pairing amplitude.",
    )
    kitaev_scan_parser.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic instead of open boundary conditions.",
    )

    phase_9_8_parser = subparsers.add_parser(
        "phase-9-8",
        help="Run Phase 9.8 with simple progress and resource monitoring.",
    )
    mode = phase_9_8_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the ten reserved geometry checks; no scientific search.",
    )
    mode.add_argument(
        "--full",
        dest="full_run",
        action="store_true",
        help="Run the complete frozen search and any triggered disorder stages.",
    )
    phase_9_8_parser.add_argument(
        "--output",
        type=Path,
        help="Fresh output directory; full runs default to a timestamped results path.",
    )

    research_parser = subparsers.add_parser(
        "phase-10-research", help="Run the frozen paired Class-D research campaign with live progress.",
    )
    research_mode = research_parser.add_mutually_exclusive_group(required=True)
    research_mode.add_argument("--preflight", dest="research_mode", action="store_const", const="preflight")
    research_mode.add_argument("--full", dest="research_mode", action="store_const", const="full")
    research_mode.add_argument("--resume", dest="research_mode", action="store_const", const="resume")
    research_parser.add_argument("--output", type=Path, default=Path("results/phase_10_research_v1"))

    calibration_parser = subparsers.add_parser(
        "phase-10-calibration",
        help="Run the frozen square-neighborhood calibration with live progress.",
    )
    calibration_mode = calibration_parser.add_mutually_exclusive_group(required=True)
    calibration_mode.add_argument(
        "--preflight", dest="calibration_mode", action="store_const", const="preflight"
    )
    calibration_mode.add_argument(
        "--full", dest="calibration_mode", action="store_const", const="full"
    )
    calibration_mode.add_argument(
        "--resume", dest="calibration_mode", action="store_const", const="resume"
    )
    calibration_parser.add_argument(
        "--output", type=Path, default=Path("results/phase_10_calibration_v1")
    )

    size_methods_parser = subparsers.add_parser(
        "phase-10-size-methods",
        help="Run the frozen finite-size and bulk-region comparison with live progress.",
    )
    size_methods_mode = size_methods_parser.add_mutually_exclusive_group(required=True)
    size_methods_mode.add_argument(
        "--preflight", dest="size_methods_mode", action="store_const", const="preflight"
    )
    size_methods_mode.add_argument(
        "--full", dest="size_methods_mode", action="store_const", const="full"
    )
    size_methods_mode.add_argument(
        "--resume", dest="size_methods_mode", action="store_const", const="resume"
    )
    size_methods_parser.add_argument(
        "--output", type=Path, default=Path("results/phase_10_size_methods_v1")
    )
    return parser


def run_kitaev_scan(args: argparse.Namespace) -> None:
    """Run a chemical-potential scan and display its plots."""
    if args.mu_min >= args.mu_max:
        raise ValueError("--mu-min must be smaller than --mu-max.")
    if args.num_points < 2:
        raise ValueError("--num-points must be at least 2.")

    mu_values = np.linspace(args.mu_min, args.mu_max, args.num_points)
    result = scan_kitaev_mu(
        mu_values=mu_values,
        L=args.L,
        t=args.t,
        delta=args.delta,
        periodic=args.periodic,
    )

    plot_spectrum_vs_parameter(
        parameter_values=result.mu_values,
        spectra=result.spectra,
        xlabel="Chemical potential μ",
        ylabel="Energy",
        title="Kitaev chain spectrum",
        show=False,
    )
    plot_gap_vs_parameter(
        parameter_values=result.mu_values,
        gaps=result.gaps,
        xlabel="Chemical potential μ",
        ylabel="Gap",
        title="Kitaev chain gap",
        show=False,
    )

    import matplotlib.pyplot as plt

    plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TopoSC Lab command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "kitaev-scan":
            run_kitaev_scan(args)
        elif args.command == "phase-10-research":
            from toposc_lab.search.phase_10_campaign import run_research_campaign

            report = run_research_campaign(args.output, mode=args.research_mode)
            print(f"Bericht: {report.resolve()}", flush=True)
        elif args.command == "phase-10-calibration":
            from toposc_lab.search.phase_10_calibration_campaign import (
                run_calibration_campaign,
            )

            report = run_calibration_campaign(args.output, mode=args.calibration_mode)
            print(f"Bericht: {report.resolve()}", flush=True)
        elif args.command == "phase-10-size-methods":
            from toposc_lab.search.phase_10_size_methods_campaign import (
                run_size_methods_campaign,
            )

            report = run_size_methods_campaign(args.output, mode=args.size_methods_mode)
            print(f"Bericht: {report.resolve()}", flush=True)
        elif args.command == "phase-9-8":
            from toposc_lab.phase_9_8_cli import run_phase_9_8_command

            return run_phase_9_8_command(args)
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer; bereits versiegelte Artefakte bleiben erhalten.")
        return 130

    return 0
