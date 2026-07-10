"""Command-line interface for TopoSC Lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

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
    except ValueError as error:
        parser.error(str(error))

    return 0
