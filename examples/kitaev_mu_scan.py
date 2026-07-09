from __future__ import annotations

import numpy as np

from toposc_lab.scans.kitaev_scan import scan_kitaev_mu
from toposc_lab.visualization.plots import (
    plot_gap_vs_parameter,
    plot_spectrum_vs_parameter,
)


def main() -> None:
    mu_values = np.linspace(-4.0, 4.0, 161)

    result = scan_kitaev_mu(
        mu_values=mu_values,
        L=60,
        t=1.0,
        delta=1.0,
        periodic=False,
    )

    plot_spectrum_vs_parameter(
        parameter_values=result.mu_values,
        spectra=result.spectra,
        xlabel="Chemical potential μ",
        ylabel="Energy",
        title="Kitaev chain spectrum",
    )

    plot_gap_vs_parameter(
        parameter_values=result.mu_values,
        gaps=result.gaps,
        xlabel="Chemical potential μ",
        ylabel="Gap",
        title="Kitaev chain gap",
    )


if __name__ == "__main__":
    main()