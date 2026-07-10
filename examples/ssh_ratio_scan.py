from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.scans.ssh_scan import scan_ssh_ratio
from toposc_lab.visualization.plots import (
    plot_gap_vs_parameter,
    plot_spectrum_vs_parameter,
)


def main() -> None:
    # v/w < 1: topologisch; v/w > 1: trivial.
    ratio_values = np.linspace(0.0, 2.0, 101)

    result = scan_ssh_ratio(
        ratio_values=ratio_values,
        n_cells=30,
        w=1.0,
        boundary="open",
    )

    plot_spectrum_vs_parameter(
        parameter_values=result.parameter_values,
        spectra=result.spectra,
        xlabel="Ratio v/w",
        ylabel="Energy",
        title="SSH spectrum",
        show=False,
    )

    plot_gap_vs_parameter(
        parameter_values=result.parameter_values,
        gaps=result.bulk_gaps,
        xlabel="Ratio v/w",
        ylabel="Bulk gap",
        title="SSH bulk gap",
        show=False,
    )

    plot_gap_vs_parameter(
        parameter_values=result.parameter_values,
        gaps=result.zero_mode_counts,
        xlabel="Ratio v/w",
        ylabel="Number of edge states",
        title="SSH edge states",
        show=False,
    )

    plt.show()


if __name__ == "__main__":
    main()