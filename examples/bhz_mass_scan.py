from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.scans.bhz_scan import scan_bhz_mass
from toposc_lab.visualization.plots import (
    plot_gap_vs_parameter,
    plot_spectrum_vs_parameter,
)


def main() -> None:
    # Das BHZ-Modell wechselt seine Phase bei m = -2, 0 und 2.
    mass_values = np.linspace(-3.0, 3.0, 121)

    # Periodische Ränder: reines Bulk-Spektrum ohne helikale Randzustände.
    result = scan_bhz_mass(
        mass_values=mass_values,
        n_x=6,
        n_y=6,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    plot_spectrum_vs_parameter(
        parameter_values=result.mass_values,
        spectra=result.spectra,
        xlabel="Mass parameter m",
        ylabel="Energy",
        title="BHZ bulk spectrum",
        show=False,
    )

    plot_gap_vs_parameter(
        parameter_values=result.mass_values,
        gaps=result.gaps,
        xlabel="Mass parameter m",
        ylabel="Bulk gap",
        title="BHZ bulk gap",
        show=False,
    )

    plt.show()


if __name__ == "__main__":
    main()