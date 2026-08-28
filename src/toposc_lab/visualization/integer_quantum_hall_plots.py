"""Publication-style figures for the integer quantum Hall dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.quantum_hall.integer_quantum_hall import (
    CONDUCTANCE_QUANTUM_SIEMENS,
    EdgeModeSpectrum,
    IQHEParameters,
    IQHEState,
    JOULE_PER_MEV,
    PlateauScan,
    broadened_density_of_states,
    edge_mode_spectrum,
    iqhe_state,
    plateau_scan,
)
from toposc_lab.visualization.style import add_panel_label, paper_style


def _branch_label(orbital: int, spin: int, parameters: IQHEParameters) -> str:
    if parameters.spin_mode == "zeeman":
        spin_label = "+" if spin > 0 else "-"
        return rf"$n={orbital},s={spin_label}$"
    if parameters.spin_mode == "degenerate":
        return rf"$n={orbital}$ (2x Spin)"
    return rf"$n={orbital}$"


def integer_quantum_hall_dashboard(
    parameters: IQHEParameters,
) -> tuple[plt.Figure, IQHEState, PlateauScan]:
    """Build the synchronized occupation, DOS and transport dashboard."""
    state = iqhe_state(parameters)
    scan = plateau_scan(parameters)
    density_of_states = broadened_density_of_states(parameters, state)

    visible = state.orbital_indices <= parameters.maximum_level
    energies_mev = state.energies_joule[visible] / JOULE_PER_MEV
    occupations = state.occupations[visible]
    orbitals = state.orbital_indices[visible]
    spins = state.spin_projections[visible]
    chemical_potential_mev = state.chemical_potential_joule / JOULE_PER_MEV
    dos_energy_mev = density_of_states.energy_joule / JOULE_PER_MEV
    dos_scale = JOULE_PER_MEV / 1.0e15
    total_dos = density_of_states.density_per_joule_m2 * dos_scale
    occupied_dos = density_of_states.occupied_density_per_joule_m2 * dos_scale

    with paper_style():
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(12.2, 8.3),
            constrained_layout=True,
        )
        level_axes, dos_axes, hall_axes, longitudinal_axes = axes.flat

        for energy, occupation, orbital, spin in zip(
            energies_mev,
            occupations,
            orbitals,
            spins,
            strict=True,
        ):
            level_axes.hlines(energy, 0.0, 1.0, color="0.78", linewidth=3.0)
            level_axes.hlines(
                energy,
                0.0,
                occupation,
                color="#0072B2",
                linewidth=4.0,
            )
            level_axes.text(
                1.03,
                energy,
                _branch_label(int(orbital), int(spin), parameters),
                va="center",
                fontsize=8,
            )

        level_axes.axhline(
            chemical_potential_mev,
            color="#D55E00",
            linestyle="--",
            label=rf"$\mu={chemical_potential_mev:.3g}$ meV",
        )
        level_axes.set(
            xlim=(0.0, 1.42),
            xlabel="Besetzungsanteil",
            ylabel="Energie (meV)",
            title="Landau-Niveaus und Besetzung",
        )
        level_axes.legend(loc="upper left")

        dos_axes.plot(total_dos, dos_energy_mev, color="0.25", label="gesamte DOS")
        dos_axes.fill_betweenx(
            dos_energy_mev,
            0.0,
            occupied_dos,
            color="#56B4E9",
            alpha=0.65,
            label="besetzter Anteil",
        )
        dos_axes.axhline(chemical_potential_mev, color="#D55E00", linestyle="--")
        dos_axes.set(
            xlabel=r"DOS ($10^{15}$ m$^{-2}$ meV$^{-1}$)",
            ylabel="Energie (meV)",
            title=rf"Gauß-Verbreiterung $\Gamma={parameters.landau_broadening_mev:g}$ meV",
        )
        dos_axes.legend(loc="upper right")

        hall_axes.plot(
            scan.magnetic_field_tesla,
            scan.hall_resistivity_ohm / 1.0e3,
            color="#0072B2",
            label=r"Plateau-Vorschau $\rho_{xy}$",
        )
        hall_axes.plot(
            scan.magnetic_field_tesla,
            scan.classical_hall_resistivity_ohm / 1.0e3,
            color="0.35",
            linestyle="--",
            label=r"klassisch $-B/(n_e e)$",
        )
        hall_axes.axvline(
            parameters.magnetic_field_tesla,
            color="#D55E00",
            linestyle=":",
        )
        if np.isfinite(state.transport.hall_resistivity_ohm):
            hall_axes.scatter(
                [parameters.magnetic_field_tesla],
                [state.transport.hall_resistivity_ohm / 1.0e3],
                color="#D55E00",
                zorder=5,
            )
        hall_axes.set(
            xlabel="Magnetfeld B (T)",
            ylabel=r"Hall-Widerstand $\rho_{xy}$ (k$\Omega$)",
            title="Klassische Hall-Linie und quantisierte Plateaus",
        )
        hall_axes.legend(loc="upper left")

        longitudinal_axes.plot(
            scan.magnetic_field_tesla,
            scan.longitudinal_resistivity_ohm / 1.0e3,
            color="#D55E00",
            label=r"phänomenologisch $\rho_{xx}$",
        )
        longitudinal_axes.axvline(
            parameters.magnetic_field_tesla,
            color="#0072B2",
            linestyle=":",
        )
        longitudinal_axes.scatter(
            [parameters.magnetic_field_tesla],
            [state.transport.longitudinal_resistivity_ohm / 1.0e3],
            color="#0072B2",
            zorder=5,
        )
        longitudinal_axes.set(
            xlabel="Magnetfeld B (T)",
            ylabel=r"Längswiderstand $\rho_{xx}$ (k$\Omega$)",
            title="Übergangspeaks zwischen den Plateaus",
        )
        longitudinal_axes.legend(loc="upper left")

        common_energy_limits = (
            float(dos_energy_mev[0]),
            float(dos_energy_mev[-1]),
        )
        level_axes.set_ylim(common_energy_limits)
        dos_axes.set_ylim(common_energy_limits)
        add_panel_label(level_axes, "(a)")
        add_panel_label(dos_axes, "(b)")
        add_panel_label(hall_axes, "(c)")
        add_panel_label(longitudinal_axes, "(d)")

    return figure, state, scan


def integer_quantum_hall_edge_figure(
    parameters: IQHEParameters,
) -> tuple[plt.Figure, EdgeModeSpectrum]:
    """Show confinement, chiral dispersion, drift and quantized edge current."""
    spectrum = edge_mode_spectrum(parameters)
    x_values = spectrum.x_over_l_b
    potential_mev = spectrum.potential_joule / JOULE_PER_MEV
    energy_mev = spectrum.energies_joule / JOULE_PER_MEV
    chemical_potential_mev = spectrum.chemical_potential_joule / JOULE_PER_MEV
    k_order = np.argsort(spectrum.k_l_b)

    with paper_style():
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(12.2, 8.2),
            constrained_layout=True,
        )
        potential_axes, dispersion_axes, velocity_axes, current_axes = axes.flat

        potential_axes.plot(x_values, potential_mev, color="#172554")
        potential_axes.fill_between(x_values, 0.0, potential_mev, color="#56B4E9", alpha=0.25)
        potential_axes.axvline(0.0, color="0.65", linewidth=0.8)
        potential_axes.set(
            xlabel=r"Leitzentrum $X/l_B$",
            ylabel="Confinement V(X) (meV)",
            title="Weiche Probenränder",
        )

        for branch_index, branch_energy in enumerate(energy_mev):
            label = _branch_label(
                int(spectrum.orbital_indices[branch_index]),
                int(spectrum.spin_projections[branch_index]),
                parameters,
            )
            dispersion_axes.plot(
                spectrum.k_l_b[k_order],
                branch_energy[k_order],
                label=label,
            )
        dispersion_axes.axhline(
            chemical_potential_mev,
            color="#D55E00",
            linestyle="--",
            label=rf"$\mu={chemical_potential_mev:.3g}$ meV",
        )
        for side, color, label in (
            (-1, "#009E73", "linker Rand: $v_y>0$"),
            (1, "#CC79A7", "rechter Rand: $v_y<0$"),
        ):
            selected = spectrum.crossing_sides == side
            if np.any(selected):
                branch_indices = spectrum.crossing_branch_indices[selected]
                crossing_energies = energy_mev[
                    branch_indices,
                    np.asarray(
                        [
                            np.argmin(abs(x_values - position))
                            for position in spectrum.crossing_x_over_l_b[selected]
                        ]
                    ),
                ]
                dispersion_axes.scatter(
                    spectrum.crossing_k_l_b[selected],
                    crossing_energies,
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                    s=36,
                    zorder=5,
                    label=label,
                )
        dispersion_axes.set(
            xlabel=r"Landau-Gauge-Impuls $k l_B=-X/l_B$",
            ylabel="Energie (meV)",
            title="Chirale Randdispersion",
        )
        dispersion_axes.legend(loc="upper center", ncols=2, fontsize=8)

        velocity_axes.plot(x_values, spectrum.drift_velocity_m_s / 1.0e3, color="#0072B2")
        velocity_axes.axhline(0.0, color="0.35", linewidth=0.8)
        velocity_axes.fill_between(
            x_values,
            0.0,
            spectrum.drift_velocity_m_s / 1.0e3,
            where=spectrum.drift_velocity_m_s >= 0.0,
            color="#009E73",
            alpha=0.25,
            label="Bewegung in +y",
        )
        velocity_axes.fill_between(
            x_values,
            0.0,
            spectrum.drift_velocity_m_s / 1.0e3,
            where=spectrum.drift_velocity_m_s <= 0.0,
            color="#CC79A7",
            alpha=0.25,
            label="Bewegung in -y",
        )
        velocity_axes.set(
            xlabel=r"Leitzentrum $X/l_B$",
            ylabel=r"Driftgeschwindigkeit $v_y$ (km/s)",
            title=r"$v_y=-(eB)^{-1}\,\partial_X V$",
        )
        velocity_axes.legend(loc="upper center", ncols=2)

        selected_voltage = parameters.hall_voltage_microvolt
        voltage_extent = max(200.0, 2.0 * abs(selected_voltage))
        hall_voltages = np.linspace(-voltage_extent, voltage_extent, 301)
        currents_na = (
            spectrum.mode_count_per_edge
            * CONDUCTANCE_QUANTUM_SIEMENS
            * hall_voltages
            * 1.0e3
        )
        current_axes.plot(
            hall_voltages,
            currents_na,
            color="#0072B2",
            label=rf"$N={spectrum.mode_count_per_edge}$ Kanäle",
        )
        current_axes.scatter(
            [selected_voltage],
            [spectrum.edge_current_ampere * 1.0e9],
            color="#D55E00",
            zorder=5,
            label="gewählter Hall-Bias",
        )
        current_axes.axhline(0.0, color="0.65", linewidth=0.8)
        current_axes.axvline(0.0, color="0.65", linewidth=0.8)
        current_axes.set(
            xlabel=r"Hall-Spannung $V_H$ ($\mu$V)",
            ylabel=r"Randstrom $I_y$ (nA)",
            title=r"$I_y=N(e^2/h)V_H$",
        )
        current_axes.legend(loc="upper left")

        add_panel_label(potential_axes, "(a)")
        add_panel_label(dispersion_axes, "(b)")
        add_panel_label(velocity_axes, "(c)")
        add_panel_label(current_axes, "(d)")

    return figure, spectrum


__all__ = [
    "integer_quantum_hall_dashboard",
    "integer_quantum_hall_edge_figure",
]
