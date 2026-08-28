"""Streamlit presentation for IQHE roadmap stages 0 through 2."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.quantum_hall.integer_quantum_hall import (
    CONDUCTANCE_QUANTUM_SIEMENS,
    EdgeModeSpectrum,
    FLUX_QUANTUM_WEBER,
    IQHEParameters,
    JOULE_PER_MEV,
    VON_KLITZING_CONSTANT_OHM,
    iqhe_state,
)
from toposc_lab.app.integer_quantum_hall_edge_live import (
    render_integer_quantum_hall_edge_live,
)
from toposc_lab.visualization.export import figure_to_bytes
from toposc_lab.visualization.integer_quantum_hall_plots import (
    integer_quantum_hall_dashboard,
    integer_quantum_hall_edge_figure,
)


_SPIN_LABELS = {
    "spinless": "Spinlos / ein aufgelöster Zweig",
    "degenerate": "Zweifach spinentartet",
    "zeeman": "Zeeman-aufgespalten",
}


def _figure_downloads(
    streamlit: Any,
    figure: plt.Figure,
    *,
    filename_stem: str,
) -> None:
    png_column, pdf_column = streamlit.columns(2)
    png_column.download_button(
        "PNG herunterladen (300 dpi)",
        data=figure_to_bytes(figure, file_format="png", dpi=300),
        file_name=f"{filename_stem}.png",
        mime="image/png",
        key=f"{filename_stem}-png",
    )
    pdf_column.download_button(
        "PDF herunterladen (Vektor)",
        data=figure_to_bytes(figure, file_format="pdf", dpi=300),
        file_name=f"{filename_stem}.pdf",
        mime="application/pdf",
        key=f"{filename_stem}-pdf",
    )


def render_integer_quantum_hall_inputs(streamlit: Any) -> IQHEParameters:
    """Render physical controls for the first IQHE work package."""
    streamlit.caption("Alle Regler aktualisieren Besetzung, DOS und Transport gemeinsam.")

    streamlit.subheader("Probe und Zustand")
    magnetic_field = float(
        streamlit.number_input(
            "Magnetfeld B (T)",
            min_value=0.1,
            max_value=30.0,
            value=5.0,
            step=0.1,
            format="%.3f",
            key="iqhe-magnetic-field",
        )
    )
    density_1e15 = float(
        streamlit.number_input(
            "Elektronendichte n_e (10^15 m^-2)",
            min_value=0.01,
            max_value=20.0,
            value=10.0 / FLUX_QUANTUM_WEBER / 1.0e15,
            step=0.1,
            format="%.3f",
            key="iqhe-electron-density",
        )
    )
    temperature = float(
        streamlit.number_input(
            "Temperatur T (K)",
            min_value=0.0,
            max_value=100.0,
            value=1.5,
            step=0.1,
            format="%.3f",
            key="iqhe-temperature",
        )
    )
    effective_mass = float(
        streamlit.number_input(
            "Effektive Masse m*/m_e",
            min_value=0.005,
            max_value=5.0,
            value=0.067,
            step=0.01,
            format="%.4f",
            key="iqhe-effective-mass",
        )
    )
    spin_mode = streamlit.selectbox(
        "Spinbehandlung",
        options=tuple(_SPIN_LABELS),
        format_func=lambda value: _SPIN_LABELS[value],
        key="iqhe-spin-mode",
        help=(
            "Tongs Abschnitt 2 startet effektiv spinlos. Unaufgelöste zweifache "
            "Entartung lässt nur gerade Plateau-Schritte erscheinen."
        ),
    )

    with streamlit.expander("Numerik und Expertenparameter", expanded=False):
        maximum_level = int(
            streamlit.slider(
                "Höchstes dargestelltes Landau-Niveau n_max",
                min_value=2,
                max_value=20,
                value=8,
                key="iqhe-maximum-level",
            )
        )
        g_factor = float(
            streamlit.number_input(
                "Effektiver g-Faktor",
                min_value=-30.0,
                max_value=30.0,
                value=-0.44,
                step=0.1,
                format="%.3f",
                disabled=spin_mode != "zeeman",
                key="iqhe-g-factor",
            )
        )
        broadening = float(
            streamlit.number_input(
                "Landau-DOS-Verbreiterung Γ (meV)",
                min_value=0.005,
                max_value=20.0,
                value=0.30,
                step=0.05,
                format="%.3f",
                key="iqhe-broadening",
                help="Aktuell eine normierte Gauß-Verbreiterung, noch kein Unordnungsmodell.",
            )
        )
        transition_width = float(
            streamlit.slider(
                "Plateau-Übergangsbreite δν",
                min_value=0.01,
                max_value=0.50,
                value=0.08,
                step=0.01,
                key="iqhe-transition-width",
            )
        )
        longitudinal_peak_kohm = float(
            streamlit.number_input(
                "ρ_xx-Peakhöhe (kΩ)",
                min_value=0.0,
                max_value=25.0,
                value=1.5,
                step=0.1,
                key="iqhe-rho-xx-peak",
            )
        )
        scan_min = float(
            streamlit.number_input(
                "B-Scan Minimum (T)",
                min_value=0.05,
                max_value=29.0,
                value=1.5,
                step=0.1,
                key="iqhe-scan-min",
            )
        )
        scan_max = float(
            streamlit.number_input(
                "B-Scan Maximum (T)",
                min_value=scan_min + 0.01,
                max_value=30.0,
                value=max(12.0, scan_min + 0.1),
                step=0.1,
                key="iqhe-scan-max",
            )
        )
        scan_points = int(
            streamlit.slider(
                "B-Scan Stützstellen",
                min_value=101,
                max_value=1_201,
                value=1_001,
                step=50,
                key="iqhe-scan-points",
            )
        )

    with streamlit.expander("Randmoden nach Abschnitt 2.1.1", expanded=False):
        edge_sample_width_l_b = float(
            streamlit.slider(
                "Probenbreite W/l_B",
                min_value=12.0,
                max_value=60.0,
                value=24.0,
                step=1.0,
                key="iqhe-edge-sample-width",
            )
        )
        edge_smoothness_l_b = float(
            streamlit.slider(
                "Rand-Skalenlänge lambda/l_B",
                min_value=0.5,
                max_value=min(8.0, 0.5 * edge_sample_width_l_b - 0.5),
                value=min(3.0, 0.5 * edge_sample_width_l_b - 0.5),
                step=0.1,
                key="iqhe-edge-smoothness",
                help="Tong verwendet die lokale lineare Näherung für lambda deutlich größer l_B.",
            )
        )
        edge_confinement_energy_mev = float(
            streamlit.number_input(
                "Confinement-Energie V_0 (meV)",
                min_value=1.0,
                max_value=500.0,
                value=45.0,
                step=1.0,
                key="iqhe-edge-confinement",
            )
        )
        edge_confinement_power = float(
            streamlit.slider(
                "Formexponent der weichen Wand",
                min_value=1.0,
                max_value=4.0,
                value=2.0,
                step=0.25,
                key="iqhe-edge-power",
            )
        )
        edge_maximum_level = int(
            streamlit.slider(
                "Höchstes Randdispersions-Niveau",
                min_value=0,
                max_value=min(12, maximum_level),
                value=min(4, maximum_level),
                key="iqhe-edge-maximum-level",
            )
        )
        hall_voltage_microvolt = float(
            streamlit.number_input(
                "Hall-Spannung V_H (µV)",
                min_value=-1_000.0,
                max_value=1_000.0,
                value=100.0,
                step=10.0,
                key="iqhe-edge-hall-voltage",
            )
        )
        skipping_orbit_radius_l_b = float(
            streamlit.slider(
                "Skipping-Orbit-Radius R/l_B",
                min_value=0.4,
                max_value=4.0,
                value=1.5,
                step=0.1,
                key="iqhe-edge-orbit-radius",
            )
        )
        edge_animation_speed = float(
            streamlit.slider(
                "Animations-Zeitlupe",
                min_value=0.2,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key="iqhe-edge-animation-speed",
            )
        )

    return IQHEParameters(
        magnetic_field_tesla=magnetic_field,
        electron_density_m2=density_1e15 * 1.0e15,
        effective_mass_ratio=effective_mass,
        temperature_kelvin=temperature,
        maximum_level=maximum_level,
        spin_mode=spin_mode,
        g_factor=g_factor,
        landau_broadening_mev=broadening,
        plateau_transition_width=transition_width,
        longitudinal_peak_resistivity_ohm=longitudinal_peak_kohm * 1.0e3,
        field_scan_min_tesla=scan_min,
        field_scan_max_tesla=scan_max,
        field_scan_points=scan_points,
        edge_sample_width_l_b=edge_sample_width_l_b,
        edge_smoothness_l_b=edge_smoothness_l_b,
        edge_confinement_energy_mev=edge_confinement_energy_mev,
        edge_confinement_power=edge_confinement_power,
        edge_maximum_level=edge_maximum_level,
        hall_voltage_microvolt=hall_voltage_microvolt,
        skipping_orbit_radius_l_b=skipping_orbit_radius_l_b,
        edge_animation_speed=edge_animation_speed,
    )


def _level_rows(parameters: IQHEParameters) -> list[dict[str, str | int]]:
    state = iqhe_state(parameters)
    visible = state.orbital_indices <= parameters.maximum_level
    rows: list[dict[str, str | int]] = []
    for orbital, spin, degeneracy, energy, occupation, density in zip(
        state.orbital_indices[visible],
        state.spin_projections[visible],
        state.branch_degeneracies[visible],
        state.energies_joule[visible],
        state.occupations[visible],
        state.density_contributions_m2[visible],
        strict=True,
    ):
        spin_text = "-" if spin == 0 else f"{int(spin):+d}"
        rows.append(
            {
                "n": int(orbital),
                "s": spin_text,
                "Entartung": f"{degeneracy:g}",
                "E (meV)": f"{energy / JOULE_PER_MEV:.6g}",
                "Besetzung": f"{occupation:.6f}",
                "Dichteanteil (10^15 m^-2)": f"{density / 1.0e15:.6g}",
            }
        )
    return rows


def _finite_or_dash(value: float, *, scale: float = 1.0, digits: str = ".5g") -> str:
    if not np.isfinite(value):
        return "kein Plateau"
    return format(value / scale, digits)


def _edge_crossing_rows(spectrum: EdgeModeSpectrum) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for branch, side, position, momentum, velocity, degeneracy in zip(
        spectrum.crossing_branch_indices,
        spectrum.crossing_sides,
        spectrum.crossing_x_over_l_b,
        spectrum.crossing_k_l_b,
        spectrum.crossing_velocity_m_s,
        spectrum.crossing_degeneracies,
        strict=True,
    ):
        spin = int(spectrum.spin_projections[branch])
        rows.append(
            {
                "Rand": "links" if side < 0 else "rechts",
                "n": int(spectrum.orbital_indices[branch]),
                "s": "-" if spin == 0 else f"{spin:+d}",
                "Entartung": f"{degeneracy:g}",
                "X/l_B": f"{position:.5g}",
                "k l_B": f"{momentum:.5g}",
                "v_y (km/s)": f"{velocity / 1.0e3:.5g}",
            }
        )
    return rows


def show_integer_quantum_hall_lab(streamlit: Any, parameters: IQHEParameters) -> None:
    """Render conventions and the synchronized IQHE plateau dashboard."""
    state = iqhe_state(parameters)
    streamlit.title("2. Integer Quantum Hall Effect - Etappen 0 bis 2")
    streamlit.caption(
        "Interaktive Umsetzung der Grundlagen aus David Tong, The Quantum Hall Effect, "
        "Abschnitte 2.1 und 2.1.1 (arXiv:1606.06687v2, gedruckte Seiten 42-47)."
    )
    streamlit.info(
        "Tong-Konvention: Elektronenladung -e, B in +z-Richtung. Daher gilt für ν "
        "gefüllte spinaufgelöste Landau-Niveaus ρ_xy=-h/(νe²) und σ_xy=+νe²/h."
    )

    metrics = streamlit.columns(6)
    metrics[0].metric("Füllfaktor ν", f"{state.filling_factor:.6g}")
    metrics[1].metric(
        "Nächstes Plateau",
        str(state.nearest_plateau_index) if state.nearest_plateau_index > 0 else "keines",
    )
    metrics[2].metric(
        "Chemisches Potential μ",
        f"{state.chemical_potential_joule / JOULE_PER_MEV:.5g} meV",
    )
    metrics[3].metric(
        "ρ_xy",
        _finite_or_dash(
            state.transport.hall_resistivity_ohm,
            scale=1.0e3,
        )
        + (" kΩ" if np.isfinite(state.transport.hall_resistivity_ohm) else ""),
    )
    metrics[4].metric(
        "ρ_xx",
        f"{state.transport.longitudinal_resistivity_ohm / 1.0e3:.5g} kΩ",
    )
    metrics[5].metric(
        "σ_xy / (e²/h)",
        _finite_or_dash(
            state.transport.hall_conductivity_siemens,
            scale=CONDUCTANCE_QUANTUM_SIEMENS,
        ),
    )

    dashboard_tab, edge_tab, conventions_tab, reproducibility_tab = streamlit.tabs(
        (
            "Plateau-Dashboard",
            "2.1.1 Chirale Randmoden",
            "0. Konventionen",
            "Grenzen & Reproduzierbarkeit",
        )
    )

    with dashboard_tab:
        streamlit.markdown(
            "Ändern Sie zuerst **B** oder **n_e**. Der Füllfaktor verschiebt dann gleichzeitig "
            "das chemische Potential, die Besetzung und die Position auf den Transportkurven."
        )
        figure, _, _ = integer_quantum_hall_dashboard(parameters)
        streamlit.pyplot(figure)
        _figure_downloads(
            streamlit,
            figure,
            filename_stem="integer_quantum_hall_plateau_dashboard",
        )
        plt.close(figure)

        streamlit.warning(
            "Die Gauß-Verbreiterung, Plateau-Übergangsbreite und ρ_xx-Peaks sind in "
            "dieser Etappe bewusst phänomenologisch. Sie demonstrieren die Messsignatur, "
            "aber noch nicht deren mikroskopische Entstehung durch Lokalisierung."
        )
        streamlit.markdown(
            "**So lesen Sie die vier Graphen:** (a) Blau ist der besetzte Anteil jedes "
            "Niveaus. (b) zeigt dieselben Zustände als verbreiterte DOS. (c) stellt die "
            "klassische Hall-Linie den Plateaus gegenüber. (d) markiert Übergänge durch "
            "Längswiderstands-Peaks."
        )
        with streamlit.expander("Aktuelle Landau-Zweige und Besetzungen"):
            streamlit.dataframe(_level_rows(parameters), hide_index=True, width="stretch")

    with edge_tab:
        streamlit.markdown(
            "Das Confinement hebt die k-Entartung am Rand auf. Über "
            "**X=-k l_B²** wird ein Ortsgradient zu einer Dispersion; deren Steigung "
            "bestimmt Richtung und Geschwindigkeit des Kanals."
        )
        render_integer_quantum_hall_edge_live(streamlit, parameters)
        streamlit.caption(
            "Die bewegten Punkte sind eine semiklassische Skipping-Orbit-Deutung. "
            "Die rechts gezeichneten Energiezweige sind die quantenmechanischen Zustände."
        )

        edge_figure, edge_spectrum = integer_quantum_hall_edge_figure(parameters)
        edge_metrics = streamlit.columns(5)
        edge_metrics[0].metric("Kanäle je Rand", str(edge_spectrum.mode_count_per_edge))
        edge_metrics[1].metric("Linke Kanäle", str(edge_spectrum.left_mode_count))
        edge_metrics[2].metric("Rechte Kanäle", str(edge_spectrum.right_mode_count))
        edge_metrics[3].metric(
            "Quantisierter Randstrom",
            f"{edge_spectrum.edge_current_ampere * 1.0e9:.6g} nA",
        )
        edge_metrics[4].metric(
            "Glattheit lambda/l_B",
            f"{parameters.edge_smoothness_l_b:.4g}",
        )
        streamlit.pyplot(edge_figure)
        _figure_downloads(
            streamlit,
            edge_figure,
            filename_stem="integer_quantum_hall_edge_modes",
        )
        plt.close(edge_figure)

        streamlit.latex(
            r"E_n(k)\simeq\hbar\omega_B(n+\tfrac12)+V(X),\qquad X=-k l_B^2"
        )
        streamlit.latex(
            r"v_y=\frac{1}{\hbar}\frac{\partial E_n}{\partial k}="
            r"-\frac{1}{eB}\frac{\partial V}{\partial X}"
        )
        streamlit.latex(r"I_y=N\frac{e^2}{h}V_H")
        if parameters.edge_smoothness_l_b <= 1.0:
            streamlit.warning(
                "Für lambda <= l_B ist die lokale lineare Näherung aus Abschnitt 2.1.1 "
                "nicht kontrolliert. Verwenden Sie für die Tong-Näherung lambda/l_B > 1."
            )
        if edge_spectrum.mode_count_per_edge != state.nearest_plateau_index:
            streamlit.warning(
                "Die Zahl sichtbarer Fermi-Schnittpunkte stimmt nicht mit dem nächsten "
                "Plateauindex überein. Erhöhen Sie V_0 oder das höchste dargestellte "
                "Randniveau und prüfen Sie, ob μ in einer Bulk-Lücke liegt."
            )
        streamlit.info(
            "Im Gleichgewicht laufen die beiden Gegenränder entgegengesetzt und der "
            "Nettostrom verschwindet. Erst eine chemische Potentialdifferenz erzeugt "
            "den gezeigten Hallstrom. Außerdem ist die Aussage 'nur der Rand trägt "
            "Strom' darstellungsabhängig: Bei einem über die Probe gekippten Potential "
            "können auch Bulk-Zustände beitragen."
        )
        with streamlit.expander("Alle Fermi-Schnittpunkte"):
            crossing_rows = _edge_crossing_rows(edge_spectrum)
            if crossing_rows:
                streamlit.dataframe(crossing_rows, hide_index=True, width="stretch")
            else:
                streamlit.write("Keine Randbranch schneidet das chemische Potential.")

    with conventions_tab:
        streamlit.subheader("Festgelegte Vorzeichen und Tensoren")
        streamlit.latex(r"q=-e,\qquad e>0,\qquad \mathbf B=B\hat{\mathbf z},\quad B>0")
        streamlit.latex(r"\nu=\frac{n_e h}{eB}=\frac{n_e\Phi_0}{B}")
        streamlit.latex(
            r"\begin{pmatrix}E_x\\E_y\end{pmatrix}="
            r"\begin{pmatrix}\rho_{xx}&\rho_{xy}\\-\rho_{xy}&\rho_{xx}\end{pmatrix}"
            r"\begin{pmatrix}J_x\\J_y\end{pmatrix}"
        )
        streamlit.latex(
            r"\begin{pmatrix}J_x\\J_y\end{pmatrix}="
            r"\begin{pmatrix}\sigma_{xx}&\sigma_{xy}\\-\sigma_{xy}&\sigma_{xx}\end{pmatrix}"
            r"\begin{pmatrix}E_x\\E_y\end{pmatrix}"
        )
        streamlit.latex(
            r"\rho_{xx}=0,\quad \rho_{xy}=-\frac{h}{\nu e^2},\quad "
            r"\sigma_{xx}=0,\quad \sigma_{xy}=+\nu\frac{e^2}{h}"
        )
        streamlit.markdown(
            f"Der Widerstandsquant beträgt **R_K = h/e² = "
            f"{VON_KLITZING_CONSTANT_OHM:.8f} Ω**. Die App invertiert die beiden "
            "Tensoren numerisch und testet diese Identität."
        )
        streamlit.subheader("Spin-Modus")
        streamlit.write(
            "Spinlos bzw. Zeeman-aufgelöst erlaubt Schritte Δν=1. Bei vollständig "
            "unaufgelöster zweifacher Spinentartung verwendet die Plateau-Vorschau Δν=2."
        )

    with reproducibility_tab:
        reconstructed_density = float(np.sum(state.density_contributions_m2))
        relative_error = abs(reconstructed_density - parameters.electron_density_m2) / (
            parameters.electron_density_m2
        )
        checks = streamlit.columns(3)
        checks[0].metric("Dichterekonstruktion", f"{reconstructed_density / 1e15:.8g} x10¹⁵ m⁻²")
        checks[1].metric("Relativer Dichtefehler", f"{relative_error:.3e}")
        checks[2].metric("Spinmodus", _SPIN_LABELS[parameters.spin_mode])
        streamlit.markdown(
            "Die Besetzungen werden bei T>0 aus der Fermi-Dirac-Verteilung und der "
            "Teilchenzahlgleichung gelöst. Bei T=0 liegt μ für vollständig gefüllte "
            "Niveaus in der Mitte der Energielücke."
        )
        streamlit.download_button(
            "Vollständige Parameter als JSON herunterladen",
            data=json.dumps(asdict(parameters), indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="integer_quantum_hall_parameters.json",
            mime="application/json",
            key="iqhe-parameters-json",
        )
        streamlit.caption(
            "Quelle: David Tong, The Quantum Hall Effect, Abschnitt 2.1, "
            "arXiv:1606.06687v2. Erläuterungen und Abbildungen wurden für dieses "
            "Lernlabor neu erstellt."
        )


__all__ = ["render_integer_quantum_hall_inputs", "show_integer_quantum_hall_lab"]
