"""Streamlit presentation for the complete Section 1.4 Landau-level lab."""

from __future__ import annotations

import json
from dataclasses import asdict
from math import pi
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.quantum_hall.landau_levels import (
    BOLTZMANN_CONSTANT,
    LandauLevelParameters,
    landau_gauge_state,
    landau_level_energies_joule,
    landau_scales,
)
from toposc_lab.visualization.export import figure_to_bytes
from toposc_lab.visualization.landau_level_plots import (
    JOULE_TO_MEV,
    cyclotron_drift_animation,
    degeneracy_figure,
    landau_gauge_state_figure,
    landau_spectrum_figure,
    symmetric_gauge_state_figure,
)


def _figure_downloads(
    streamlit: Any,
    figure: plt.Figure,
    *,
    filename_stem: str,
    key: str,
) -> None:
    """Offer the same publication exports as the main workspace."""
    png_column, pdf_column = streamlit.columns(2)
    png_column.download_button(
        "PNG herunterladen (300 dpi)",
        data=figure_to_bytes(figure, file_format="png", dpi=300),
        file_name=f"{filename_stem}.png",
        mime="image/png",
        key=f"landau-png::{key}",
    )
    pdf_column.download_button(
        "PDF herunterladen (Vektor)",
        data=figure_to_bytes(figure, file_format="pdf", dpi=300),
        file_name=f"{filename_stem}.pdf",
        mime="application/pdf",
        key=f"landau-pdf::{key}",
    )


def render_landau_level_inputs(streamlit: Any) -> LandauLevelParameters:
    """Render every physical and visual parameter used by the lab."""
    streamlit.caption("Alle Regler wirken sofort auf Formeln, Graphen und Simulation.")

    streamlit.subheader("Feld und Material")
    magnetic_field = float(
        streamlit.number_input(
            "Magnetfeld B (T)",
            min_value=0.01,
            max_value=30.0,
            value=1.0,
            step=0.1,
            format="%.3f",
            help="B zeigt senkrecht zur Ebene in +z-Richtung.",
            key="landau-magnetic-field",
        )
    )
    effective_mass_ratio = float(
        streamlit.number_input(
            "Effektive Masse m*/m_e",
            min_value=0.005,
            max_value=5.0,
            value=0.067,
            step=0.01,
            format="%.4f",
            help="0,067 entspricht ungefähr GaAs; 1 entspricht einem freien Elektron.",
            key="landau-effective-mass",
        )
    )
    electric_field = float(
        streamlit.number_input(
            "Elektrisches Feld E_x (V/m)",
            min_value=-1_000_000.0,
            max_value=1_000_000.0,
            value=0.0,
            step=100.0,
            help="Ein Vorzeichenwechsel kehrt die E×B-Drift um.",
            key="landau-electric-field",
        )
    )
    include_zeeman = bool(
        streamlit.checkbox(
            "Zeeman-Aufspaltung anzeigen",
            value=False,
            key="landau-include-zeeman",
        )
    )
    g_factor = float(
        streamlit.number_input(
            "Effektiver g-Faktor",
            min_value=-20.0,
            max_value=20.0,
            value=-0.44,
            step=0.1,
            format="%.3f",
            help="Der Betrag setzt die Spinaufspaltung; das Vorzeichen vertauscht die Reihenfolge.",
            key="landau-g-factor",
        )
    )

    streamlit.subheader("Zustand und Darstellung")
    maximum_level = int(
        streamlit.slider(
            "Höchstes gezeichnetes Landau-Niveau n_max",
            min_value=0,
            max_value=10,
            value=5,
            key="landau-maximum-level",
        )
    )
    selected_level = int(
        streamlit.slider(
            "Ausgewähltes Niveau n",
            min_value=0,
            max_value=maximum_level,
            value=min(0, maximum_level),
            key="landau-selected-level",
        )
    )
    wave_number_l_b = float(
        streamlit.slider(
            "Landau-Gauge-Wellenzahl k l_B",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            key="landau-wave-number",
        )
    )
    angular_momentum = int(
        streamlit.slider(
            "Symmetrische Gauge: Drehimpuls m",
            min_value=0,
            max_value=30,
            value=0,
            key="landau-angular-momentum",
        )
    )
    view_extent_l_b = float(
        streamlit.slider(
            "Sichtfenster in Einheiten von l_B",
            min_value=3.0,
            max_value=12.0,
            value=6.0,
            step=0.5,
            key="landau-view-extent",
        )
    )

    streamlit.subheader("Probe und Bahn")
    sample_width_um = float(
        streamlit.number_input(
            "Probenbreite L_x (µm)",
            min_value=0.05,
            max_value=100.0,
            value=1.0,
            step=0.1,
            key="landau-sample-width",
        )
    )
    sample_height_um = float(
        streamlit.number_input(
            "Probenhöhe L_y (µm)",
            min_value=0.05,
            max_value=100.0,
            value=1.0,
            step=0.1,
            key="landau-sample-height",
        )
    )
    orbit_radius_l_b = float(
        streamlit.slider(
            "Klassischer Bahnradius R/l_B",
            min_value=0.2,
            max_value=8.0,
            value=2.0,
            step=0.1,
            key="landau-orbit-radius",
        )
    )
    orbit_phase = float(
        streamlit.slider(
            "Startphase φ/π",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            key="landau-orbit-phase",
        )
        * pi
    )

    return LandauLevelParameters(
        magnetic_field_tesla=magnetic_field,
        effective_mass_ratio=effective_mass_ratio,
        electric_field_v_per_m=electric_field,
        sample_width_m=sample_width_um * 1.0e-6,
        sample_height_m=sample_height_um * 1.0e-6,
        maximum_level=maximum_level,
        selected_level=selected_level,
        wave_number_l_b=wave_number_l_b,
        angular_momentum=angular_momentum,
        g_factor=g_factor,
        include_zeeman=include_zeeman,
        view_extent_l_b=view_extent_l_b,
        orbit_radius_l_b=orbit_radius_l_b,
        orbit_phase_radians=orbit_phase,
    )


def _energy_table(parameters: LandauLevelParameters) -> list[dict[str, str | int]]:
    levels = np.arange(parameters.maximum_level + 1)
    energies = landau_level_energies_joule(parameters, levels)
    rows: list[dict[str, str | int]] = []
    for index, energy in zip(levels, energies, strict=True):
        row: dict[str, str | int] = {
            "n": int(index),
            "E_n (meV)": f"{energy * JOULE_TO_MEV:.6g}",
            "E_n/k_B (K)": f"{energy / BOLTZMANN_CONSTANT:.6g}",
        }
        if parameters.include_zeeman:
            spin_up = landau_level_energies_joule(
                parameters,
                np.asarray([index]),
                spin_projection=1,
            )[0]
            spin_down = landau_level_energies_joule(
                parameters,
                np.asarray([index]),
                spin_projection=-1,
            )[0]
            row["E_{s=+1} (meV)"] = f"{spin_up * JOULE_TO_MEV:.6g}"
            row["E_{s=-1} (meV)"] = f"{spin_down * JOULE_TO_MEV:.6g}"
        rows.append(row)
    return rows


def _parameter_rows(parameters: LandauLevelParameters) -> list[dict[str, str]]:
    return [
        {
            "Parameter": "B",
            "Wert": f"{parameters.magnetic_field_tesla:g} T",
            "Bedeutung": "Magnetfeld senkrecht zur Ebene",
        },
        {
            "Parameter": "m*",
            "Wert": f"{parameters.effective_mass_ratio:g} m_e",
            "Bedeutung": "effektive Zyklotronmasse",
        },
        {
            "Parameter": "E_x",
            "Wert": f"{parameters.electric_field_v_per_m:g} V/m",
            "Bedeutung": "elektrisches Feld in x-Richtung",
        },
        {
            "Parameter": "L_x × L_y",
            "Wert": f"{parameters.sample_width_m * 1e6:g} × {parameters.sample_height_m * 1e6:g} µm²",
            "Bedeutung": "Fläche für die Entartungszählung",
        },
        {
            "Parameter": "n",
            "Wert": str(parameters.selected_level),
            "Bedeutung": "Landau-Niveau / Zyklotronquantenzahl",
        },
        {
            "Parameter": "k l_B",
            "Wert": f"{parameters.wave_number_l_b:g}",
            "Bedeutung": "Impulsquantenzahl in Landau-Gauge",
        },
        {
            "Parameter": "m",
            "Wert": str(parameters.angular_momentum),
            "Bedeutung": "Drehimpuls- und Entartungsquantenzahl in symmetrischer Gauge",
        },
        {
            "Parameter": "g",
            "Wert": f"{parameters.g_factor:g}",
            "Bedeutung": "effektiver Zeeman-g-Faktor",
        },
        {
            "Parameter": "R/l_B",
            "Wert": f"{parameters.orbit_radius_l_b:g}",
            "Bedeutung": "Radius der klassischen Bahn",
        },
        {
            "Parameter": "φ",
            "Wert": f"{parameters.orbit_phase_radians / pi:g} π",
            "Bedeutung": "Startphase der Bahn",
        },
    ]


def show_landau_level_lab(streamlit: Any, parameters: LandauLevelParameters) -> None:
    """Render the complete, interactive Section 1.4 learning experience."""
    scales = landau_scales(parameters)
    streamlit.title("1.4 Landau-Niveaus – interaktives Labor")
    streamlit.caption(
        "Vollständige interaktive Umsetzung von Abschnitt 1.4 in David Tong, "
        "The Quantum Hall Effect, arXiv:1606.06687v2 (gedruckte Seiten 14–27)."
    )
    streamlit.info(
        "Konvention: Das Teilchen hat Ladung −e, B zeigt in +z-Richtung und E in "
        "+x-Richtung. Deshalb zeigt eine positive E×B-Drift hier in +y-Richtung."
    )

    metrics = streamlit.columns(6)
    metrics[0].metric("Magnetische Länge l_B", f"{scales.magnetic_length_m * 1e9:.3f} nm")
    metrics[1].metric(
        "Zyklotronenergie ℏω_B", f"{scales.cyclotron_energy_joule * JOULE_TO_MEV:.4g} meV"
    )
    metrics[2].metric(
        "Zyklotronfrequenz", f"{scales.cyclotron_frequency_rad_s / (2 * pi) / 1e12:.4g} THz"
    )
    metrics[3].metric("Flussquanten N_Φ", f"{scales.flux_quanta:.4g}")
    metrics[4].metric("E×B-Drift v_y", f"{scales.drift_velocity_m_s:.4g} m/s")
    metrics[5].metric(
        "E-Feld-Verschiebung",
        f"{scales.electric_displacement_m / scales.magnetic_length_m:.4g} l_B",
    )

    (
        overview_tab,
        quantization_tab,
        landau_gauge_tab,
        electric_field_tab,
        symmetric_gauge_tab,
        degeneracy_tab,
    ) = streamlit.tabs(
        (
            "Überblick",
            "Quantisierung",
            "1.4.1 Landau-Gauge",
            "1.4.2 E-Feld",
            "1.4.3 Symmetrische Gauge",
            "Entartung & Spin",
        )
    )

    with overview_tab:
        streamlit.markdown(
            """
            Ein freies Teilchen besitzt ohne Magnetfeld ein kontinuierliches Spektrum.
            Das Magnetfeld koppelt die beiden mechanischen Impulskomponenten und macht
            daraus einen harmonischen Oszillator. Seine diskreten Energien sind die
            **Landau-Niveaus**. Die zweite ebene Freiheitsgrad erscheint als sehr große
            Entartung: Viele unterschiedliche Wellenfunktionen besitzen dieselbe Energie.
            """
        )
        spectrum_figure, _ = landau_spectrum_figure(parameters)
        streamlit.pyplot(spectrum_figure)
        _figure_downloads(
            streamlit,
            spectrum_figure,
            filename_stem="landau_level_spectrum",
            key="spectrum",
        )
        plt.close(spectrum_figure)

        streamlit.markdown(
            "**So lesen Sie die Graphen:** Links wachsen Abstand und Energie der "
            "Niveaus linear mit B. Rechts sind alle Zustände eines Niveaus bei E=0 "
            "flach und damit in k entartet; ein E-Feld kippt diese Linien."
        )
        with streamlit.expander("Alle aktuellen Parameter und reproduzierbarer Export"):
            streamlit.dataframe(
                _parameter_rows(parameters), hide_index=True, width="stretch"
            )
            streamlit.download_button(
                "Parameter als JSON herunterladen",
                data=json.dumps(asdict(parameters), indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="landau_level_parameters.json",
                mime="application/json",
                key="landau-parameters-json",
            )

    with quantization_tab:
        streamlit.subheader("Vom geladenen Teilchen zum harmonischen Oszillator")
        streamlit.markdown(
            "Der kanonische Impuls p ist eichabhängig. Physikalisch messbar ist der "
            "mechanische Impuls π = p + eA. Seine Komponenten kommutieren im Magnetfeld nicht."
        )
        streamlit.latex(
            r"H=\frac{1}{2m^\ast}(\mathbf p+e\mathbf A)^2=\frac{\boldsymbol\pi^2}{2m^\ast}"
        )
        streamlit.latex(r"[\pi_x,\pi_y]=-i e\hbar B")
        streamlit.markdown("Mit Leiteroperatoren wird genau die Oszillatoralgebra sichtbar:")
        streamlit.latex(r"a=\frac{\pi_x-i\pi_y}{\sqrt{2e\hbar B}},\qquad [a,a^\dagger]=1")
        streamlit.latex(
            r"H=\hbar\omega_B\left(a^\dagger a+\frac12\right),\qquad \omega_B=\frac{eB}{m^\ast}"
        )
        streamlit.latex(r"E_n=\hbar\omega_B\left(n+\frac12\right),\qquad n=0,1,2,\ldots")
        streamlit.dataframe(_energy_table(parameters), hide_index=True, width="stretch")
        streamlit.markdown(
            "**Wichtig:** Diese Algebra bestimmt die Energien, aber noch nicht die "
            "zweite Quantenzahl. Die fehlende ebene Freiheitsgrad steckt in der Entartung, "
            "die in den beiden Eichwahlen unterschiedlich sichtbar wird."
        )

    with landau_gauge_tab:
        streamlit.subheader("Streifenförmige Zustände in rechteckiger Geometrie")
        streamlit.latex(
            r"\mathbf A=Bx\,\hat{\mathbf y},\qquad H=\frac{1}{2m^\ast}\left[p_x^2+(p_y+eBx)^2\right]"
        )
        streamlit.latex(r"\psi_{n,k}(x,y)\propto e^{iky}H_n(x/l_B+k l_B)e^{-(x/l_B+k l_B)^2/2}")
        streamlit.latex(r"l_B=\sqrt{\frac{\hbar}{eB}},\qquad x_c=-k l_B^2")
        streamlit.markdown(
            "Die ebene Welle ist in y ausgedehnt; quer dazu sitzt ein harmonischer "
            "Oszillator am Leitzentrum x_c. k verändert die Lage, aber nicht die Energie."
        )
        gauge_figure, _ = landau_gauge_state_figure(parameters)
        streamlit.pyplot(gauge_figure)
        _figure_downloads(
            streamlit,
            gauge_figure,
            filename_stem="landau_gauge_wavefunction",
            key="landau-gauge",
        )
        plt.close(gauge_figure)
        zero_field_state = landau_gauge_state(parameters)
        if abs(zero_field_state.center_over_l_b) > parameters.view_extent_l_b:
            streamlit.warning(
                "Das gewählte Leitzentrum liegt außerhalb des Sichtfensters. "
                "Verringern Sie |k l_B| oder vergrößern Sie das Sichtfenster."
            )
        streamlit.markdown(
            "Die Dichte ist entlang y konstant, obwohl die Phase e^{iky} dort variiert. "
            "Linearkombinationen der entarteten k-Zustände können ganz andere räumliche "
            "Profile erzeugen."
        )

    with electric_field_tab:
        streamlit.subheader("Elektrisches Feld: gekippte Niveaus und E×B-Drift")
        streamlit.latex(r"H=\frac{1}{2m^\ast}\left[p_x^2+(p_y+eBx)^2\right]-eE_xx")
        streamlit.latex(
            r"E_{n,k}=\hbar\omega_B\left(n+\frac12\right)+eE_x k l_B^2-\frac{m^\ast}{2}\left(\frac{E_x}{B}\right)^2"
        )
        streamlit.latex(r"v_y=\frac{1}{\hbar}\frac{\partial E_{n,k}}{\partial k}=\frac{E_x}{B}")
        streamlit.markdown(
            "Das E-Feld hebt die k-Entartung auf. Die Wellenpakete driften senkrecht "
            "zu E und B; ihre Geschwindigkeit ist unabhängig von n und von der Masse."
        )
        if parameters.electric_field_v_per_m == 0.0:
            streamlit.info(
                "Setzen Sie E_x in der Seitenleiste ungleich null, um die Verschiebung, "
                "die geneigten Niveaus und die Drift zu sehen."
            )
        shifted_figure, _ = landau_gauge_state_figure(
            parameters,
            include_electric_shift=True,
        )
        streamlit.pyplot(shifted_figure)
        _figure_downloads(
            streamlit,
            shifted_figure,
            filename_stem="landau_electric_field_shift",
            key="electric-shift",
        )
        plt.close(shifted_figure)

        streamlit.plotly_chart(
            cyclotron_drift_animation(parameters),
            width="stretch",
            config={"displaylogo": False, "responsive": True},
        )
        streamlit.caption(
            "Die graue Kurve zeigt zwei Zyklotronperioden. Das orange Teilchen "
            "umkreist das blaue Leitzentrum, das mit E_x/B entlang y driftet."
        )
        streamlit.markdown(
            "Vorzeichenhinweis: Durch quadratische Ergänzung liegt das Oszillatorzentrum "
            "für den hier geschriebenen Hamiltonoperator bei "
            "x_c = −k l_B² + m*E_x/(eB²). Diese Gleichung und die Wellenfunktionsverschiebung "
            "sind konsistent; eine gegenteilige Ladungs- oder Potentialkonvention kehrt das Vorzeichen um."
        )

    with symmetric_gauge_tab:
        streamlit.subheader("Rotationssymmetrische Basis und Lowest Landau Level")
        streamlit.latex(
            r"\mathbf A=-\frac12\mathbf r\times\mathbf B=-\frac{yB}{2}\hat{\mathbf x}+\frac{xB}{2}\hat{\mathbf y}"
        )
        streamlit.markdown(
            "Neben a,a† existiert jetzt ein zweites Leiteroperatorpaar b,b†. a† erhöht "
            "die Energiequantenzahl n; b† erhöht m, ohne die Energie zu verändern."
        )
        streamlit.latex(
            r"|n,m\rangle=\frac{(a^\dagger)^n(b^\dagger)^m}{\sqrt{n!m!}}|0,0\rangle,\qquad E_{n,m}=E_n"
        )
        streamlit.latex(
            r"z=x-iy,\qquad \psi_{\mathrm{LLL},m}\propto\left(\frac{z}{l_B}\right)^m e^{-|z|^2/(4l_B^2)}"
        )
        streamlit.latex(
            r"J\psi_{\mathrm{LLL},m}=\hbar m\psi_{\mathrm{LLL},m},\qquad r_{\mathrm{max}}=\sqrt{2m}\,l_B"
        )
        symmetric_figure, _ = symmetric_gauge_state_figure(parameters)
        streamlit.pyplot(symmetric_figure)
        _figure_downloads(
            streamlit,
            symmetric_figure,
            filename_stem="symmetric_gauge_lll",
            key="symmetric-gauge",
        )
        plt.close(symmetric_figure)
        streamlit.markdown(
            "Das Skript verwendet bewusst z=x−iy (entgegen einer häufigen Konvention), "
            "damit die LLL-Wellenfunktionen für B>0 holomorph in z sind. Höhere m verschieben "
            "die Dichte auf größere Ringe."
        )

    with degeneracy_tab:
        streamlit.subheader("Warum jedes Niveau makroskopisch viele Zustände besitzt")
        streamlit.latex(
            r"\Phi_0=\frac{2\pi\hbar}{e}=\frac{h}{e},\qquad \mathcal N=\frac{BA}{\Phi_0}=\frac{A}{2\pi l_B^2}"
        )
        degeneracy_plot, _ = degeneracy_figure(parameters)
        streamlit.pyplot(degeneracy_plot)
        _figure_downloads(
            streamlit,
            degeneracy_plot,
            filename_stem="landau_level_degeneracy",
            key="degeneracy",
        )
        plt.close(degeneracy_plot)
        streamlit.markdown(
            f"Die gewählte Probe enthält **{scales.flux_quanta:.6g} Flussquanten**. "
            "Auf einem endlichen Torus muss der Gesamtfluss kompatibel quantisiert sein; "
            "bei den frei wählbaren Rechteckmaßen ist N_Φ deshalb eine kontinuierliche "
            "Flusszählung und nicht immer eine ganze Hilbertraumdimension."
        )

        streamlit.latex(
            r"X=x-\frac{\pi_y}{m^\ast\omega_B},\qquad Y=y+\frac{\pi_x}{m^\ast\omega_B},\qquad [X,Y]=i l_B^2"
        )
        streamlit.warning(
            "Präzisierung zur letzten Skriptseite: Aus dem Kommutator folgt die "
            "Robertson-Grenze ΔX·ΔY ≥ l_B²/2. Die Fläche 2πl_B² ist dagegen die "
            "semiklassische Zustandsfläche, die zur Entartungszählung A/(2πl_B²) führt. "
            "Die App hält Unsicherheitsprodukt und Zustandsfläche bewusst getrennt."
        )

        streamlit.subheader("Spin-Hinweis aus Abschnitt 1.4")
        zeeman_ratio = abs(scales.zeeman_splitting_joule) / scales.cyclotron_energy_joule
        streamlit.latex(r"\Delta_Z=g\mu_BB,\qquad \Delta_{\mathrm{LL}}=\hbar\omega_B")
        spin_metrics = streamlit.columns(3)
        spin_metrics[0].metric(
            "|Δ_Z|", f"{abs(scales.zeeman_splitting_joule) * JOULE_TO_MEV:.4g} meV"
        )
        spin_metrics[1].metric("Δ_LL", f"{scales.cyclotron_energy_joule * JOULE_TO_MEV:.4g} meV")
        spin_metrics[2].metric("|Δ_Z| / Δ_LL", f"{zeeman_ratio:.4g}")
        streamlit.markdown(
            "Für freie Elektronen mit m*=m_e und g≈2 sind beide Skalen gleich. In "
            "Festkörpern ändern Bandstruktur, effektive Masse und g-Faktor die Reihenfolge. "
            "Mit den GaAs-Startwerten ist die Zeeman-Energie ungefähr 68-mal kleiner als "
            "die Zyklotronenergie – genau der Größenordnungshinweis aus dem Skript."
        )

        streamlit.caption(
            "Quelle und fachliche Gliederung: David Tong, The Quantum Hall Effect, "
            "Abschnitt 1.4, arXiv:1606.06687v2. Texte wurden für dieses interaktive "
            "Lernmodul neu formuliert; Gleichungen verwenden die oben genannte Konvention."
        )


__all__ = ["render_landau_level_inputs", "show_landau_level_lab"]
