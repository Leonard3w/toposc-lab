"""Erste interaktive Research-Workspace-Oberflaeche fuer TopOSC-Lab.

Die App enthaelt absichtlich keine Modellphysik. Sie liest die registrierten
Pydantic-Parameter, baut ein Modell ueber die Registry und verwendet danach
dieselben Solver, Observablen und Plotfunktionen wie die Python-API.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from types import UnionType
from typing import Any, get_args, get_origin

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.app.registry import MODEL_REGISTRY, ModelSpec
from toposc_lab.app.model_guides import model_guide, observable_guides
from toposc_lab.bosons.ideal_bose_gas import (
    BOLTZMANN_CONSTANT,
    IdealBoseEinsteinCondensationParameters,
    IdealBoseGasParameters,
)
from toposc_lab.gases.ideal_quantum_gases import (
    IdealQuantumGasParameters,
    QuantumGasResult,
    analyze_ideal_quantum_gas,
)
from toposc_lab.gases.ensembles import (
    BoseGrandCanonicalParameters,
    BoseMicrocanonicalParameters,
    ClassicalCanonicalParameters,
    ClassicalEnsembleResult,
    ClassicalGrandCanonicalParameters,
    ClassicalMicrocanonicalParameters,
    analyze_bose_grand_canonical,
    analyze_bose_microcanonical,
    analyze_classical_canonical,
    analyze_classical_grand_canonical,
    analyze_classical_microcanonical,
    sample_classical_particles,
)
from toposc_lab.app.study_workspace import (
    common_scalar_observables,
    scan_parameter_name,
    study_summary,
)
from toposc_lab.data.studies import (
    StudyData,
    StudyMetadata,
    study_from_parameter_scan,
)
from toposc_lab.observables.localization import localization_profile_from_result
from toposc_lab.observables.spectrum import (
    bulk_gap,
    count_zero_modes,
    lowest_abs_energy,
)
from toposc_lab.observables.symmetries import check_hermitian
from toposc_lab.scans.analysis import analyze_parameter_scan
from toposc_lab.scans.model_scan import model_parameter_scan
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.visualization.lattice_plots import plot_localization_on_lattice
from toposc_lab.visualization.matrix_plots import plot_matrix
from toposc_lab.visualization.ideal_bose_gas_lab import (
    ideal_bec_learning_figure,
    ideal_bose_gas_live_material,
)
from toposc_lab.visualization.quantum_gas_plots import (
    MODE_OCCUPATION_LOG_RANGE,
    bose_condensation_figure,
    quantum_gas_state_figure,
    quantum_statistics_schematic,
)
from toposc_lab.visualization.ensemble_plots import (
    bose_grand_canonical_figure,
    bose_microcanonical_figure,
    classical_ensemble_motion_figure,
)
from toposc_lab.visualization.plots import plot_eigenvalue_spectrum
from toposc_lab.visualization.study_plots import (
    plot_study_comparison,
    plot_study_observable,
    plot_study_spectrum,
)
from toposc_lab.visualization.export import figure_to_bytes
from toposc_lab.visualization.style import add_panel_label, paper_style


def _streamlit_module() -> Any:
    """Importiere Streamlit erst beim Start der optionalen App."""
    try:
        import streamlit
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The graphical interface needs Streamlit. Install it with "
            "pip install -e '.[app]'."
        ) from error

    return streamlit


def _parameter_label(name: str) -> str:
    """Formatiere einen technischen Parameternamen fuer die Benutzeroberflaeche."""
    return name.replace("_", " ").capitalize()


def _scannable_parameter_names(specification: ModelSpec) -> tuple[str, ...]:
    """Bestimme kontinuierliche Pydantic-Parameter direkt fuer den Scan-Dialog."""
    names: list[str] = []

    for name, parameter in specification.parameter_model.model_fields.items():
        possible_types = (parameter.annotation, *get_args(parameter.annotation))

        if float in possible_types:
            names.append(name)

    return tuple(names)


def _scan_default(
    specification: ModelSpec,
    parameter_name: str,
) -> tuple[float, float, int]:
    """Liefere gespeicherte oder automatisch bestimmte Startwerte fuer Scans."""
    scan_defaults = getattr(specification, "scan_defaults", {})

    if parameter_name in scan_defaults:
        return scan_defaults[parameter_name]

    center = float(specification.validated_default_parameters()[parameter_name])
    span = max(abs(center), 1.0)
    return center - span, center + span, 61


def _base_annotation(annotation: Any) -> tuple[type[Any] | Any, bool]:
    """Entferne optional ``None`` aus einer Pydantic-Typannotation."""
    arguments = get_args(annotation)
    origin = get_origin(annotation)

    if origin in (UnionType, None) or arguments:
        non_none_arguments = [argument for argument in arguments if argument is not type(None)]

        if len(non_none_arguments) == 1 and len(arguments) != 1:
            return non_none_arguments[0], True

    return annotation, False


def _integer_lower_bound(field: Any) -> int | None:
    """Lese eine einfache ganzzahlige Untergrenze aus Pydantic-Metadaten."""
    lower_bound: int | None = None

    for constraint in field.metadata:
        if hasattr(constraint, "ge") and constraint.ge is not None:
            lower_bound = int(constraint.ge)
        elif hasattr(constraint, "gt") and constraint.gt is not None:
            lower_bound = int(constraint.gt) + 1

    return lower_bound


def _render_parameter_inputs(streamlit: Any, specification: ModelSpec) -> dict[str, Any]:
    """Erzeuge aus den Pydantic-Feldern eines Modells passende Widgets."""
    parameter_values: dict[str, Any] = {}
    defaults = specification.validated_default_parameters()

    for name, field in specification.parameter_model.model_fields.items():
        default = defaults[name]
        label = _parameter_label(name)
        key = f"parameter::{specification.key}::{name}"
        help_text = field.description
        annotation, is_optional = _base_annotation(field.annotation)

        # Alle bisher vorhandenen Randbedingungen haben diese zwei Optionen.
        if name.startswith("boundary"):
            options = ("open", "periodic")
            parameter_values[name] = streamlit.selectbox(
                label,
                options=options,
                index=options.index(default),
                help=help_text,
                key=key,
            )
        elif is_optional and annotation is int:
            value = streamlit.text_input(
                f"{label} (empty = random)",
                value="" if default is None else str(default),
                help=help_text,
                key=key,
            )
            # Pydantic validiert die Eingabe erst beim Klick auf Run. Dadurch
            # wird eine unvollstaendige Texteingabe nicht zum App-Absturz.
            parameter_values[name] = None if not value.strip() else value
        elif annotation is int:
            parameter_values[name] = int(
                streamlit.number_input(
                    label,
                    min_value=_integer_lower_bound(field),
                    value=int(default),
                    step=1,
                    help=help_text,
                    key=key,
                )
            )
        elif annotation is float:
            parameter_values[name] = float(
                streamlit.number_input(
                    label,
                    value=float(default),
                    step=0.1,
                    help=help_text,
                    key=key,
                )
            )
        else:
            parameter_values[name] = streamlit.text_input(
                label,
                value=str(default),
                help=help_text,
                key=key,
            )

    return parameter_values


def _solve_serialized_model(model_key: str, serialized_parameters: str) -> Any:
    """Baue und loese ein Modell aus einem stabilen Cache-Schluessel."""
    specification = MODEL_REGISTRY.get(model_key)
    parameter_values = json.loads(serialized_parameters)
    model = specification.build(parameter_values)

    return ExactDiagonalizationSolver().solve_model(model)


def _simulation_archive(result: Any) -> bytes:
    """Erzeuge einen portablen Download des vollstaendigen Einzelergebnisses."""
    metadata = {
        "model_name": result.model_name,
        "parameters": result.parameters,
        "basis_layout": {
            "spatial_shape": result.basis_layout.spatial_shape,
            "components_per_site": result.basis_layout.components_per_site,
            "ordering": result.basis_layout.ordering,
            "component_labels": result.basis_layout.component_labels,
        },
        "metadata": result.metadata,
    }
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        eigenvalues=result.eigenvalues,
        eigenvectors=result.eigenvectors,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return buffer.getvalue()


def _study_archive(study: StudyData) -> bytes:
    """Erzeuge einen sicheren NPZ-Download fuer eine im UI erzeugte Studie."""
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        __toposc_lab_metadata_json__=np.asarray(
            study.metadata.model_dump_json()
        ),
        **study.arrays,
    )
    return buffer.getvalue()


def _study_from_uploaded_bytes(data: bytes) -> StudyData:
    """Load an uploaded study without relying on a long-lived module cache.

    Streamlit keeps imported modules alive while its source file is reloaded.
    Parsing the small public NPZ schema here makes uploads reliable after a
    package update as well.  Like the data API, this deliberately forbids
    Pickle payloads.
    """
    if not isinstance(data, bytes):
        raise TypeError("uploaded study data must be bytes")

    metadata_key = "__toposc_lab_metadata_json__"
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            if metadata_key not in archive.files:
                raise ValueError("missing TopOSC-Lab metadata")

            metadata_value = archive[metadata_key]
            if metadata_value.ndim != 0:
                raise ValueError("study metadata must be a scalar JSON value")

            metadata = StudyMetadata.model_validate_json(str(metadata_value.item()))
            arrays = {
                name: archive[name].copy()
                for name in archive.files
                if name != metadata_key
            }
    except (OSError, ValueError) as error:
        raise ValueError("could not load the uploaded .npz study") from error

    return StudyData(metadata=metadata, arrays=arrays)


def _add_figure_downloads(
    streamlit: Any,
    figure: plt.Figure,
    *,
    filename_stem: str,
    key: str,
) -> None:
    """Add high-resolution PNG and vector-PDF downloads below a figure."""
    png_column, pdf_column = streamlit.columns(2)
    png_column.download_button(
        "Download PNG (300 dpi)",
        data=figure_to_bytes(figure, file_format="png", dpi=300),
        file_name=f"{filename_stem}.png",
        mime="image/png",
        key=f"png::{key}",
    )
    pdf_column.download_button(
        "Download PDF (vector)",
        data=figure_to_bytes(figure, file_format="pdf", dpi=300),
        file_name=f"{filename_stem}.pdf",
        mime="application/pdf",
        key=f"pdf::{key}",
    )


def _format_complex(value: complex, *, precision: int = 3) -> str:
    """Format a numerical Hamiltonian entry compactly for a matrix table."""
    real = float(np.real(value))
    imaginary = float(np.imag(value))
    threshold = 10.0 ** (-(precision + 2))

    if abs(imaginary) < threshold:
        return f"{real:.{precision}g}"
    if abs(real) < threshold:
        return f"{imaginary:.{precision}g}i"
    return f"{real:.{precision}g}{imaginary:+.{precision}g}i"


def _show_model_guide(
    streamlit: Any,
    specification: ModelSpec,
    parameter_values: dict[str, Any],
) -> None:
    """Show the formula, implementation steps and numerical matrix of a model."""
    guide = model_guide(specification.key)
    model = specification.build(parameter_values)
    layout = model.basis_layout

    streamlit.subheader(guide.title)
    streamlit.write(guide.summary)

    overview_tab, parameters_tab, matrix_tab, observables_tab = streamlit.tabs(
        ("Hamiltonian", "Parameters and basis", "Numerical matrix", "Observables")
    )

    with overview_tab:
        streamlit.latex(guide.hamiltonian_latex)
        streamlit.markdown("**How the code constructs this matrix**")
        for index, step in enumerate(guide.construction_steps, start=1):
            streamlit.markdown(f"{index}. {step}")
        streamlit.markdown("**Assumptions**")
        for assumption in guide.assumptions:
            streamlit.markdown(f"- {assumption}")
        if guide.reference is not None:
            streamlit.caption(f"Reference: {guide.reference}")

    with parameters_tab:
        parameter_rows = []
        for parameter in guide.parameters:
            parameter_rows.append(
                {
                    "parameter": parameter.name,
                    "symbol": parameter.symbol,
                    "current value": parameter_values.get(parameter.name, "-"),
                    "physical meaning": parameter.meaning,
                    "effect in the matrix": parameter.numerical_role,
                }
            )
        streamlit.dataframe(parameter_rows, hide_index=True, use_container_width=True)
        streamlit.markdown("**Basis used by the numerical Hamiltonian**")
        streamlit.write(guide.basis_description)
        streamlit.json(
            {
                "spatial shape": layout.spatial_shape,
                "spatial sites": layout.n_sites,
                "components per site": layout.components_per_site,
                "components": layout.component_labels,
                "ordering": layout.ordering,
                "Hamiltonian dimension": layout.dimension,
            }
        )

    with matrix_tab:
        if layout.dimension > 1024:
            streamlit.warning(
                "The selected matrix is larger than 1024 x 1024. Reduce the "
                "lattice size to inspect its dense numerical representation."
            )
        else:
            hamiltonian = model.hamiltonian()
            streamlit.caption(
                "This is the actual H built with the current parameters, not a schematic matrix."
            )
            representation_label = streamlit.selectbox(
                "Matrix representation",
                options=("Magnitude |H|", "Real part", "Imaginary part", "Phase"),
                key=f"matrix-representation::{specification.key}",
            )
            representations = {
                "Magnitude |H|": "magnitude",
                "Real part": "real",
                "Imaginary part": "imaginary",
                "Phase": "phase",
            }
            figure, _ = plot_matrix(
                hamiltonian,
                representation=representations[representation_label],
                title=f"{guide.title}: {representation_label}",
                show=False,
            )
            streamlit.pyplot(figure)
            _add_figure_downloads(
                streamlit,
                figure,
                filename_stem=f"{specification.key}_hamiltonian_{representations[representation_label]}",
                key=f"matrix::{specification.key}::{representation_label}",
            )
            plt.close(figure)

            maximum_block = min(layout.dimension, 32)
            default_block = min(layout.dimension, 12)
            block_size = int(
                streamlit.slider(
                    "Leading matrix block to display",
                    min_value=1,
                    max_value=maximum_block,
                    value=default_block,
                    key=f"matrix-block::{specification.key}",
                )
            )
            matrix_block = hamiltonian[:block_size, :block_size]
            formatted_block = np.asarray(
                [[_format_complex(value) for value in row] for row in matrix_block]
            )
            streamlit.markdown(
                f"**Top-left {block_size} x {block_size} block of H** "
                "(rows i, columns j)"
            )
            streamlit.dataframe(formatted_block, use_container_width=True)

    with observables_tab:
        streamlit.caption(
            "These are the definitions used by the Python API, scans and workspace."
        )
        for observable in observable_guides():
            with streamlit.expander(observable.label):
                streamlit.latex(observable.formula_latex)
                streamlit.markdown(f"**Calculation:** {observable.calculation}")
                streamlit.markdown(f"**Meaning:** {observable.interpretation}")


def _render_ideal_bose_gas_lab_inputs(
    streamlit: Any,
) -> tuple[IdealBoseGasParameters, IdealBoseEinsteinCondensationParameters]:
    """Render externally imposed Bose-gas conditions without a scan slider."""
    streamlit.caption(
        "Change a condition, then read the Bose-versus-classical comparison below."
    )
    temperature_nk = float(
        streamlit.number_input(
            "Temperature T (nK)",
            min_value=1.0,
            value=80.0,
            step=1.0,
            help="Thermal reservoir temperature. Higher T populates more excited modes.",
            key="bose-lab-temperature",
        )
    )
    chemical_potential_nk = float(
        streamlit.number_input(
            "Chemical potential mu/k_B (nK)",
            min_value=-500.0,
            max_value=-0.001,
            value=-25.0,
            step=1.0,
            help="For bosons in this box mu must stay below the ground-state energy epsilon_0=0.",
            key="bose-lab-chemical-potential",
        )
    )
    box_length_um = float(
        streamlit.number_input(
            "Periodic box length L (um)",
            min_value=1.0,
            value=20.0,
            step=1.0,
            help="L sets the momentum spacing: k_i=2 pi n_i/L.",
            key="bose-lab-box-length",
        )
    )
    mass_amu = float(
        streamlit.number_input(
            "Particle mass (u)",
            min_value=0.1,
            value=87.0,
            step=1.0,
            help="87 u is approximately rubidium-87.",
            key="bose-lab-mass",
        )
    )
    maximum_mode_index = int(
        streamlit.number_input(
            "Maximum momentum quantum number |n_x|, |n_y|",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            help="The visible mode set contains (2 n_max + 1)^2 momentum states.",
            key="bose-lab-mode-cutoff",
        )
    )
    particle_number = int(
        streamlit.number_input(
            "Total particle number N (BEC experiment)",
            min_value=100,
            max_value=10_000_000,
            value=100_000,
            step=1_000,
            help="Fixed total particle number used only for the three-dimensional BEC tab.",
            key="bose-lab-bec-particle-number",
        )
    )
    streamlit.caption(
        "For the BEC tab, L is interpreted as the side length of a three-dimensional cubic box."
    )
    statistics_parameters = IdealBoseGasParameters(
        temperature=temperature_nk * 1.0e-9,
        chemical_potential=chemical_potential_nk * 1.0e-9 * BOLTZMANN_CONSTANT,
        box_length=box_length_um * 1.0e-6,
        particle_mass_amu=mass_amu,
        maximum_mode_index=maximum_mode_index,
    )
    condensation_parameters = IdealBoseEinsteinCondensationParameters(
        temperature=temperature_nk * 1.0e-9,
        particle_number=particle_number,
        box_length=box_length_um * 1.0e-6,
        particle_mass_amu=mass_amu,
    )
    return statistics_parameters, condensation_parameters


def _show_ideal_bose_gas_lab(
    streamlit: Any,
    parameters: IdealBoseGasParameters,
    condensation_parameters: IdealBoseEinsteinCondensationParameters,
) -> None:
    """Show an educational Bose-versus-classical statistics experiment."""
    comparison, figure = ideal_bose_gas_live_material(parameters)
    result = comparison.bose
    streamlit.subheader("Why are bosons different? A state-occupation experiment")
    streamlit.caption(
        "Every panel uses the same finite two-dimensional box and the same mean number "
        "of particles. Only the statistical law is different."
    )
    metrics = streamlit.columns(5)
    metrics[0].metric("Expected particle number <N>", f"{result.expected_particle_number:.3f}")
    metrics[1].metric("Bose fraction in k = 0", f"{comparison.bose_ground_state_fraction:.1%}")
    metrics[2].metric("Classical fraction in k = 0", f"{comparison.classical_ground_state_fraction:.1%}")
    metrics[3].metric("Bosonic ground-state enhancement", f"{comparison.ground_state_enhancement:.2f}x")
    metrics[4].metric("Effectively populated Bose modes", f"{result.effective_number_of_modes:.1f}")

    material_tab, condensation_tab, experiment_tab, equations_tab, reading_tab = streamlit.tabs(
        (
            "See the statistics",
            "Bose-Einstein condensation",
            "Try these experiments",
            "Equations and calculation",
            "How to read it",
        )
    )
    with material_tab:
        streamlit.plotly_chart(
            figure,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
        streamlit.caption(
            "Each square is one momentum state (n_x, n_y). Bright means that a large "
            "fraction of the gas occupies that state. Panel (c) is the key: red states are "
            "more populated by bosons than by classical particles; blue states are depleted."
        )

    with condensation_tab:
        condensation_result, condensation_figure = ideal_bec_learning_figure(
            condensation_parameters
        )
        condensation_metrics = streamlit.columns(4)
        condensation_metrics[0].metric(
            "Critical temperature T_c",
            f"{condensation_result.critical_temperature / 1.0e-9:.2f} nK",
        )
        condensation_metrics[1].metric(
            "Reduced temperature T / T_c",
            f"{condensation_result.reduced_temperature:.3f}",
        )
        condensation_metrics[2].metric(
            "Condensate fraction N0 / N",
            f"{condensation_result.condensate_fraction:.2%}",
        )
        condensation_metrics[3].metric(
            "Condensate population N0",
            f"{condensation_result.condensate_number:,.0f}",
        )
        streamlit.plotly_chart(
            condensation_figure,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
        streamlit.markdown(
            "This is the standard **three-dimensional, homogeneous, ideal Bose gas at fixed "
            "total particle number**. Below \\(T_c\\), thermal excited states cannot hold all "
            "particles; the remainder occupies the single ground state \\(k=0\\). That "
            "macroscopic ground-state population is Bose--Einstein condensation.\n\n"
            "The statistics tab deliberately remains a finite **two-dimensional** box. It shows "
            "the distribution mechanism clearly, but it does not claim a true finite-temperature "
            "thermodynamic BEC transition."
        )

    with experiment_tab:
        streamlit.markdown(
            "### First experiment: cool the gas\n"
            "Keep **mu/k_B = -25 nK** and compare **T = 300 nK**, **120 nK** and **30 nK**. "
            "At lower temperature the bright centre of panel (a) becomes stronger: bosons "
            "prefer the lowest momentum states. In panel (d), the red Bose curve rises above "
            "the dashed blue classical curve at low energy.\n\n"
            "### Second experiment: move mu toward zero\n"
            "Keep **T = 120 nK** and compare **mu/k_B = -100 nK**, **-25 nK** and **-2 nK**. "
            "The k = 0 Bose fraction grows rapidly. This is the finite-box precursor of the "
            "macroscopic ground-state occupation associated with Bose--Einstein condensation.\n\n"
            "### Third experiment: change the box size\n"
            "Keep **T** and **mu** fixed, then compare **L = 10 um** with **L = 40 um**. "
            "A larger box has more closely spaced momentum states. You can see additional "
            "low-energy states around the centre."
        )
        streamlit.info(
            "Read one quantity at a time: first the bright centre, then the red/blue excess "
            "map, then the two curves. The changing pattern is the quantum-statistical effect."
        )

    with equations_tab:
        streamlit.latex(r"\hat H=\sum_{\mathbf{k}}\varepsilon_{\mathbf{k}}\,a_{\mathbf{k}}^\dagger a_{\mathbf{k}},\qquad \hat N=\sum_{\mathbf{k}}a_{\mathbf{k}}^\dagger a_{\mathbf{k}}")
        streamlit.latex(r"\hat K=\hat H-\mu\hat N=\sum_{\mathbf{k}}(\varepsilon_{\mathbf{k}}-\mu)a_{\mathbf{k}}^\dagger a_{\mathbf{k}}")
        streamlit.latex(r"\varepsilon_{\mathbf{k}}=\frac{\hbar^2|\mathbf{k}|^2}{2m},\qquad k_x=\frac{2\pi n_x}{L},\quad k_y=\frac{2\pi n_y}{L}")
        streamlit.latex(r"n_{\mathbf{k}}=\langle a_{\mathbf{k}}^\dagger a_{\mathbf{k}}\rangle=\frac{1}{\exp[\beta(\varepsilon_{\mathbf{k}}-\mu)]-1}")
        streamlit.markdown(
            "1. The program enumerates every visible integer pair \\(n_x,n_y\\).\n"
            "2. From it, it calculates momentum and kinetic energy.\n"
            "3. It inserts \\(T\\) and \\(\\mu\\) into the Bose-Einstein formula for every mode.\n"
            "4. Summing all \\(n_k\\) gives \\(\\langle N\\rangle\\); summing \\(\\varepsilon_k n_k\\) gives \\(\\langle H\\rangle\\).\n"
            "5. For a fair comparison, it builds a Maxwell--Boltzmann distribution with the "
            "same \\(\\langle N\\rangle\\): \\(n_k^{\\mathrm{cl}}\\propto e^{-\\beta\\varepsilon_k}\\)."
        )
        streamlit.markdown("### Three-dimensional BEC calculation")
        streamlit.latex(
            r"T_c=\frac{2\pi\hbar^2}{m k_B}\left(\frac{N/V}{\zeta(3/2)}\right)^{2/3}"
        )
        streamlit.latex(
            r"\frac{N_0}{N}=\begin{cases}1-(T/T_c)^{3/2}, & T<T_c,\\0, & T\geq T_c.\end{cases}"
        )

    with reading_tab:
        streamlit.markdown(
            "- **Panel (a):** Bose state fractions. The central square is the ground state \\(k=0\\).\n"
            "- **Panel (b):** a classical gas with exactly the same mean particle number. It is a reference, not a second simulation with different conditions.\n"
            "- **Panel (c):** red means ‘bosons have moved extra population here’; blue means ‘bosons have removed population from here’. This is bosonic bunching into low-energy states.\n"
            "- **Panel (d):** every point is a kinetic-energy shell; the plotted value is the occupation of one state in that shell. The red-versus-blue separation at low energy is the cleanest signature.\n"
            "- **Why no moving particles?** A free homogeneous ideal gas stays uniform in real space. Its informative physics is the occupation of momentum states. Traps, interactions and collisions will make real-space density dynamics meaningful later."
        )


def _render_quantum_gas_lab_inputs(
    streamlit: Any,
) -> tuple[IdealQuantumGasParameters, int, str]:
    """Render one common set of conditions for the three gas calculators."""
    streamlit.caption(
        "All three calculators use the same fixed N, T, L and mass. Only the statistical law changes."
    )
    temperature_nk = float(
        streamlit.number_input(
            "Temperature T (nK)",
            min_value=1.0,
            value=80.0,
            step=1.0,
            help="Absolute temperature of the thermal reservoir.",
            key="quantum-gas-temperature",
        )
    )
    particle_number = int(
        streamlit.number_input(
            "Total particle number N",
            min_value=100,
            max_value=10_000_000,
            value=100_000,
            step=1_000,
            help="The total N is fixed in every calculator; mu is solved from this condition.",
            key="quantum-gas-particle-number",
        )
    )
    box_length_um = float(
        streamlit.number_input(
            "Cubic-box side length L (um)",
            min_value=1.0,
            max_value=500.0,
            value=20.0,
            step=1.0,
            help="The volume is V=L^3; periodic boundary conditions are assumed.",
            key="quantum-gas-box-length",
        )
    )
    mass_amu = float(
        streamlit.number_input(
            "Particle mass (u)",
            min_value=0.1,
            max_value=1_000.0,
            value=87.0,
            step=1.0,
            help="87 u is approximately rubidium-87. The calculators are one-component gases.",
            key="quantum-gas-mass",
        )
    )
    maximum_mode_index = int(
        streamlit.number_input(
            "Visible momentum range |n_x|, |n_y|",
            min_value=3,
            max_value=30,
            value=12,
            step=1,
            help="Only the displayed n_z=0 slice. It does not truncate the thermodynamic calculation.",
            key="quantum-gas-visible-modes",
        )
    )
    display_scale = streamlit.selectbox(
        "Occupation colour scale",
        options=("Automatic contrast", "Locked absolute scale"),
        help="Automatic contrast makes one calculation readable. Use the locked scale when comparing several parameter settings.",
        key="quantum-gas-colour-scale",
    )
    return (
        IdealQuantumGasParameters(
            temperature=temperature_nk * 1.0e-9,
            particle_number=particle_number,
            box_length=box_length_um * 1.0e-6,
            particle_mass_amu=mass_amu,
        ),
        maximum_mode_index,
        "auto" if display_scale == "Automatic contrast" else "fixed",
    )


def _gas_common_metrics(streamlit: Any, result: QuantumGasResult) -> None:
    """Render quantities that have identical meanings in all gas tabs."""
    metrics = streamlit.columns(5)
    metrics[0].metric("Density n", f"{result.number_density:.3e} m^-3")
    metrics[1].metric("Thermal wavelength lambda_T", f"{result.thermal_wavelength * 1.0e6:.3f} um")
    metrics[2].metric("Phase-space density n lambda_T^3", f"{result.phase_space_density:.3g}")
    metrics[3].metric("Chemical potential mu/k_B", f"{result.chemical_potential / BOLTZMANN_CONSTANT / 1.0e-9:.3f} nK")
    metrics[4].metric(
        "Mean energy per particle",
        f"{result.mean_energy_per_particle / BOLTZMANN_CONSTANT / 1.0e-9:.3f} nK",
    )


def _show_classical_gas_calculator(
    streamlit: Any,
    result: QuantumGasResult,
    maximum_mode_index: int,
    colour_scale_mode: str,
) -> None:
    """Render the separate Maxwell--Boltzmann calculator and its validity check."""
    streamlit.subheader("Classical ideal gas — Maxwell--Boltzmann statistics")
    streamlit.caption(
        "Particles are distinguishable for the purpose of counting states; quantum exchange effects are neglected."
    )
    _gas_common_metrics(streamlit, result)
    if result.classical_regime:
        streamlit.success(
            "Classical approximation is self-consistent: n lambda_T^3 < 0.1, so quantum-statistical effects are weak."
        )
    else:
        streamlit.warning(
            "Classical approximation is not reliable here: n lambda_T^3 is not much smaller than one. Compare the Bose-gas tab."
        )
    streamlit.plotly_chart(
        quantum_gas_state_figure(
            result,
            maximum_mode_index=maximum_mode_index,
            scale_mode=colour_scale_mode,
        ),
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
    )
    with streamlit.expander("What this calculator means", expanded=True):
        streamlit.latex(r"n_{\mathbf{k}}=\exp[-\beta(\varepsilon_{\mathbf{k}}-\mu)]")
        streamlit.markdown(
            "**Sketch:** thermal energy spreads the population smoothly over many momentum states. "
            "There is no upper occupation limit and no special enhancement of an already occupied state.\n\n"
            "The map is an n_z = 0 slice through three-dimensional momentum space. The colour is "
            "the mean occupation of one quantum state, on the same fixed scale used in every tab."
        )


def _show_bose_gas_calculator(
    streamlit: Any,
    result: QuantumGasResult,
    maximum_mode_index: int,
    colour_scale_mode: str,
) -> None:
    """Render Bose--Einstein statistics together with the 3D BEC transition."""
    streamlit.subheader("Ideal Bose gas — Bose--Einstein statistics and condensation")
    streamlit.caption(
        "One-component bosons may share a one-particle state without an upper occupation limit."
    )
    _gas_common_metrics(streamlit, result)
    if result.critical_temperature is not None:
        bose_metrics = streamlit.columns(3)
        bose_metrics[0].metric("Critical temperature T_c", f"{result.critical_temperature / 1.0e-9:.3f} nK")
        bose_metrics[1].metric("T / T_c", f"{result.parameters.temperature / result.critical_temperature:.3f}")
        bose_metrics[2].metric("Condensate fraction N0 / N", f"{result.condensate_fraction:.2%}")
    streamlit.plotly_chart(
        quantum_gas_state_figure(
            result,
            maximum_mode_index=maximum_mode_index,
            scale_mode=colour_scale_mode,
        ),
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
    )
    streamlit.plotly_chart(
        bose_condensation_figure(result),
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
    )
    with streamlit.expander("What this calculator means", expanded=True):
        streamlit.latex(r"n_{\mathbf{k}}=\frac{1}{\exp[\beta(\varepsilon_{\mathbf{k}}-\mu)]-1}")
        streamlit.latex(r"\frac{N_0}{N}=1-\left(\frac{T}{T_c}\right)^{3/2}\quad (T<T_c)")
        streamlit.markdown(
            "**Sketch:** bosons can pile into the same low-energy state. Above T_c, the thermal "
            "cloud contains all particles. Below T_c, the excited states are saturated and the "
            "remaining particles form a macroscopic k = 0 condensate.\n\n"
            "The centre of panel (a) is the k = 0 state. If there is a condensate, it contains N0 "
            "and becomes bright on the fixed occupation scale."
        )


def _show_quantum_gas_lab(
    streamlit: Any,
    parameters: IdealQuantumGasParameters,
    maximum_mode_index: int,
    colour_scale_mode: str,
) -> None:
    """Render three separated, rigorously comparable quantum-gas calculators."""
    streamlit.title("Quantum-gas equilibrium laboratory")
    streamlit.caption(
        "Two separate calculations under identical conditions. The chemical potential is solved "
        "for each statistical law so that both calculators have the same total particle number N."
    )
    streamlit.info(
        f"All momentum maps use the fixed scale log10(mean occupation) from "
        f"{MODE_OCCUPATION_LOG_RANGE[0]:.0f} to {MODE_OCCUPATION_LOG_RANGE[1]:.0f}. "
        "A colour therefore keeps the same physical meaning when you change T, N, L or mass."
    )
    classical_result = analyze_ideal_quantum_gas(parameters, "classical")
    bose_result = analyze_ideal_quantum_gas(parameters, "boson")
    classical_tab, bose_tab, guide_tab = streamlit.tabs(
        ("Classical gas", "Bose gas + BEC", "How to experiment")
    )
    with classical_tab:
        _show_classical_gas_calculator(
            streamlit,
            classical_result,
            maximum_mode_index,
            colour_scale_mode,
        )
    with bose_tab:
        _show_bose_gas_calculator(
            streamlit,
            bose_result,
            maximum_mode_index,
            colour_scale_mode,
        )
    with guide_tab:
        streamlit.subheader("The three counting rules")
        streamlit.caption("Schematic only: it explains the rules; the other tabs contain the calculated data.")
        streamlit.plotly_chart(
            quantum_statistics_schematic(),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
        streamlit.markdown(
            "### A reliable learning sequence\n"
            "1. Start with **low density**: choose a large L, small N, or high T. "
            "Then n lambda_T^3 is much smaller than 1, and Bose statistics approaches classical behaviour.\n"
            "2. Increase density by raising N, reducing L, or cooling. Quantum statistics becomes visible.\n"
            "3. In the Bose tab, cool below T/T_c = 1: the condensate fraction and the central k = 0 mode grow.\n\n"
            "### Scope and assumptions\n"
            "This is a non-interacting, homogeneous, three-dimensional, one-component gas in the "
            "thermodynamic limit. It correctly describes equilibrium quantum statistics. It does not "
            "yet include collisions, traps, scattering length, superfluid dynamics or finite-size corrections."
        )


def _render_ensemble_lab_inputs(
    streamlit: Any,
) -> tuple[str, str, Any, dict[str, float | int]]:
    """Select an ensemble and create only its physically independent inputs."""
    gas_kind = streamlit.selectbox(
        "Gas",
        options=("Classical ideal gas", "Ideal Bose gas"),
        key="ensemble-lab-gas-kind",
    )
    ensemble = streamlit.radio(
        "Ensemble",
        options=("Canonical", "Grand canonical", "Microcanonical"),
        key="ensemble-lab-ensemble",
    )
    streamlit.caption(
        "Only independent variables appear below. Derived quantities such as temperature or chemical potential are calculated by the program."
    )

    box_length_um = float(
        streamlit.number_input(
            "Box side length L (um)",
            min_value=1.0,
            max_value=500.0,
            value=20.0,
            step=1.0,
            key="ensemble-lab-box-length",
        )
    )
    mass_amu = float(
        streamlit.number_input(
            "Particle mass (u)",
            min_value=0.1,
            max_value=1_000.0,
            value=87.0,
            step=1.0,
            key="ensemble-lab-mass",
        )
    )
    box_length = box_length_um * 1.0e-6
    settings: dict[str, float | int] = {}

    if gas_kind == "Classical ideal gas":
        if ensemble == "Canonical":
            temperature_nk = float(
                streamlit.number_input(
                    "Temperature T (nK)", min_value=1.0, value=80.0, step=1.0,
                    key="classical-canonical-temperature",
                )
            )
            particle_number = int(
                streamlit.number_input(
                    "Fixed particle number N", min_value=2, max_value=100_000, value=200, step=10,
                    key="classical-canonical-particles",
                )
            )
            parameters: Any = ClassicalCanonicalParameters(
                temperature=temperature_nk * 1.0e-9,
                particle_number=particle_number,
                box_length=box_length,
                particle_mass_amu=mass_amu,
            )
        elif ensemble == "Grand canonical":
            temperature_nk = float(
                streamlit.number_input(
                    "Temperature T (nK)", min_value=1.0, value=80.0, step=1.0,
                    key="classical-grand-temperature",
                )
            )
            chemical_potential_nk = float(
                streamlit.number_input(
                    "Chemical potential mu/k_B (nK)", min_value=-1_000.0, max_value=1_000.0,
                    value=-300.0, step=10.0, key="classical-grand-mu",
                )
            )
            parameters = ClassicalGrandCanonicalParameters(
                temperature=temperature_nk * 1.0e-9,
                chemical_potential=chemical_potential_nk * 1.0e-9 * BOLTZMANN_CONSTANT,
                box_length=box_length,
                particle_mass_amu=mass_amu,
            )
        else:
            particle_number = int(
                streamlit.number_input(
                    "Fixed particle number N", min_value=2, max_value=100_000, value=200, step=10,
                    key="classical-micro-particles",
                )
            )
            energy_per_particle_nk = float(
                streamlit.number_input(
                    "Fixed energy per particle E/(N k_B) (nK)", min_value=0.1,
                    value=120.0, step=1.0, key="classical-micro-energy",
                )
            )
            parameters = ClassicalMicrocanonicalParameters(
                particle_number=particle_number,
                total_energy=energy_per_particle_nk * particle_number * 1.0e-9 * BOLTZMANN_CONSTANT,
                box_length=box_length,
                particle_mass_amu=mass_amu,
            )
        settings["visible_particle_count"] = int(
            streamlit.number_input(
                "Visible representative particles", min_value=20, max_value=500, value=160, step=10,
                key="classical-visible-particles",
            )
        )
        settings["duration_ms"] = float(
            streamlit.number_input(
                "Displayed motion time (ms)", min_value=0.1, max_value=50.0, value=5.0, step=0.5,
                key="classical-motion-duration",
            )
        )
        return gas_kind, ensemble, parameters, settings

    if ensemble == "Canonical":
        temperature_nk = float(
            streamlit.number_input(
                "Temperature T (nK)", min_value=1.0, value=80.0, step=1.0,
                key="bose-canonical-temperature",
            )
        )
        particle_number = int(
            streamlit.number_input(
                "Fixed particle number N", min_value=100, max_value=10_000_000, value=100_000, step=1_000,
                key="bose-canonical-particles",
            )
        )
        parameters = IdealQuantumGasParameters(
            temperature=temperature_nk * 1.0e-9,
            particle_number=particle_number,
            box_length=box_length,
            particle_mass_amu=mass_amu,
        )
        settings["visible_modes"] = int(
            streamlit.number_input(
                "Visible momentum range |n_x|, |n_y|", min_value=3, max_value=30, value=12, step=1,
                key="bose-canonical-visible-modes",
            )
        )
    elif ensemble == "Grand canonical":
        temperature_nk = float(
            streamlit.number_input(
                "Temperature T (nK)", min_value=1.0, value=80.0, step=1.0,
                key="bose-grand-temperature",
            )
        )
        chemical_potential_nk = float(
            streamlit.number_input(
                "Chemical potential mu/k_B (nK)", min_value=-1_000.0, max_value=-0.001,
                value=-15.0, step=1.0, key="bose-grand-mu",
                help="Must be below zero. Moving it toward zero increases both occupation and particle-number fluctuations.",
            )
        )
        parameters = BoseGrandCanonicalParameters(
            temperature=temperature_nk * 1.0e-9,
            chemical_potential=chemical_potential_nk * 1.0e-9 * BOLTZMANN_CONSTANT,
            box_length=box_length,
            particle_mass_amu=mass_amu,
        )
        settings["visible_modes"] = int(
            streamlit.number_input(
                "Visible momentum range |n_x|, |n_y|", min_value=3, max_value=30, value=12, step=1,
                key="bose-grand-visible-modes",
            )
        )
    else:
        particle_number = int(
            streamlit.number_input(
                "Fixed particle number N", min_value=1, max_value=14, value=8, step=1,
                key="bose-micro-particles",
            )
        )
        energy_quanta = int(
            streamlit.number_input(
                "Fixed total energy E / epsilon_1", min_value=0, max_value=40, value=8, step=1,
                key="bose-micro-energy",
                help="epsilon_1 is the energy of the n = plus or minus 1 box mode.",
            )
        )
        maximum_mode_index = int(
            streamlit.number_input(
                "Included modes |n| <=", min_value=1, max_value=6, value=4, step=1,
                key="bose-micro-cutoff",
            )
        )
        parameters = BoseMicrocanonicalParameters(
            particle_number=particle_number,
            energy_quanta=energy_quanta,
            maximum_mode_index=maximum_mode_index,
            box_length=box_length,
            particle_mass_amu=mass_amu,
        )
    if ensemble != "Microcanonical":
        display_scale = streamlit.selectbox(
            "Occupation colour scale",
            options=("Automatic contrast", "Locked absolute scale"),
            help="Automatic contrast is best for understanding one state. Locked scale preserves identical colours while comparing parameter settings.",
            key=f"bose-{ensemble.lower().replace(' ', '-')}-colour-scale",
        )
        settings["colour_scale_mode"] = (
            "auto" if display_scale == "Automatic contrast" else "fixed"
        )
    return gas_kind, ensemble, parameters, settings


def _classical_ensemble_table(result: ClassicalEnsembleResult) -> list[dict[str, str]]:
    """Create a compact fixed-versus-fluctuating dashboard for a classical ensemble."""
    return [
        {
            "Quantity": "Particle number N",
            "Status": "fixed" if result.particle_number_fixed is not None else "fluctuates",
            "Mean": f"{result.particle_number_mean:.3g}",
            "Variance": f"{result.number_variance:.3g}",
        },
        {
            "Quantity": "Total energy E",
            "Status": "fixed" if result.total_energy_fixed is not None else "fluctuates",
            "Mean / k_B": f"{result.total_energy_mean / BOLTZMANN_CONSTANT / 1.0e-9:.3g} nK",
            "Variance": f"{result.energy_variance / (BOLTZMANN_CONSTANT * 1.0e-9) ** 2:.3g} nK^2",
        },
        {
            "Quantity": "Temperature T",
            "Status": "derived" if result.ensemble == "microcanonical" else "fixed",
            "Mean": f"{result.temperature / 1.0e-9:.3g} nK",
            "Variance": "not an independent variable",
        },
    ]


def _show_classical_ensemble_lab(
    streamlit: Any,
    ensemble: str,
    parameters: Any,
    settings: dict[str, float | int],
) -> None:
    """Render classical ensemble thermodynamics and the physically valid animation."""
    if ensemble == "Canonical":
        result = analyze_classical_canonical(parameters)
    elif ensemble == "Grand canonical":
        result = analyze_classical_grand_canonical(parameters)
    else:
        result = analyze_classical_microcanonical(parameters)
    sample = sample_classical_particles(
        result,
        visible_particle_count=int(settings["visible_particle_count"]),
    )
    streamlit.subheader(f"Classical ideal gas - {ensemble.lower()} ensemble")
    streamlit.caption(
        "The moving points are a representative x-y projection. In an ideal gas they move ballistically and wrap through periodic boundaries; collisions are intentionally absent."
    )
    metrics = streamlit.columns(5)
    metrics[0].metric("Mean particle number", f"{result.particle_number_mean:.4g}")
    metrics[1].metric("Temperature", f"{result.temperature / 1.0e-9:.3f} nK")
    metrics[2].metric("Mean total energy / k_B", f"{result.total_energy_mean / BOLTZMANN_CONSTANT / 1.0e-9:.3g} nK")
    metrics[3].metric("mu / k_B", f"{result.chemical_potential / BOLTZMANN_CONSTANT / 1.0e-9:.3f} nK")
    metrics[4].metric("n lambda_T^3", f"{result.phase_space_density:.3g}")
    streamlit.dataframe(_classical_ensemble_table(result), hide_index=True, use_container_width=True)
    streamlit.plotly_chart(
        classical_ensemble_motion_figure(
            result,
            sample,
            duration_ms=float(settings["duration_ms"]),
        ),
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
    )
    streamlit.latex(r"H=\sum_{i=1}^{N}\frac{p_i^2}{2m}")
    if ensemble == "Canonical":
        streamlit.markdown(
            "**Canonical:** T, N and V are fixed. Energy may fluctuate between thermal configurations; the red curve is the Maxwell speed law expected at the imposed temperature."
        )
    elif ensemble == "Grand canonical":
        streamlit.markdown(
            "**Grand canonical:** T, mu and V are fixed. Both N and E fluctuate. For a classical ideal gas the number distribution is Poisson, so Var(N) = mean(N)."
        )
    else:
        streamlit.markdown(
            "**Microcanonical:** E, N and V are fixed exactly. The displayed temperature is inferred from how the number of allowed states changes with energy; it is not an input."
        )


def _show_bose_ensemble_lab(
    streamlit: Any,
    ensemble: str,
    parameters: Any,
    settings: dict[str, float | int],
) -> None:
    """Render Bose ensembles without replacing quantum states by fake trajectories."""
    streamlit.subheader(f"Ideal Bose gas - {ensemble.lower()} ensemble")
    if ensemble == "Canonical":
        result = analyze_ideal_quantum_gas(parameters, "boson")
        metrics = streamlit.columns(5)
        metrics[0].metric("Fixed N", f"{parameters.particle_number:,}")
        metrics[1].metric("T / T_c", f"{parameters.temperature / result.critical_temperature:.3f}")
        metrics[2].metric("Condensate fraction N0 / N", f"{result.condensate_fraction:.2%}")
        metrics[3].metric("mu / k_B", f"{result.chemical_potential / BOLTZMANN_CONSTANT / 1.0e-9:.3f} nK")
        metrics[4].metric("Entropy S / k_B", f"{result.entropy_over_kb:.4g}")
        streamlit.plotly_chart(
            quantum_gas_state_figure(
                result,
                maximum_mode_index=int(settings["visible_modes"]),
                scale_mode=str(settings["colour_scale_mode"]),
            ),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
        streamlit.plotly_chart(
            bose_condensation_figure(result),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
        streamlit.markdown(
            "**Canonical Bose gas:** T, N and V are fixed. Below T_c, the thermal cloud is saturated and the remaining fixed particles occupy the k = 0 condensate."
        )
    elif ensemble == "Grand canonical":
        result = analyze_bose_grand_canonical(parameters)
        metrics = streamlit.columns(5)
        metrics[0].metric("Mean particle number <N>", f"{result.mean_particle_number:.4g}")
        metrics[1].metric("Number variance", f"{result.number_variance:.4g}")
        metrics[2].metric("Ground-state occupation n0", f"{result.ground_state_occupation:.4g}")
        metrics[3].metric("mu / k_B", f"{parameters.chemical_potential / BOLTZMANN_CONSTANT / 1.0e-9:.3f} nK")
        metrics[4].metric("Entropy S / k_B", f"{result.entropy_over_kb:.4g}")
        if result.mean_particle_number < 1.0:
            streamlit.warning(
                "This setting is a near-vacuum regime: the mean particle number is below one. The faint momentum plot is therefore expected; it is not a dense Bose gas or a condensate. Use the canonical ensemble to study BEC, or move mu closer to zero to study reservoir-driven filling."
            )
        streamlit.plotly_chart(
            bose_grand_canonical_figure(
                result,
                maximum_mode_index=int(settings["visible_modes"]),
                scale_mode=str(settings["colour_scale_mode"]),
            ),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
        streamlit.latex(r"\langle N\rangle=\frac{V}{\lambda_T^3}g_{3/2}(e^{\beta\mu}),\qquad \mathrm{Var}(N)=\frac{V}{\lambda_T^3}g_{1/2}(e^{\beta\mu})")
        streamlit.markdown(
            "**Grand-canonical Bose gas:** T, mu and V are fixed. Move mu toward zero to increase the low-energy occupation and particle-number fluctuations. The exactly singular point mu = 0 is excluded here because a reservoir alone does not determine a finite condensate population."
        )
    else:
        result = analyze_bose_microcanonical(parameters)
        metrics = streamlit.columns(5)
        metrics[0].metric("Fixed N", str(parameters.particle_number))
        metrics[1].metric("Fixed E / epsilon_1", str(parameters.energy_quanta))
        metrics[2].metric("Exact Fock microstates", f"{result.microstate_count:,}")
        metrics[3].metric("Entropy S / k_B", f"{result.entropy_over_kb:.4g}")
        metrics[4].metric(
            "Inferred temperature",
            "not resolved" if result.inferred_temperature is None else f"{result.inferred_temperature / 1.0e-9:.3g} nK",
        )
        streamlit.plotly_chart(
            bose_microcanonical_figure(result),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
        streamlit.latex(r"\sum_n n_n=N,\qquad \sum_n n^2 n_n=E/\varepsilon_1")
        streamlit.markdown(
            "**Microcanonical Bose gas:** this is an exact, small finite one-dimensional mode calculation. The program enumerates every Fock state consistent with the fixed N and E, then averages occupations uniformly over those states. A stationary equilibrium state does not provide classical particle trajectories; the physically meaningful output is its quantum-state occupation."
        )


def _show_ensemble_dynamics_lab(
    streamlit: Any,
    gas_kind: str,
    ensemble: str,
    parameters: Any,
    settings: dict[str, float | int],
) -> None:
    """Render the ensemble laboratory with a common, explicit physical scope."""
    streamlit.title("Ensembles and dynamics laboratory")
    streamlit.caption(
        "Choose the ensemble first, then observe which variables are fixed and which fluctuate. The laboratory never treats ensemble labels as cosmetic settings."
    )
    if gas_kind == "Classical ideal gas":
        _show_classical_ensemble_lab(streamlit, ensemble, parameters, settings)
    else:
        try:
            _show_bose_ensemble_lab(streamlit, ensemble, parameters, settings)
        except ValueError as error:
            streamlit.error(f"This finite microcanonical system could not be enumerated: {error}")


def _create_parameter_scan_study(
    model_key: str,
    serialized_parameters: str,
    parameter_name: str,
    start: float,
    stop: float,
    n_points: int,
) -> StudyData:
    """Berechne einen Scan und verpacke Resultate und Observablen als Studie."""
    if start >= stop:
        raise ValueError("Scan start must be smaller than scan stop")

    specification = MODEL_REGISTRY.get(model_key)
    parameter_values = json.loads(serialized_parameters)
    scan_values = np.linspace(start, stop, n_points)
    scan = model_parameter_scan(
        parameter_name,
        scan_values,
        parameter_model=specification.parameter_model,
        model_factory=specification.factory,
        base_parameters=parameter_values,
        metadata={
            "scan_start": start,
            "scan_stop": stop,
            "n_points": n_points,
        },
    )
    analysis = analyze_parameter_scan(scan)
    reference_parameters = dict(scan.results[0].parameters)
    reference_parameters.pop(parameter_name, None)
    random_seed = reference_parameters.get("disorder_seed")

    metadata = StudyMetadata(
        study_name=f"{specification.display_name}: {parameter_name} scan",
        model_name=scan.results[0].model_name,
        model_parameters=reference_parameters,
        random_seed=random_seed if isinstance(random_seed, int) else None,
        description="Interactive 1D parameter scan created in TopOSC Lab.",
        tags=("parameter-scan", model_key),
    )

    return study_from_parameter_scan(
        scan,
        metadata,
        observables=analysis.observable_arrays(),
    )


def _show_empty_scan_workspace(streamlit: Any) -> None:
    """Erklaere, welche Resultate der Scan-Workspace erzeugt."""
    streamlit.info("Configure a parameter range in the sidebar, then run the scan.")
    streamlit.markdown(
        """
        Every point is a complete validated simulation. The resulting study
        contains spectra, minimum energy, bulk-gap estimate, zero-mode count,
        IPR and edge weight, together with the fixed model parameters.
        """
    )


def _show_scan_workspace(
    streamlit: Any,
    study: StudyData,
    *,
    run_id: int,
) -> None:
    """Zeige Spektrum, Observablen, Metadaten und Download eines Scans."""
    streamlit.subheader(study.metadata.study_name)
    spectrum_tab, observables_tab, metadata_tab = streamlit.tabs(
        ("Spectrum and gap", "Observables", "Metadata and export")
    )

    with spectrum_tab:
        with paper_style():
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            plot_study_spectrum(
                study,
                axes=axes[0],
                title="Spectrum",
                show=False,
            )
            plot_study_observable(
                study,
                "bulk_gaps",
                axes=axes[1],
                title="Bulk-gap estimate",
                ylabel="Bulk gap",
                show=False,
            )
            add_panel_label(axes[0], "(a)")
            add_panel_label(axes[1], "(b)")
            streamlit.pyplot(figure)
            _add_figure_downloads(
                streamlit,
                figure,
                filename_stem="toposc_parameter_scan_overview",
                key=f"scan-overview::{run_id}",
            )
            plt.close(figure)

    with observables_tab:
        observable_names = (
            "minimum_abs_energy",
            "bulk_gaps",
            "zero_mode_counts",
            "inverse_participation_ratios",
            "edge_weights",
        )
        available_observables = [
            name for name in observable_names if name in study.arrays
        ]
        selected_observable = streamlit.selectbox(
            "Observable",
            options=available_observables,
            key=f"scan-observable::{run_id}",
        )
        figure, _ = plot_study_observable(
            study,
            selected_observable,
            title=selected_observable.replace("_", " ").capitalize(),
            drawstyle=(
                "steps-mid"
                if selected_observable.endswith("_counts")
                or "chern" in selected_observable
                else "default"
            ),
            show=False,
        )
        streamlit.pyplot(figure)
        _add_figure_downloads(
            streamlit,
            figure,
            filename_stem=f"toposc_{selected_observable}",
            key=f"scan-observable::{run_id}::{selected_observable}",
        )
        plt.close(figure)

        streamlit.dataframe(
            {
                study.metadata.scan_parameters["parameter_name"]: study.arrays[
                    "parameter_values"
                ],
                selected_observable: study.arrays[selected_observable],
            },
            use_container_width=True,
        )
        streamlit.caption(
            "Edge weight is physically meaningful for open boundaries; for "
            "periodic boundaries it remains only a grid-based diagnostic."
        )

    with metadata_tab:
        streamlit.json(study.metadata.model_dump(mode="json"))
        streamlit.download_button(
            "Download reproducible study (.npz)",
            data=_study_archive(study),
            file_name="toposc_parameter_scan.npz",
            mime="application/octet-stream",
        )


def _next_study_label(
    entries: dict[str, dict[str, Any]],
    suggested_label: str,
) -> str:
    """Create a readable unique label for a newly loaded study."""
    occupied = {str(entry["label"]) for entry in entries.values()}
    base_label = suggested_label or "unnamed study"

    if base_label not in occupied:
        return base_label

    index = 2
    while f"{base_label} ({index})" in occupied:
        index += 1
    return f"{base_label} ({index})"


def _show_study_explorer(streamlit: Any) -> None:
    """Load, inspect and compare complete reproducible study archives."""
    streamlit.subheader("Study explorer")
    streamlit.caption(
        "Load one or several TopOSC-Lab .npz studies. Comparisons are enabled "
        "only when their scan axis and observable are compatible."
    )

    entries = streamlit.session_state.setdefault("toposc_loaded_studies", {})
    uploads = streamlit.file_uploader(
        "Add saved studies (.npz)",
        type=("npz",),
        accept_multiple_files=True,
        help="Files produced by 'Download reproducible study (.npz)'.",
    )

    for upload in uploads or ():
        data = upload.getvalue()
        digest = sha256(data).hexdigest()

        if digest in entries:
            continue

        try:
            study = _study_from_uploaded_bytes(data)
        except (TypeError, ValueError) as error:
            streamlit.error(f"{upload.name}: {error}")
            continue

        entries[digest] = {
            "label": _next_study_label(entries, Path(upload.name).stem),
            "study": study,
        }

    latest_scan = streamlit.session_state.get("toposc_parameter_scan")
    if latest_scan is not None and streamlit.button("Add latest parameter scan"):
        study = latest_scan["study"]
        digest = f"workspace::{id(study)}"
        if digest not in entries:
            entries[digest] = {
                "label": _next_study_label(entries, "latest workspace scan"),
                "study": study,
            }

    if not entries:
        streamlit.info(
            "Create a parameter scan or upload a saved .npz study to start "
            "comparing reproducible results."
        )
        return

    labels = [str(entry["label"]) for entry in entries.values()]
    labels_to_entries = {
        str(entry["label"]): entry for entry in entries.values()
    }
    selected_labels = streamlit.multiselect(
        "Studies to inspect",
        options=labels,
        default=labels,
    )
    selected_studies = {
        label: labels_to_entries[label]["study"] for label in selected_labels
    }

    remove_labels = streamlit.multiselect(
        "Remove loaded studies",
        options=labels,
        key="remove-loaded-studies",
    )
    if remove_labels and streamlit.button("Remove selected studies"):
        for digest, entry in tuple(entries.items()):
            if entry["label"] in remove_labels:
                del entries[digest]
        streamlit.rerun()

    if not selected_studies:
        streamlit.info("Select at least one study to inspect it.")
        return

    streamlit.dataframe(
        [study_summary(study) for study in selected_studies.values()],
        hide_index=True,
        use_container_width=True,
    )

    comparison_tab, spectrum_tab, metadata_tab = streamlit.tabs(
        ("Compare observables", "Inspect spectrum", "Metadata")
    )

    with comparison_tab:
        try:
            parameter_name = scan_parameter_name(selected_studies)
            observable_names = common_scalar_observables(selected_studies)
        except ValueError as error:
            streamlit.warning(f"These studies cannot be compared: {error}")
            observable_names = ()
        else:
            streamlit.caption(f"Shared scan axis: {parameter_name}")

        if not observable_names:
            streamlit.info("The selected studies do not share a scalar observable.")
        else:
            observable_name = streamlit.selectbox(
                "Shared observable",
                options=observable_names,
                format_func=_parameter_label,
            )
            figure, _ = plot_study_comparison(
                selected_studies,
                observable_name,
                title=_parameter_label(observable_name),
                ylabel=_parameter_label(observable_name),
                show=False,
            )
            streamlit.pyplot(figure)
            _add_figure_downloads(
                streamlit,
                figure,
                filename_stem=f"toposc_comparison_{observable_name}",
                key=f"study-comparison::{observable_name}::{','.join(selected_labels)}",
            )
            plt.close(figure)

    with spectrum_tab:
        spectrum_labels = [
            label
            for label, study in selected_studies.items()
            if "spectra" in study.arrays
        ]
        if not spectrum_labels:
            streamlit.info("None of the selected studies contains a saved spectrum.")
        else:
            spectrum_label = streamlit.selectbox(
                "Study for spectrum inspection",
                options=spectrum_labels,
            )
            figure, _ = plot_study_spectrum(
                selected_studies[spectrum_label],
                title=f"{spectrum_label}: spectrum",
                show=False,
            )
            streamlit.pyplot(figure)
            _add_figure_downloads(
                streamlit,
                figure,
                filename_stem="toposc_study_spectrum",
                key=f"study-spectrum::{spectrum_label}",
            )
            plt.close(figure)

    with metadata_tab:
        selected_label = streamlit.selectbox(
            "Study metadata",
            options=tuple(selected_studies),
        )
        streamlit.json(
            selected_studies[selected_label].metadata.model_dump(mode="json")
        )


def _show_empty_workspace(streamlit: Any) -> None:
    """Describe the available simulation workspace before its first run."""
    streamlit.info("Choose a model, adjust its parameters, then run a simulation.")
    streamlit.markdown(
        """
        **The workspace supports:**

        - every currently implemented model and all of its parameters,
        - spectra, geometry-aware wavefunction plots and core observables,
        - parameter validation, reproducible disorder seeds and result download.

        Use *Parameter scan* for reproducible 1D studies and *Study explorer*
        to load, compare and export saved results.  Phase diagrams and disorder
        ensembles are the next research-workspace modules.
        """
    )


def run_app() -> None:
    """Starte die Streamlit-Research-Workspace-Seite."""
    streamlit = _streamlit_module()
    streamlit.set_page_config(
        page_title="TopOSC Lab",
        page_icon="T",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    streamlit.markdown(
        """
        <style>
        .stApp { background: #fbfcfe; }
        [data-testid="stSidebar"] {
            background: #f4f7fb;
            border-right: 1px solid #dce4ef;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dce4ef;
            border-radius: 10px;
            padding: 0.75rem;
        }
        h1, h2, h3 { color: #172554; }
        [data-testid="stTabs"] button { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    @streamlit.cache_data(show_spinner=False)
    def cached_simulation(model_key: str, serialized_parameters: str) -> Any:
        return _solve_serialized_model(model_key, serialized_parameters)

    @streamlit.cache_data(show_spinner=False)
    def cached_parameter_scan(
        model_key: str,
        serialized_parameters: str,
        parameter_name: str,
        start: float,
        stop: float,
        n_points: int,
    ) -> StudyData:
        return _create_parameter_scan_study(
            model_key,
            serialized_parameters,
            parameter_name,
            start,
            stop,
            n_points,
        )

    specifications = MODEL_REGISTRY.specifications()
    specification_by_key = {specification.key: specification for specification in specifications}
    run_requested = False
    scan_requested = False
    scan_parameter_name = ""
    scan_start = 0.0
    scan_stop = 0.0
    scan_points = 0
    parameter_values: dict[str, Any] = {}

    with streamlit.sidebar:
        streamlit.title("TopOSC Lab")
        project_area = streamlit.radio(
            "Research area",
            options=(
                "Topological superconductors",
                "Quantum gases",
                "Research studies",
            ),
        )
        streamlit.divider()
        if project_area == "Research studies":
            workspace_mode = "Study explorer"
            streamlit.header("Study explorer")
            streamlit.caption("Load, compare and export reproducible studies.")
        elif project_area == "Quantum gases":
            workspace_mode = streamlit.radio(
                "Quantum-gas workspace",
                options=("Equilibrium statistics", "Ensembles and dynamics"),
            )
            if workspace_mode == "Equilibrium statistics":
                streamlit.header("Common gas conditions")
                (
                    quantum_gas_parameters,
                    quantum_gas_visible_modes,
                    quantum_gas_colour_scale_mode,
                ) = (
                    _render_quantum_gas_lab_inputs(streamlit)
                )
            else:
                streamlit.header("Ensemble setup")
                (
                    ensemble_gas_kind,
                    ensemble_kind,
                    ensemble_parameters,
                    ensemble_settings,
                ) = _render_ensemble_lab_inputs(streamlit)
        else:
            workspace_mode = streamlit.radio(
                "Topological workspace",
                options=("Single simulation", "Parameter scan", "Model guide"),
            )
            streamlit.header("Simulation setup")
            selected_key = streamlit.selectbox(
                "Model",
                options=tuple(specification_by_key),
                format_func=lambda key: specification_by_key[key].display_name,
            )
            specification = specification_by_key[selected_key]
            streamlit.caption(specification.category)
            streamlit.write(specification.description)

            if workspace_mode == "Single simulation":
                with streamlit.form(f"parameters::{selected_key}"):
                    parameter_values = _render_parameter_inputs(streamlit, specification)
                    run_requested = streamlit.form_submit_button("Run simulation")
            elif workspace_mode == "Model guide":
                with streamlit.form(f"guide::{selected_key}"):
                    parameter_values = _render_parameter_inputs(streamlit, specification)
                    streamlit.form_submit_button("Update model guide")
            else:
                scan_options = _scannable_parameter_names(specification)
                scan_parameter_name = streamlit.selectbox(
                    "Parameter to scan",
                    options=scan_options,
                    format_func=_parameter_label,
                )
                default_start, default_stop, default_points = _scan_default(
                    specification,
                    scan_parameter_name,
                )

                with streamlit.form(f"scan::{selected_key}::{scan_parameter_name}"):
                    parameter_values = _render_parameter_inputs(streamlit, specification)
                    streamlit.caption(
                        "The selected scan parameter above is replaced by the scan axis."
                    )
                    scan_start = float(
                        streamlit.number_input(
                            "Scan start",
                            value=default_start,
                            key=f"scan-start::{selected_key}::{scan_parameter_name}",
                        )
                    )
                    scan_stop = float(
                        streamlit.number_input(
                            "Scan stop",
                            value=default_stop,
                            key=f"scan-stop::{selected_key}::{scan_parameter_name}",
                        )
                    )
                    scan_points = int(
                        streamlit.number_input(
                            "Number of scan points",
                            min_value=2,
                            max_value=201,
                            value=min(default_points, 61),
                            step=1,
                            key=f"scan-points::{selected_key}::{scan_parameter_name}",
                        )
                    )
                    scan_requested = streamlit.form_submit_button("Run parameter scan")

                try:
                    preview_parameters = {
                        **parameter_values,
                        scan_parameter_name: scan_start,
                    }
                    dimension = specification.build(preview_parameters).basis_layout.dimension
                    estimated_work = scan_points * dimension**3
                    streamlit.caption(
                        f"Matrix dimension: {dimension}; {scan_points} exact diagonalizations."
                    )
                    if estimated_work > 1_000_000_000:
                        streamlit.warning(
                            "This scan may take a long time. Consider fewer scan points "
                            "or a smaller lattice."
                        )
                except (TypeError, ValueError):
                    pass

            if (
                parameter_values.get("disorder_strength", 0.0) > 0.0
                and parameter_values.get("disorder_seed") is None
            ):
                streamlit.warning(
                    "Set a disorder seed before recording a reproducible study."
                )

    if project_area == "Topological superconductors":
        streamlit.title("TopOSC Lab")
        streamlit.caption("Topological superconductors and lattice-model research workspace")
    elif project_area == "Quantum gases":
        streamlit.title("TopOSC Lab / Quantum gases")
        streamlit.caption("Ideal-gas equilibrium, ensembles and statistically correct dynamics")
    else:
        streamlit.title("TopOSC Lab / Research studies")
        streamlit.caption("Load, compare and export reproducible numerical studies")

    if workspace_mode == "Study explorer":
        _show_study_explorer(streamlit)
        return

    if workspace_mode == "Equilibrium statistics":
        _show_quantum_gas_lab(
            streamlit,
            quantum_gas_parameters,
            quantum_gas_visible_modes,
            quantum_gas_colour_scale_mode,
        )
        return

    if workspace_mode == "Ensembles and dynamics":
        _show_ensemble_dynamics_lab(
            streamlit,
            ensemble_gas_kind,
            ensemble_kind,
            ensemble_parameters,
            ensemble_settings,
        )
        return

    if workspace_mode == "Model guide":
        try:
            _show_model_guide(streamlit, specification, parameter_values)
        except (TypeError, ValueError) as error:
            streamlit.error(f"The selected guide parameters are not valid: {error}")
        return

    if workspace_mode == "Parameter scan":
        if scan_requested:
            try:
                serialized_parameters = json.dumps(parameter_values, sort_keys=True)
                with streamlit.spinner("Running validated parameter scan..."):
                    study = cached_parameter_scan(
                        selected_key,
                        serialized_parameters,
                        scan_parameter_name,
                        scan_start,
                        scan_stop,
                        scan_points,
                    )

                streamlit.session_state["toposc_parameter_scan"] = {
                    "model_key": selected_key,
                    "study": study,
                    "run_id": streamlit.session_state.get("toposc_scan_run_id", 0)
                    + 1,
                }
                streamlit.session_state["toposc_scan_run_id"] = (
                    streamlit.session_state["toposc_parameter_scan"]["run_id"]
                )
            except (TypeError, ValueError) as error:
                streamlit.error(f"The selected scan is not valid: {error}")

        scan_workspace = streamlit.session_state.get("toposc_parameter_scan")

        if scan_workspace is None:
            _show_empty_scan_workspace(streamlit)
        else:
            if selected_key != scan_workspace["model_key"]:
                streamlit.info(
                    "The workspace shows the previous scan. Run the selected "
                    "model to replace it."
                )
            _show_scan_workspace(
                streamlit,
                scan_workspace["study"],
                run_id=scan_workspace["run_id"],
            )
        return

    if run_requested:
        try:
            serialized_parameters = json.dumps(parameter_values, sort_keys=True)
            with streamlit.spinner("Building Hamiltonian and diagonalizing..."):
                result = cached_simulation(selected_key, serialized_parameters)

            streamlit.session_state["toposc_simulation"] = {
                "model_key": selected_key,
                "parameter_values": parameter_values,
                "result": result,
                "run_id": streamlit.session_state.get("toposc_run_id", 0) + 1,
            }
            streamlit.session_state["toposc_run_id"] = (
                streamlit.session_state["toposc_simulation"]["run_id"]
            )
        except (TypeError, ValueError) as error:
            streamlit.error(f"The selected parameters are not valid: {error}")

    simulation = streamlit.session_state.get("toposc_simulation")

    if simulation is None:
        _show_empty_workspace(streamlit)
        return

    active_specification = MODEL_REGISTRY.get(simulation["model_key"])
    active_model = active_specification.build(simulation["parameter_values"])
    result = simulation["result"]

    if selected_key != simulation["model_key"]:
        streamlit.info(
            "The workspace shows the previous simulation. "
            "Run the selected model to replace it."
        )

    streamlit.subheader(active_specification.display_name)
    overview_tab, localization_tab, observables_tab, metadata_tab = streamlit.tabs(
        ("Overview", "Localization", "Observables", "Metadata and export")
    )

    with overview_tab:
        metric_columns = streamlit.columns(4)
        metric_columns[0].metric("Hamiltonian dimension", result.dimension)
        metric_columns[1].metric("Minimum |E|", f"{lowest_abs_energy(result.eigenvalues):.3e}")
        metric_columns[2].metric("Bulk-gap estimate", f"{bulk_gap(result.eigenvalues):.3e}")
        metric_columns[3].metric("Zero modes (1e-8)", count_zero_modes(result.eigenvalues, 1e-8))

        figure, _ = plot_eigenvalue_spectrum(
            result.eigenvalues,
            title=f"{active_specification.display_name} spectrum",
            show=False,
        )
        streamlit.pyplot(figure)
        _add_figure_downloads(
            streamlit,
            figure,
            filename_stem=f"{simulation['model_key']}_spectrum",
            key=f"single-spectrum::{simulation['run_id']}",
        )
        plt.close(figure)

    with localization_tab:
        nearest_zero_state = int(np.argmin(np.abs(result.eigenvalues)))
        state_index = streamlit.slider(
            "Eigenstate index",
            min_value=0,
            max_value=result.n_states - 1,
            value=nearest_zero_state,
            key=f"state::{simulation['run_id']}",
        )
        profile = localization_profile_from_result(result, state_index=state_index)
        streamlit.caption(f"Selected energy: {result.eigenvalues[state_index]:.6e}")

        figure, _ = plot_localization_on_lattice(
            active_model.lattice,
            profile,
            title=f"State {state_index} on {active_specification.display_name}",
            show=False,
        )
        streamlit.pyplot(figure)
        _add_figure_downloads(
            streamlit,
            figure,
            filename_stem=f"{simulation['model_key']}_localization",
            key=f"single-localization::{simulation['run_id']}::{state_index}",
        )
        plt.close(figure)

        localization_columns = streamlit.columns(4)
        localization_columns[0].metric("IPR", f"{profile.inverse_participation_ratio:.4f}")
        localization_columns[1].metric("Participation ratio", f"{profile.participation_ratio:.2f}")
        localization_columns[2].metric("Edge weight", f"{profile.edge_weight:.4f}")
        localization_columns[3].metric("Bulk weight", f"{profile.bulk_weight:.4f}")
        streamlit.caption(
            "Components: " + ", ".join(profile.component_labels)
        )

    with observables_tab:
        tolerance = streamlit.number_input(
            "Zero-mode tolerance",
            min_value=1.0e-12,
            value=1.0e-8,
            format="%.1e",
        )
        hamiltonian = active_model.hamiltonian()
        hermitian = check_hermitian(hamiltonian, tolerance=float(tolerance))

        streamlit.dataframe(
            [
                {
                    "Observable": "Minimum absolute energy",
                    "Value": lowest_abs_energy(result.eigenvalues),
                },
                {
                    "Observable": "Bulk-gap estimate",
                    "Value": bulk_gap(result.eigenvalues, float(tolerance)),
                },
                {
                    "Observable": "Zero-mode count",
                    "Value": count_zero_modes(
                        result.eigenvalues,
                        float(tolerance),
                    ),
                },
                {"Observable": "Hermitian", "Value": hermitian.satisfied},
                {"Observable": "Hermiticity residual", "Value": hermitian.residual},
            ],
            hide_index=True,
            use_container_width=True,
        )
        streamlit.info(
            "Model-specific invariants such as Chern number, Z2 index and Berry "
            "curvature will appear in the upcoming scan and topology workspaces."
        )

    with metadata_tab:
        streamlit.json(
            {
                "model": result.model_name,
                "parameters": result.parameters,
                "basis_layout": {
                    "spatial_shape": result.basis_layout.spatial_shape,
                    "components_per_site": result.basis_layout.components_per_site,
                    "ordering": result.basis_layout.ordering,
                    "component_labels": result.basis_layout.component_labels,
                },
            }
        )
        streamlit.download_button(
            "Download full simulation result (.npz)",
            data=_simulation_archive(result),
            file_name=f"{simulation['model_key']}_result.npz",
            mime="application/octet-stream",
        )


def main() -> None:
    """Starte die App ueber den Konsolenbefehl ``toposc-ui``."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Install the optional interface first: pip install -e '.[app]'"
        ) from error

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    run_app()
