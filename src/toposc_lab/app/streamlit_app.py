"""Erste interaktive Research-Workspace-Oberflaeche fuer TopOSC-Lab.

Die App enthaelt absichtlich keine Modellphysik. Sie liest die registrierten
Pydantic-Parameter, baut ein Modell ueber die Registry und verwendet danach
dieselben Solver, Observablen und Plotfunktionen wie die Python-API.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from types import UnionType
from typing import Any, get_args, get_origin

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.app.registry import MODEL_REGISTRY, ModelSpec
from toposc_lab.observables.localization import localization_profile_from_result
from toposc_lab.observables.spectrum import (
    bulk_gap,
    count_zero_modes,
    lowest_abs_energy,
)
from toposc_lab.observables.symmetries import check_hermitian
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.visualization.lattice_plots import plot_localization_on_lattice
from toposc_lab.visualization.plots import plot_eigenvalue_spectrum


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


def _show_empty_workspace(streamlit: Any) -> None:
    """Erklaere die erste Ausbaustufe der Forschungsoberflaeche."""
    streamlit.info("Choose a model, adjust its parameters, then run a simulation.")
    streamlit.markdown(
        """
        **This first workspace already supports:**

        - every currently implemented model and all of its parameters,
        - spectra, geometry-aware wavefunction plots and core observables,
        - parameter validation, reproducible disorder seeds and result download.

        The next UI modules will add parameter scans, study loading/comparison,
        phase diagrams and batch disorder ensembles.
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

    @streamlit.cache_data(show_spinner=False)
    def cached_simulation(model_key: str, serialized_parameters: str) -> Any:
        return _solve_serialized_model(model_key, serialized_parameters)

    streamlit.title("TopOSC Lab")
    streamlit.caption("Interactive workspace for topological quantum models")

    specifications = MODEL_REGISTRY.specifications()
    specification_by_key = {specification.key: specification for specification in specifications}

    with streamlit.sidebar:
        streamlit.header("Simulation setup")
        selected_key = streamlit.selectbox(
            "Model",
            options=tuple(specification_by_key),
            format_func=lambda key: specification_by_key[key].display_name,
        )
        specification = specification_by_key[selected_key]
        streamlit.caption(specification.category)
        streamlit.write(specification.description)

        with streamlit.form(f"parameters::{selected_key}"):
            parameter_values = _render_parameter_inputs(streamlit, specification)
            run_requested = streamlit.form_submit_button("Run simulation")

        if (
            parameter_values.get("disorder_strength", 0.0) > 0.0
            and parameter_values.get("disorder_seed") is None
        ):
            streamlit.warning(
                "Set a disorder seed before recording a reproducible study."
            )

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
