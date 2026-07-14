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

    streamlit.title("TopOSC Lab")
    streamlit.caption("Interactive workspace for topological quantum models")

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
        workspace_mode = streamlit.radio(
            "Workspace",
            options=("Single simulation", "Parameter scan", "Study explorer"),
        )
        if workspace_mode == "Study explorer":
            streamlit.header("Study explorer")
            streamlit.caption("Load, compare and export reproducible studies.")
        else:
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

    if workspace_mode == "Study explorer":
        _show_study_explorer(streamlit)
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
