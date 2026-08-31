"""Run and visualize the complete Toposc-Lab Phase 1-7 workflow."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    BasicScoreComponent,
    GeometryEvaluationConfig,
    GeometryEvaluationContext,
    GeometryModelAdapter,
    ModelGeometryRequirements,
    ObjectiveDirection,
    ObjectiveQuantity,
    ObjectiveSpec,
    TopologyIntegrationInput,
    compute_basic_scalar_score,
    evaluate_geometry,
    evaluate_multi_objectives,
)
from toposc_lab.geometry import (
    Geometry,
    honeycomb,
    kagome,
    random_graph,
    save_geometry,
    sierpinski_gasket,
    square,
    validate_geometry,
)
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_bdg_hamiltonian,
    build_spinless_p_wave_pairing,
    build_tight_binding_hamiltonian,
)
from toposc_lab.models.chiral_p_wave import (
    ChiralPWaveModel,
    ChiralPWaveParameters,
)
from toposc_lab.topology import (
    BottIndexResult,
    SymmetryClassification,
    TopologyCapability,
    TopologyDispatchContext,
    TopologyDispatchDecision,
    TopologyMethod,
    bott_index,
    dispatch_topology_methods,
)
from toposc_lab.visualization import plot_geometry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        choices=(
            "square",
            "honeycomb",
            "kagome",
            "sierpinski",
            "random-graph",
        ),
        default="square",
        help="geometry family evaluated by the demo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/masterplan_demo"),
        help="directory for the PNG figure and serialized geometry",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figure without opening an interactive window",
    )
    return parser


class _GraphPWaveModel(BaseModel):
    """Spinless graph p-wave model for abstract oriented geometries."""

    def __init__(
        self,
        geometry: Geometry,
        *,
        hopping: float,
        chemical_potential: float,
        pairing: float,
    ) -> None:
        self.geometry = geometry
        self.hopping = hopping
        self.chemical_potential = chemical_potential
        self.pairing = pairing

    @property
    def model_name(self) -> str:
        return "GraphPWaveModel"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "hopping": self.hopping,
            "chemical_potential": self.chemical_potential,
            "pairing": self.pairing,
            "pairing_convention": "oriented_graph_p_wave",
        }

    @property
    def nambu_basis(self) -> NambuBasis:
        return NambuBasis(n_sites=self.geometry.n_sites, ordering="component_major")

    @property
    def basis_layout(self) -> BasisLayout:
        return self.nambu_basis.basis_layout

    def hamiltonian(self) -> np.ndarray:
        normal = build_tight_binding_hamiltonian(
            self.geometry,
            onsite=-self.chemical_potential,
            hopping=-self.hopping,
        )
        pairing = build_spinless_p_wave_pairing(
            self.geometry,
            pairing=self.pairing,
        )
        return build_bdg_hamiltonian(normal, pairing, basis=self.nambu_basis)


def _geometry(name: str) -> Geometry:
    if name == "square":
        return square(n_x=5, n_y=5)
    if name == "honeycomb":
        return honeycomb(n_x=4, n_y=4)
    if name == "kagome":
        return kagome(n_x=3, n_y=3)
    if name == "sierpinski":
        return sierpinski_gasket(order=3)
    if name == "random-graph":
        return random_graph(50, 0.3, seed=144)
    raise AssertionError(f"unsupported demo geometry: {name}")


def _model_adapter(
    geometry: Geometry,
    parameters: ChiralPWaveParameters,
) -> GeometryModelAdapter:
    if geometry.coordinates is None:
        def graph_model_factory(candidate: Geometry) -> BaseModel:
            return _GraphPWaveModel(
                candidate,
                hopping=parameters.hopping,
                chemical_potential=parameters.chemical_potential,
                pairing=parameters.pairing,
            )

        def graph_nambu_basis_resolver(model: BaseModel) -> NambuBasis:
            if not isinstance(model, _GraphPWaveModel):
                raise TypeError("the abstract-graph demo requires _GraphPWaveModel")
            return model.nambu_basis

        return GeometryModelAdapter(
            model_factory=graph_model_factory,
            requirements=ModelGeometryRequirements(
                require_connected=True,
                require_edges=True,
            ),
            nambu_basis_resolver=graph_nambu_basis_resolver,
        )

    def model_factory(geometry: Geometry) -> BaseModel:
        return ChiralPWaveModel(geometry, parameters)

    def nambu_basis_resolver(model: BaseModel) -> NambuBasis:
        if not isinstance(model, ChiralPWaveModel):
            raise TypeError("the demo requires ChiralPWaveModel")
        return model.nambu_basis

    return GeometryModelAdapter(
        model_factory=model_factory,
        requirements=ModelGeometryRequirements(
            require_connected=True,
            require_edges=True,
            required_spatial_axes=(0, 1),
        ),
        nambu_basis_resolver=nambu_basis_resolver,
    )


TopologyHook = Callable[
    [GeometryEvaluationContext],
    tuple[TopologyIntegrationInput, ...],
]


def _topology_setup(
    geometry_name: str,
) -> tuple[
    TopologyHook | None,
    TopologyDispatchDecision | None,
    list[BottIndexResult],
]:
    """Enable a physically explicit class-D Bott calculation for the square."""
    diagnostics: list[BottIndexResult] = []
    if geometry_name != "square":
        return None, None, diagnostics

    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=1,
        chiral_symmetry=False,
    )
    dispatch = dispatch_topology_methods(
        TopologyDispatchContext(
            physical_dimension=2,
            embedding_dimension=2,
            classification=classification,
            capabilities=frozenset(
                {
                    TopologyCapability.BULK_GAP_EVIDENCE,
                    TopologyCapability.BASIS_COORDINATES,
                    TopologyCapability.COORDINATE_PERIODS,
                }
            ),
        )
    )
    dispatch.require(TopologyMethod.BOTT_2D)

    def topology_hook(
        context: GeometryEvaluationContext,
    ) -> tuple[TopologyIntegrationInput, ...]:
        coordinates = context.geometry.coordinates
        if coordinates is None or coordinates.shape[1] < 2:
            raise ValueError("the Bott demonstration requires explicit 2D coordinates")
        spatial_coordinates = coordinates[:, :2]
        basis_coordinates = np.vstack((spatial_coordinates, spatial_coordinates))
        coordinate_periods = np.ptp(spatial_coordinates, axis=0) + 1.0
        diagnostic = bott_index(
            context.hamiltonian,
            basis_coordinates,
            coordinate_periods,
            classification,
        )
        diagnostics.append(diagnostic)
        return (diagnostic,)

    return topology_hook, dispatch, diagnostics


def main() -> None:
    arguments = _parser().parse_args()
    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    geometry = _geometry(arguments.geometry)
    geometry_report = validate_geometry(geometry)
    geometry_report.raise_for_errors()

    parameters = ChiralPWaveParameters(
        hopping=1.0,
        chemical_potential=0.5,
        pairing=0.8,
        chirality=1,
        plane_axes=(0, 1),
    )
    topology_hook, topology_dispatch, bott_diagnostics = _topology_setup(
        arguments.geometry
    )
    run = evaluate_geometry(
        geometry,
        adapter=_model_adapter(geometry, parameters),
        config=GeometryEvaluationConfig(
            reference_energy=0.0,
            zero_mode_tolerance=1.0e-8,
            low_energy_count=8,
            boundary_localization_threshold=0.5,
            numerical_tolerance=1.0e-10,
            require_resolved_topology=topology_dispatch is not None,
        ),
        topology_hook=topology_hook,
        topology_dispatch=topology_dispatch,
        seed=None,
    )

    if run.evaluation is None or run.simulation_result is None:
        if run.failure is not None:
            details = f"{run.failure.stage.value}: {run.failure.message}"
        elif run.validity.errors:
            details = "; ".join(
                f"{issue.code}: {issue.message}"
                for issue in run.validity.errors
            )
        else:
            details = "evaluation stopped without a result or diagnostic"
        raise RuntimeError(details)
    if not run.is_valid:
        issues = "; ".join(issue.message for issue in run.validity.errors)
        raise RuntimeError(f"the evaluated candidate is invalid: {issues}")

    evaluation = run.evaluation
    simulation = run.simulation_result
    selected_state = min(
        evaluation.low_energy_states,
        key=lambda index: abs(evaluation.low_energy_states[index]),
    )
    localization = evaluation.localization[selected_state]
    majorana = evaluation.majorana_metrics[selected_state]

    score_weights = {
        BasicScoreComponent.NORMALIZED_GAP: 2.0,
        BasicScoreComponent.MAXIMUM_IPR: 1.0,
        BasicScoreComponent.MAXIMUM_BOUNDARY_WEIGHT: 2.0,
        BasicScoreComponent.MAXIMUM_MAJORANA_SELF_CONJUGACY: 2.0,
    }
    if evaluation.topology:
        score_weights[BasicScoreComponent.TOPOLOGICAL_METHOD_FRACTION] = 3.0
    score = compute_basic_scalar_score(
        evaluation,
        weights=score_weights,
        gap_scale=1.0,
    )
    objective_specs = [
        ObjectiveSpec(
            "maximize_gap",
            ObjectiveQuantity.GAP,
            ObjectiveDirection.MAXIMIZE,
        ),
        ObjectiveSpec(
            "minimize_sites",
            ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
            ObjectiveDirection.MINIMIZE,
            descriptor_name="site_count",
        ),
        ObjectiveSpec(
            "maximize_mean_degree",
            ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
            ObjectiveDirection.MAXIMIZE,
            descriptor_name="mean_degree",
        ),
    ]
    if evaluation.topology:
        objective_specs.append(
            ObjectiveSpec(
                "bott_topological",
                ObjectiveQuantity.TOPOLOGY_CLASSIFICATION,
                ObjectiveDirection.MAXIMIZE,
                topology_method=TopologyMethod.BOTT_2D,
            )
        )
    objectives = evaluate_multi_objectives(
        evaluation,
        objectives=objective_specs,
    )

    figure, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    plot_geometry(
        geometry,
        axes=axes[0, 0],
        title=f"Evaluated {arguments.geometry} ({geometry.n_sites} sites)",
        show=False,
    )

    state_indices = np.arange(simulation.eigenvalues.size)
    axes[0, 1].scatter(state_indices, simulation.eigenvalues, s=20.0)
    axes[0, 1].scatter(
        [selected_state],
        [simulation.eigenvalues[selected_state]],
        s=75.0,
        color="tab:red",
        label="nearest-zero state",
        zorder=3,
    )
    axes[0, 1].axhline(0.0, color="0.4", linewidth=0.8)
    axes[0, 1].set_title("Chiral p-wave BdG spectrum")
    axes[0, 1].set_xlabel("Eigenstate index")
    axes[0, 1].set_ylabel("Energy")
    axes[0, 1].legend(loc="best")

    sites = np.arange(geometry.n_sites)
    axes[1, 0].plot(
        sites,
        localization.probability,
        "o-",
        label="site probability",
    )
    axes[1, 0].plot(
        sites,
        majorana.polarization_magnitude,
        "s--",
        label="Majorana polarization magnitude",
    )
    axes[1, 0].set_title(f"Nearest-zero state {selected_state}")
    axes[1, 0].set_xlabel("Site index")
    axes[1, 0].set_ylabel("Weight")
    axes[1, 0].legend(loc="best")

    component_names = [component.value for component in score.components]
    component_values = list(score.components.values())
    contributions = list(score.contributions.values())
    positions = np.arange(len(component_names))
    width = 0.38
    axes[1, 1].bar(
        positions - width / 2.0,
        component_values,
        width,
        label="normalized component",
    )
    axes[1, 1].bar(
        positions + width / 2.0,
        contributions,
        width,
        label="weighted contribution",
    )
    axes[1, 1].set_xticks(positions, component_names, rotation=18, ha="right")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title(f"Transparent scalar score = {score.value:.4f}")
    axes[1, 1].legend(loc="best")

    topology_axes = axes[2, 0]
    if bott_diagnostics:
        bott = bott_diagnostics[0]
        phase_indices = np.arange(bott.commutator_eigenphases.size)
        topology_axes.scatter(
            phase_indices,
            bott.commutator_eigenphases,
            s=24.0,
            color="tab:purple",
        )
        topology_axes.axhline(np.pi, color="0.5", linestyle="--", linewidth=0.8)
        topology_axes.axhline(-np.pi, color="0.5", linestyle="--", linewidth=0.8)
        topology_axes.set_xlabel("Projected-position commutator phase index")
        topology_axes.set_ylabel("Eigenphase")
        topology_axes.set_title(
            "Class-D Bott topology: "
            f"estimate={bott.bott_estimate:.6f}, index={bott.bott_index}"
        )
    else:
        topology_axes.axis("off")
        topology_axes.text(
            0.5,
            0.5,
            "No topology method was applied.\n"
            "The demo does not infer physical dimension\n"
            "or topology from graph appearance.",
            ha="center",
            va="center",
            fontsize=12,
        )

    summary_axes = axes[2, 1]
    summary_axes.axis("off")
    reproducibility = run.reproducibility
    geometry_id = (
        "unavailable"
        if reproducibility is None
        else reproducibility.geometry_id.split(":")[-1][:20] + "..."
    )
    objective_lines = "\n".join(
        f"  {name}: {objective.value} ({objective.spec.direction.value})"
        for name, objective in objectives.objectives.items()
    )
    summary_axes.text(
        0.02,
        0.98,
        "AUDITABLE RESULT\n\n"
        f"Valid candidate: {run.is_valid}\n"
        f"Geometry: {arguments.geometry}\n"
        f"Sites / edges: {geometry.n_sites} / {geometry.n_edges}\n"
        f"Finite spectral gap: {evaluation.gap:.8f}\n"
        f"Boundary weight: {localization.edge_weight:.8f}\n"
        f"Majorana polarization norm: {majorana.polarization_norm:.8f}\n"
        f"Engineering score: {score.value:.8f}\n\n"
        "Separate objectives:\n"
        f"{objective_lines}\n\n"
        f"Exact geometry snapshot: {geometry_id}\n"
        f"Code version: "
        f"{None if reproducibility is None else reproducibility.code_version}",
        ha="left",
        va="top",
        family="monospace",
        fontsize=10.5,
    )

    figure.suptitle(
        "Toposc-Lab master-plan showcase: geometry → BdG → states → topology → score",
        fontsize=16,
    )
    figure_path = output_directory / f"masterplan_evaluation_{arguments.geometry}.png"
    figure.savefig(figure_path, dpi=180)
    geometry_path = save_geometry(
        output_directory / f"evaluated_geometry_{arguments.geometry}.npz",
        geometry,
    )

    print("Toposc-Lab Phase 1-7 evaluation completed.")
    print(f"  valid candidate:       {run.is_valid}")
    print(f"  sites / edges:         {geometry.n_sites} / {geometry.n_edges}")
    print(f"  spectral gap:          {evaluation.gap:.8f}")
    print(f"  numerical zero modes:  {evaluation.zero_mode_count}")
    print(f"  selected state:        {selected_state}")
    print(f"  boundary weight:       {localization.edge_weight:.8f}")
    print(f"  Majorana conjugacy:    {majorana.self_conjugacy:.8f}")
    print(f"  scalar score:          {score.value:.8f}")
    for topology_result in evaluation.topology:
        print(
            f"  topology ({topology_result.method.value}): "
            f"value={topology_result.invariant_value}, "
            f"topological={topology_result.is_topological}"
        )
    print("  objectives:")
    for name, objective in objectives.objectives.items():
        print(
            f"    {name}: {objective.value} "
            f"({objective.spec.direction.value})"
        )
    if run.reproducibility is not None:
        print(f"  geometry ID:           {run.reproducibility.geometry_id}")
        print(f"  code version:          {run.reproducibility.code_version}")
    print(f"  figure:                {figure_path}")
    print(f"  serialized geometry:   {geometry_path.resolve()}")
    if not evaluation.topology:
        print("  note: topology was not inferred for this geometry selection.")

    if arguments.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
