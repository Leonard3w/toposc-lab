from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    BasicScoreComponent,
    CandidateFailureStage,
    CandidateIssueCategory,
    CandidateIssueSeverity,
    CandidateStageFailure,
    CandidateValidityIssue,
    CandidateValidityReport,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    ObjectiveDirection,
    ObjectiveQuantity,
    ObjectiveSpec,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, GeometryEdge, chain
from toposc_lab.hamiltonians import NambuBasis
from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChainParameters
from toposc_lab.search import (
    DiversityPreservationPolicy,
    ElitismConfig,
    GenerationLoopConfig,
    GenerationLoopResult,
    GenerationReproductionRequest,
    GeometrySearchFamily,
    MultiObjectiveFitnessDefinition,
    NoveltyReferenceCohort,
    NoveltyScoreConfig,
    OffspringProposal,
    ScalarFitnessDefinition,
    SearchCheckpoint,
    SearchCheckpointError,
    TournamentSelectionConfig,
    create_initial_population,
    create_search_checkpoint,
    evaluate_population_novelty,
    geometry_to_genome,
    load_search_checkpoint,
    run_generation_loop,
    save_search_checkpoint,
)
from toposc_lab.search.population_fitness import FitnessPopulationMember


class _CheckpointModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, Any]:
        return {"scale": self.geometry.n_sites}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout((self.geometry.n_sites,))

    def hamiltonian(self) -> np.ndarray:
        size = self.geometry.n_sites
        return np.diag(np.linspace(-size, size, size)).astype(complex)


def _evaluate(member: FitnessPopulationMember) -> GeometryEvaluationRun:
    return evaluate_geometry(
        member.genome.to_geometry(),
        adapter=GeometryModelAdapter(_CheckpointModel),
        seed=member.generation_index * 10 + member.member_index,
        code_version="tests.checkpoint.v1",
    )


def _produce(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
    return tuple(
        OffspringProposal(
            genome=request.selected_members[index].population_member.genome,
            parent_selection_indices=(index,),
            validation_source_selection_index=index,
            operator_identifier="tests.identity.v1",
            operator_seed=request.seed,
        )
        for index in range(request.required_offspring_count)
    )


def _run(**overrides: Any) -> GenerationLoopResult:
    genomes = tuple(
        geometry_to_genome(
            Geometry(
                n_sites=size,
                edges=tuple(GeometryEdge(i, i + 1) for i in range(size - 1)),
                metadata={"fixture": "checkpoint", "array": np.array([1, 2])},
            )
        )
        for size in (2, 4, 6)
    )
    options: dict[str, Any] = {
        "initial_population": create_initial_population(genomes),
        "definition": ScalarFitnessDefinition(
            weights={BasicScoreComponent.NORMALIZED_GAP: 1.0},
            direction=ObjectiveDirection.MAXIMIZE,
        ),
        "evaluator": _evaluate,
        "config": GenerationLoopConfig(
            generation_count=2,
            selection=TournamentSelectionConfig(selection_count=3, tournament_size=1),
            elitism=ElitismConfig(minimum_elite_count=1),
            diversity=DiversityPreservationPolicy(minimum_distinct_families=1),
        ),
        "seed": 912,
        "offspring_producer": _produce,
        "producer_identifier": "tests.identity-producer.v1",
        "family_classifier": lambda genome: GeometrySearchFamily("declared", 1),
        "family_classifier_identifier": "tests.declared-family.v1",
    }
    options.update(overrides)
    return run_generation_loop(**options)


def _checkpoint(result: GenerationLoopResult | None = None) -> SearchCheckpoint:
    return create_search_checkpoint(
        _run() if result is None else result,
        evaluator_identifier="tests.checkpoint-evaluator.v1",
        code_version="tests.checkpoint.v1",
        requested_generation_count=5,
    )


def _rewrite(path: Path, transform: Any, *, update_checksum: bool = True) -> None:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        document = json.loads(archive.read("payload.json"))
    transform(manifest, document)
    payload = json.dumps(document, separators=(",", ":")).encode()
    if update_checksum:
        manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("payload.json", payload)


def test_roundtrip_preserves_history_arrays_shared_identities_and_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint()
    path = save_search_checkpoint(tmp_path / "search.zip", checkpoint)
    loaded = load_search_checkpoint(path)
    assert loaded.completed_generation_index == 2
    assert loaded.requested_generation_count == 5
    assert loaded.runtime_versions == checkpoint.runtime_versions
    assert loaded.evaluator_identifier == checkpoint.evaluator_identifier
    assert loaded.code_version == checkpoint.code_version
    restored = loaded.result
    original = checkpoint.result
    assert restored.seed == original.seed
    assert restored.initial_fitness.population is restored.initial_population
    assert restored.initial_fitness.definition is restored.definition
    assert restored.initial_diversity is not None
    assert restored.initial_diversity.population is restored.initial_population
    for index, transition in enumerate(restored.transitions):
        assert transition.source_fitness is restored.fitness_history[index]
        assert transition.elitism.source is transition.source_fitness
        assert transition.elitism.config is restored.config.elitism
        assert transition.population.validity_policy is restored.initial_population.validity_policy
        assert transition.fitness.definition is restored.definition
        assert transition.diversity is not None
        assert transition.diversity.policy is restored.config.diversity
        if transition.selection is not None:
            assert transition.selection.source is transition.source_fitness
            assert transition.selection.config is restored.config.selection
        assert transition.selection_seed == original.transitions[index].selection_seed
        for elite, member in zip(transition.elitism.elite_members, transition.population.members):
            assert elite.population_member.genome is member.genome
    for before, after in zip(original.fitness_history, restored.fitness_history, strict=True):
        for first, second in zip(before.members, after.members, strict=True):
            assert first.status is second.status
            assert first.evaluation is not None and second.evaluation is not None
            assert first.evaluation.reproducibility == second.evaluation.reproducibility
            first_sim = first.evaluation.simulation_result
            second_sim = second.evaluation.simulation_result
            assert first_sim is not None and second_sim is not None
            np.testing.assert_array_equal(first_sim.eigenvectors, second_sim.eigenvectors)
            np.testing.assert_array_equal(first_sim.eigenvalues, second_sim.eigenvalues)
            assert not second_sim.eigenvectors.flags.writeable


def test_novelty_cohort_and_neighbor_references_survive(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    population = checkpoint.result.populations[-1]
    report = evaluate_population_novelty(
        population,
        config=NoveltyScoreConfig(1),
        distance=lambda first, second: float(abs(first.n_sites - second.n_sites)),
        distance_identifier="tests.site-count.v1",
        reference_cohort=NoveltyReferenceCohort("tests.reference.v1", (population.genomes[0],)),
    )
    checkpoint = replace(checkpoint, novelty_reports=(report,))
    loaded = load_search_checkpoint(save_search_checkpoint(tmp_path / "novelty.zip", checkpoint))
    restored = loaded.novelty_reports[0]
    assert restored.population is loaded.result.populations[-1]
    assert restored.scores == report.scores
    assert restored.reference_cohort is not None
    assert restored.reference_cohort.genomes[0] is restored.population.genomes[0]
    for member in restored.members:
        assert any(member.nearest_neighbors[0] is item for item in member.comparisons)


def test_multiobjective_fitness_and_fully_elite_transitions_roundtrip(tmp_path: Path) -> None:
    definition = MultiObjectiveFitnessDefinition(
        objectives=(ObjectiveSpec("gap", ObjectiveQuantity.GAP, ObjectiveDirection.MAXIMIZE),)
    )
    genome = geometry_to_genome(chain(4))
    result = _run(
        initial_population=create_initial_population((genome,) * 3),
        definition=definition,
    )
    loaded = load_search_checkpoint(
        save_search_checkpoint(tmp_path / "multi.zip", _checkpoint(result))
    )
    assert isinstance(loaded.result.definition, MultiObjectiveFitnessDefinition)
    assert all(item.selection is None for item in loaded.result.transitions)
    assert (
        loaded.result.initial_population.genomes[0] is loaded.result.initial_population.genomes[1]
    )


def test_callback_failure_ledger_remains_present(tmp_path: Path) -> None:
    def evaluator(member: FitnessPopulationMember) -> GeometryEvaluationRun:
        if member.member_index == 0:
            raise RuntimeError("record this failure")
        return _evaluate(member)

    checkpoint = _checkpoint(_run(evaluator=evaluator))
    loaded = load_search_checkpoint(save_search_checkpoint(tmp_path / "failure.zip", checkpoint))
    for fitness in loaded.result.fitness_history:
        assert fitness.members[0].failure is not None
        assert fitness.members[0].failure.message == "record this failure"
        assert fitness.members[0].fitness is None


def test_callback_saves_each_completed_boundary_and_retains_planned_target(tmp_path: Path) -> None:
    observed: list[int] = []

    def save(prefix: GenerationLoopResult) -> None:
        observed.append(prefix.config.generation_count)
        save_search_checkpoint(tmp_path / "latest.zip", _checkpoint(prefix))

    result = _run(checkpoint_callback=save)
    assert observed == [0, 1, 2]
    loaded = load_search_checkpoint(tmp_path / "latest.zip")
    assert loaded.completed_generation_index == result.config.generation_count
    assert loaded.requested_generation_count == 5


def test_later_producer_failure_keeps_last_completed_checkpoint(tmp_path: Path) -> None:
    def producer(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
        if request.target_generation_index == 1:
            raise RuntimeError("simulated interruption")
        return _produce(request)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run(
            offspring_producer=producer,
            checkpoint_callback=lambda prefix: save_search_checkpoint(
                tmp_path / "latest.zip",
                _checkpoint(prefix),
            ),
        )
    assert load_search_checkpoint(tmp_path / "latest.zip").completed_generation_index == 0


def test_saving_failure_stops_before_next_generation(tmp_path: Path) -> None:
    calls: list[int] = []

    def producer(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
        calls.append(request.target_generation_index)
        return _produce(request)

    def fail(prefix: GenerationLoopResult) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _run(offspring_producer=producer, checkpoint_callback=fail)
    assert calls == []


def test_atomic_replacement_failure_preserves_previous_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    checkpoint = _checkpoint()
    path = save_search_checkpoint(tmp_path / "search.zip", checkpoint)
    before = path.read_bytes()

    def fail(*args: Any) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr("toposc_lab.search.checkpoint.os.replace", fail)
    with pytest.raises(OSError, match="replacement failed"):
        save_search_checkpoint(path, checkpoint)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_unsupported_metadata_fails_before_destination_write(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    path = save_search_checkpoint(tmp_path / "search.zip", checkpoint)
    before = path.read_bytes()
    simulation = checkpoint.result.initial_fitness.members[0].evaluation.simulation_result
    simulation.metadata["unsupported"] = object()
    with pytest.raises(SearchCheckpointError, match="unsupported checkpoint value"):
        save_search_checkpoint(path, checkpoint)
    assert path.read_bytes() == before


def test_exclusive_generations_publish_with_previous_checkpoint_open(
    tmp_path: Path, monkeypatch: Any
) -> None:
    checkpoint = _checkpoint()

    def denied_replace(*args: Any) -> None:
        error = PermissionError("simulated Windows checkpoint replacement denial")
        error.winerror = 5
        raise error

    monkeypatch.setattr("toposc_lab.search.checkpoint.os.replace", denied_replace)
    first = save_search_checkpoint(tmp_path / "first.zip", checkpoint, overwrite=False)
    original = first.read_bytes()
    with first.open("rb") as held_open:
        second = save_search_checkpoint(tmp_path / "second.zip", checkpoint, overwrite=False)
        assert held_open.read() == original
        assert load_search_checkpoint(second).completed_generation_index == 2
        with pytest.raises(FileExistsError):
            save_search_checkpoint(first, checkpoint, overwrite=False)
    assert first.read_bytes() == second.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["first.zip", "second.zip"]


def test_exclusive_publication_failure_cleans_only_its_temporary_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    checkpoint = _checkpoint()
    first = save_search_checkpoint(tmp_path / "first.zip", checkpoint, overwrite=False)
    original = first.read_bytes()

    def fail(*args: Any) -> None:
        raise PermissionError("publication denied")

    monkeypatch.setattr("toposc_lab.search.checkpoint.os.link", fail)
    with pytest.raises(PermissionError, match="publication denied"):
        save_search_checkpoint(tmp_path / "second.zip", checkpoint, overwrite=False)
    assert first.read_bytes() == original
    assert list(tmp_path.iterdir()) == [first]


@pytest.mark.parametrize("overwrite", [None, 0, "False"])
def test_invalid_overwrite_flag_fails_before_writing(tmp_path: Path, overwrite: Any) -> None:
    with pytest.raises(TypeError, match="overwrite must be bool"):
        save_search_checkpoint(tmp_path / "checkpoint.zip", _checkpoint(), overwrite=overwrite)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field,value", (("schema_version", 99), ("schema_version", True), ("format", "pickle"))
)
def test_manifest_compatibility_is_explicit(tmp_path: Path, field: str, value: Any) -> None:
    path = save_search_checkpoint(tmp_path / "search.zip", _checkpoint())
    _rewrite(path, lambda manifest, doc: manifest.update({field: value}))
    with pytest.raises(SearchCheckpointError, match="unsupported checkpoint"):
        load_search_checkpoint(path)


def test_checksum_detects_payload_corruption(tmp_path: Path) -> None:
    path = save_search_checkpoint(tmp_path / "search.zip", _checkpoint())
    _rewrite(path, lambda manifest, doc: doc.update({"unexpected": 1}), update_checksum=False)
    with pytest.raises(SearchCheckpointError, match="checksum"):
        load_search_checkpoint(path)


@pytest.mark.parametrize("kind", ("seed", "version", "class", "reference", "array", "extra_field"))
def test_semantic_tampering_is_rejected_even_with_updated_checksum(
    tmp_path: Path, kind: str
) -> None:
    path = save_search_checkpoint(tmp_path / "search.zip", _checkpoint())

    def tamper(manifest: Any, document: Any) -> None:
        nodes = document["nodes"]
        result = next(node for node in nodes if node["type"] == "GenerationLoopResult")
        if kind == "seed":
            result["fields"]["seed"] += 1
        elif kind == "version":
            result["fields"]["version"] = 999
        elif kind == "class":
            result["type"] = "os.system"
        elif kind == "reference":
            result["fields"]["initial_population"] = ["ref", len(nodes)]
        elif kind == "extra_field":
            result["fields"]["injected"] = 1
        else:
            simulation = next(node for node in nodes if node["type"] == "SimulationResult")
            simulation["fields"]["eigenvalues"][2] = [1000000000]

    _rewrite(path, tamper)
    with pytest.raises(SearchCheckpointError):
        load_search_checkpoint(path)


def test_size_limits_and_container_members_are_checked(tmp_path: Path) -> None:
    path = save_search_checkpoint(tmp_path / "search.zip", _checkpoint())
    with pytest.raises(SearchCheckpointError, match="max_archive_bytes"):
        load_search_checkpoint(path, max_archive_bytes=10)
    with pytest.raises(SearchCheckpointError, match="max_uncompressed_bytes"):
        load_search_checkpoint(path, max_uncompressed_bytes=10)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("../unexpected", b"data")
    with pytest.raises(SearchCheckpointError, match="exactly"):
        load_search_checkpoint(path)
    assert not (tmp_path.parent / "unexpected").exists()


def test_invalid_checkpoint_target_and_unrelated_novelty_are_rejected() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(ValueError, match="cover"):
        replace(checkpoint, requested_generation_count=1)
    unrelated = evaluate_population_novelty(
        _run().initial_population,
        config=NoveltyScoreConfig(1),
        distance=lambda first, second: 0.0,
        distance_identifier="tests.zero.v1",
    )
    with pytest.raises(ValueError, match="exact stored population"):
        replace(checkpoint, novelty_reports=(unrelated,))


def test_same_snapshot_produces_identical_file_bytes(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    first = save_search_checkpoint(tmp_path / "first.zip", checkpoint)
    second = save_search_checkpoint(tmp_path / "second.zip", checkpoint)
    assert first.read_bytes() == second.read_bytes()


def test_checkpoint_observer_preserves_search_seeds_and_decisions() -> None:
    original = _run()
    snapshots: list[GenerationLoopResult] = []
    observed = _run(checkpoint_callback=snapshots.append)
    assert [item.config.generation_count for item in snapshots] == [0, 1, 2]
    for first, second in zip(original.transitions, observed.transitions, strict=True):
        assert first.selection_seed == second.selection_seed
        assert first.reproduction_seed == second.reproduction_seed
        assert tuple(genome.n_sites for genome in first.population.genomes) == tuple(
            genome.n_sites for genome in second.population.genomes
        )
        if first.selection is not None:
            assert second.selection is not None
            assert tuple(record.winner.member_index for record in first.selection.records) == tuple(
                record.winner.member_index for record in second.selection.records
            )


def test_zero_generation_checkpoint_without_diversity_roundtrips(tmp_path: Path) -> None:
    snapshots: list[GenerationLoopResult] = []
    result = _run(
        config=GenerationLoopConfig(
            generation_count=0,
            selection=TournamentSelectionConfig(3, 1),
            elitism=ElitismConfig(1),
        ),
        family_classifier=None,
        family_classifier_identifier=None,
        checkpoint_callback=snapshots.append,
    )
    assert len(snapshots) == 1
    loaded = load_search_checkpoint(
        save_search_checkpoint(tmp_path / "initial.zip", _checkpoint(result))
    )
    assert loaded.result.transitions == ()
    assert loaded.result.initial_diversity is None


def test_invalid_evaluation_diagnostics_are_retained(tmp_path: Path) -> None:
    failure = CandidateStageFailure(
        stage=CandidateFailureStage.MODEL_CONSTRUCTION,
        error_type="RuntimeError",
        message="declared invalid evaluation",
    )
    issue = CandidateValidityIssue(
        code="model_construction_failure",
        message=failure.message,
        category=CandidateIssueCategory.EXECUTION,
        severity=CandidateIssueSeverity.ERROR,
    )

    def evaluator(member: FitnessPopulationMember) -> GeometryEvaluationRun:
        if member.member_index == 0:
            return GeometryEvaluationRun(
                simulation_result=None,
                evaluation=None,
                validity=CandidateValidityReport(issues=(issue,)),
                failure=failure,
            )
        return _evaluate(member)

    checkpoint = _checkpoint(_run(evaluator=evaluator))
    loaded = load_search_checkpoint(save_search_checkpoint(tmp_path / "invalid.zip", checkpoint))
    retained = loaded.result.initial_fitness.members[0]
    assert retained.evaluation is not None
    assert retained.evaluation.failure == failure
    assert retained.evaluation.validity.issues == (issue,)
    assert retained.fitness is None


@pytest.mark.parametrize("field", ("evaluator_identifier", "code_version"))
def test_external_policy_provenance_cannot_be_empty(field: str) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        replace(_checkpoint(), **{field: " "})


def test_duplicate_zip_members_are_rejected(tmp_path: Path) -> None:
    path = save_search_checkpoint(tmp_path / "search.zip", _checkpoint())
    with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
        archive.writestr("manifest.json", "{}")
    with pytest.raises(SearchCheckpointError, match="exactly"):
        load_search_checkpoint(path)


def test_recomputed_integer_version_cannot_be_replaced_by_boolean(tmp_path: Path) -> None:
    path = save_search_checkpoint(tmp_path / "search.zip", _checkpoint())

    def tamper(manifest: Any, document: Any) -> None:
        node = next(node for node in document["nodes"] if node["type"] == "PopulationFitnessResult")
        node["fields"]["version"] = True

    _rewrite(path, tamper)
    with pytest.raises(SearchCheckpointError, match="altered"):
        load_search_checkpoint(path)


def test_real_bdg_majorana_and_localization_results_roundtrip(tmp_path: Path) -> None:
    genome = geometry_to_genome(chain(6))

    def evaluator(member: FitnessPopulationMember) -> GeometryEvaluationRun:
        return evaluate_geometry(
            member.genome.to_geometry(),
            adapter=GeometryModelAdapter(
                model_factory=lambda geometry: GeometryKitaevChain(
                    KitaevChainParameters(
                        n_sites=geometry.n_sites,
                        hopping=1.0,
                        chemical_potential=0.4,
                        pairing=0.8,
                    ),
                ),
                nambu_basis_resolver=lambda model: NambuBasis(
                    n_sites=6,
                    ordering="component_major",
                ),
            ),
            seed=0,
            code_version="tests.kitaev.v1",
        )

    probe = evaluator(create_initial_population((genome,)).members[0])
    assert probe.is_valid, (probe.failure, probe.validity)
    original = _checkpoint(
        _run(
            initial_population=create_initial_population((genome,) * 3),
            evaluator=evaluator,
        )
    )
    loaded = load_search_checkpoint(save_search_checkpoint(tmp_path / "bdg.zip", original))
    first = original.result.final_fitness.members[0].evaluation.evaluation
    second = loaded.result.final_fitness.members[0].evaluation.evaluation
    assert first.majorana_metrics and second.majorana_metrics
    for index in first.majorana_metrics:
        np.testing.assert_array_equal(
            first.majorana_metrics[index].polarization,
            second.majorana_metrics[index].polarization,
        )
        assert first.majorana_metrics[index].total_polarization == (
            second.majorana_metrics[index].total_polarization
        )
        np.testing.assert_array_equal(
            first.localization[index].component_probabilities,
            second.localization[index].component_probabilities,
        )


def test_metadata_scalar_container_and_dtype_semantics_roundtrip(tmp_path: Path) -> None:
    original = _checkpoint()
    simulation = original.result.initial_fitness.members[0].evaluation.simulation_result
    simulation.metadata.update(
        {
            "scalar": np.float32(1.25),
            "complex": 2 + 3j,
            "bytes": b"\x00\xff",
            "typed_keys": {2: ("tuple", True)},
            "set": frozenset({1, 2}),
            "array": np.array([[1, 2]], dtype=">i4"),
        }
    )
    loaded = load_search_checkpoint(save_search_checkpoint(tmp_path / "metadata.zip", original))
    metadata = loaded.result.initial_fitness.members[0].evaluation.simulation_result.metadata
    assert type(metadata["scalar"]) is np.float32
    assert metadata["complex"] == 2 + 3j
    assert metadata["bytes"] == b"\x00\xff"
    assert metadata["typed_keys"] == {2: ("tuple", True)}
    assert metadata["set"] == frozenset({1, 2})
    assert metadata["array"].dtype.str == ">i4"
    np.testing.assert_array_equal(metadata["array"], simulation.metadata["array"])


def test_four_dimensional_and_oriented_geometry_snapshot_roundtrip(tmp_path: Path) -> None:
    geometry = Geometry(
        n_sites=2,
        coordinates=np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]),
        embedding_dimension=4,
        edges=(GeometryEdge(1, 0),),
        boundary_sites=frozenset({1}),
    )
    genome = geometry_to_genome(geometry)
    original = _checkpoint(_run(initial_population=create_initial_population((genome,) * 3)))
    loaded = load_search_checkpoint(save_search_checkpoint(tmp_path / "4d.zip", original))
    restored = loaded.result.initial_population.genomes[0]
    assert restored.embedding_dimension == 4
    assert restored.edges == (GeometryEdge(1, 0),)
    assert restored.boundary_sites == frozenset({1})
    np.testing.assert_array_equal(restored.coordinates, geometry.coordinates)
