"""Contract tests; campaign fixtures use a labelled tiny nonphysical evaluator.

No main/reference/held-out research seed is evaluated with the real physics.
The one real adapter test uses only a reserved preflight geometry seed.
"""

from __future__ import annotations

from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import numpy as np
import pytest
from scipy.stats import binomtest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import GeometryModelAdapter, ObjectiveDirection, evaluate_geometry
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    DerivedEvaluationRun,
    LexicographicFitness,
    LexicographicFitnessDefinition,
    create_initial_population,
    create_search_checkpoint,
    evaluate_population_fitness,
    identify_population_elites,
    load_search_checkpoint,
    resume_search,
    save_search_checkpoint,
    select_population_members,
)
from toposc_lab.search import phase_10_campaign as campaign
from toposc_lab.search._research_report import (
    exact_paired_pvalue,
    research_summary,
    wilson_interval,
)
from toposc_lab.search._research_runtime import ResearchMonitor
from toposc_lab.search._research_storage import (
    AttemptLedger,
    ResearchAbort,
    decode_record,
    encode_record,
    load_record,
    load_sealed,
    save_record,
)
from toposc_lab.search.generation_loop import GenerationReproductionRequest, OffspringProposal
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.mutation_validity import MutationValidityPolicy
from toposc_lab.search.phase_10_research import (
    DERIVATION_ID,
    PREFLIGHT_SEEDS,
    QUANTITIES,
    RESEARCH_PROTOCOL_COMMIT,
    TRIAL_SEEDS,
    ResearchEvaluation,
    evaluate_research_geometry,
    legal_edge_swaps,
    produce_research_offspring,
    research_protocol,
    sample_research_genome,
)
from toposc_lab.search.population_elitism import ElitismConfig
from toposc_lab.search.population_selection import TournamentSelectionConfig
from toposc_lab.search.search_benchmark import _seed_schedule, run_search_benchmark


class ToyModel(BaseModel):
    @property
    def parameters(self) -> dict[str, Any]:
        return {"test_fixture": True}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout((3,))

    def hamiltonian(self) -> np.ndarray:
        return np.diag([-1.0, 1.0, 2.0]).astype(complex)


def toy_genome(seed: int = 0) -> GeometryGenome:
    edges = [GeometryEdge(0, 1), GeometryEdge(1, 2)]
    if seed % 2:
        edges.append(GeometryEdge(0, 2))
    return GeometryGenome.from_geometry(
        Geometry(
            n_sites=3,
            edges=tuple(edges),
            coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            boundary_sites=frozenset({0, 1}),
        )
    )


def toy_evaluation(
    geometry: Geometry, code: str = "tests.research", *, descriptive: str | None = None
) -> ResearchEvaluation:
    run = evaluate_geometry(
        geometry, adapter=GeometryModelAdapter(lambda _: ToyModel()), seed=None, code_version=code
    )
    values = (1.0, 0.24 if len(geometry.edges) == 2 else 0.30, 0.9)
    derived = DerivedEvaluationRun.from_run(
        run, identifier=DERIVATION_ID, quantities=dict(zip(QUANTITIES, values, strict=True))
    )
    return ResearchEvaluation(
        derived,
        {
            "clean_eligible": True,
            "localizer_protection_proxy": values[1],
            "minimum_boundary_weight_first_four": values[2],
            "gate_reasons": [],
            "test_fixture": True,
            "descriptive": descriptive,
        },
    )


def toy_protocol(
    code: str = "tests.research", proxy: float = 0.24, seeds: tuple[int, ...] = (31,)
) -> Any:
    return replace(research_protocol(code, proxy, seeds), validity_policy=MutationValidityPolicy())


def toy_producer(request: Any, *, record: Any = None) -> tuple[OffspringProposal, ...]:
    stream = np.random.PCG64(request.seed)
    result = []
    for index in range(request.required_offspring_count):
        seed = int(stream.random_raw())
        if record:
            record(
                {
                    "generation": request.target_generation_index,
                    "parent_slot": index,
                    "operator_seed": seed,
                    "legal_swap_count": 0,
                    "chosen_swap": None,
                    "reason": "no_legal_swap",
                }
            )
        parent = request.selected_members[index].population_member.genome
        result.append(OffspringProposal(parent, (index,), index, "toy.noop", seed))
    return tuple(result)


def toy_trial(seed: int = 31) -> Any:
    return run_search_benchmark(
        toy_protocol(seeds=(seed,)),
        sampler=toy_genome,
        evaluator=lambda genome, _: toy_evaluation(genome.to_geometry()).run,
        offspring_producer=toy_producer,
    ).trials[0]


def test_ordered_fitness_drives_selection_and_complete_elite_ties() -> None:
    definition = LexicographicFitnessDefinition(
        ("gate", "protection", "boundary"), (ObjectiveDirection.MAXIMIZE,) * 3, "test.v1"
    )
    population = create_initial_population(
        [toy_genome()] * 4, validity_policy=MutationValidityPolicy()
    )
    values = [(0.0, 100.0, 1.0), (1.0, 0.3, 0.9), (1.0, 0.3, 0.9), (1.0, 0.2, 1.0)]

    def evaluate(member: Any) -> Any:
        run = toy_evaluation(member.genome.to_geometry()).run
        return DerivedEvaluationRun.from_run(
            run,
            identifier="test.v1",
            quantities=dict(zip(definition.quantities, values[member.member_index], strict=True)),
        )

    fitness = evaluate_population_fitness(population, definition=definition, evaluator=evaluate)
    elites = identify_population_elites(fitness, config=ElitismConfig(1))
    assert elites.elite_count == 2
    assert elites.tiers[0].member_indices == (1, 2)
    selection = select_population_members(fitness, config=TournamentSelectionConfig(20, 4), seed=4)
    assert {m.member_index for m in selection.selected_members} <= {1, 2}
    assert all(isinstance(m.fitness, LexicographicFitness) for m in fitness.members)
    assert (
        fitness.members[0].evaluation.evaluation.geometry_descriptors
        == evaluate(population.members[0]).evaluation.geometry_descriptors
    )


def test_lexicographic_minimize_and_bad_derivations() -> None:
    definition = LexicographicFitnessDefinition(("x",), (ObjectiveDirection.MINIMIZE,), "test.v1")
    assert (
        LexicographicFitness(definition, (2.0,)).ordering_key
        < LexicographicFitness(definition, (1.0,)).ordering_key
    )
    for bad in (True, float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            LexicographicFitness(definition, (bad,))
    with pytest.raises(ValueError):
        LexicographicFitnessDefinition(("x", "x"), (ObjectiveDirection.MAXIMIZE,) * 2, "v1")
    population = create_initial_population([toy_genome()], validity_policy=MutationValidityPolicy())
    unavailable = evaluate_population_fitness(
        population,
        definition=definition,
        evaluator=lambda _: toy_evaluation(toy_genome().to_geometry()).run,
    )
    assert unavailable.members[0].status.value == "fitness_construction_failure"
    assert unavailable.members[0].fitness is None


def test_deterministic_seed_schedule_and_equal_budgets_roundtrip(tmp_path: Path) -> None:
    protocol = toy_protocol()
    words = tuple(int(w) for w in np.random.PCG64(31).random_raw(33))
    assert _seed_schedule(31, protocol) == (words[0], words[1:], (None,) * 32)
    trial = toy_trial()
    assert trial.evolution_arm.final_progress.attempt_count == 32
    assert trial.random_arm.final_progress.attempt_count == 32
    assert trial.evolution.initial_population is trial.random_arm.fitness_history[0].population
    assert all(
        m.evaluation.reproducibility.seed is None
        for arm in (trial.evolution_arm, trial.random_arm)
        for history in arm.fitness_history
        for m in history.members
    )
    restored = decode_record(encode_record(trial))
    assert (
        restored.evolution.initial_population is restored.random_arm.fitness_history[0].population
    )
    np.testing.assert_array_equal(
        restored.evolution.initial_fitness.members[0].evaluation.simulation_result.eigenvalues,
        trial.evolution.initial_fitness.members[0].evaluation.simulation_result.eigenvalues,
    )
    checkpoint = create_search_checkpoint(
        trial.evolution, evaluator_identifier=DERIVATION_ID, code_version=protocol.code_version
    )
    path = tmp_path / "search.zip"
    save_search_checkpoint(path, checkpoint)
    loaded = load_search_checkpoint(path)
    assert isinstance(loaded.result.definition, LexicographicFitnessDefinition)
    result = resume_search(
        loaded,
        evaluator=lambda _: pytest.fail("completed search reevaluated"),
        evaluator_identifier=DERIVATION_ID,
        offspring_producer=toy_producer,
        producer_identifier=protocol.producer_identifier,
        code_version=protocol.code_version,
    )
    assert result.result.config.generation_count == 3


def test_exact_statistics_all_discordant_counts_and_zero_cases() -> None:
    assert exact_paired_pvalue(0, 0) == 1
    assert exact_paired_pvalue(0, 6) == 0.03125
    for n in range(1, 33):
        for b in range(n + 1):
            assert exact_paired_pvalue(b, n - b) == pytest.approx(
                binomtest(b, n, 0.5).pvalue, abs=1e-14
            )
    assert wilson_interval(0, 32)[0] == 0
    assert wilson_interval(32, 32)[1] == 1
    with pytest.raises(ValueError):
        exact_paired_pvalue(True, 0)
    summary = research_summary((toy_trial(),), 0.24, preflight=True)
    assert summary["exact_mcnemar_pvalue"] is None
    assert summary["primary_decision"] == "preflight_only"
    assert summary["arms"]["evolution"]["trials"][0]["first_hit"] is not None
    assert len(summary["arms"]["evolution"]["trials"][0]["elite_counts_by_transition"]) == 3
    assert summary["arms"]["random"]["trials"][0]["elite_counts_by_transition"] == []


def test_ledger_replay_failure_cannot_be_replaced_by_success(tmp_path: Path) -> None:
    timings: list[float] = []
    ledger = AttemptLedger(tmp_path / "unit", on_storage=timings.append)
    ledger.record("input.json", toy_genome())
    ledger.record("outcome.json", {"error": "LinAlgError"})
    assert len(timings) == 2 and all(seconds >= 0 for seconds in timings)
    assert ledger.storage_seconds == sum(timings)
    replay = AttemptLedger(tmp_path / "unit")
    replay.record("input.json", toy_genome())
    with pytest.raises(ResearchAbort, match="replay differs"):
        replay.record("outcome.json", {"success": True})
    assert load_record(ledger.execution / "outcome.json") == {"error": "LinAlgError"}
    with pytest.raises(ResearchAbort, match="omitted"):
        replay.seal("incomplete")


def test_ledger_hash_tampering_and_exclusive_publication(tmp_path: Path) -> None:
    ledger = AttemptLedger(tmp_path / "unit")
    ledger.record("input.json", toy_genome())
    ledger.seal({"complete": True})
    assert load_sealed(tmp_path / "unit") == {"complete": True}
    with pytest.raises(ResearchAbort):
        save_record(ledger.execution / "input.json", "overwrite")
    path = ledger.execution / "input.json"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ResearchAbort, match="hash differs"):
        load_sealed(tmp_path / "unit")


def test_real_reserved_geometry_operator_and_scientific_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genome = sample_research_genome(PREFLIGHT_SEEDS[2])
    swaps = legal_edge_swaps(genome)
    assert swaps == tuple(sorted(swaps))
    assert swaps
    assert genome.n_sites == 64 and len(genome.edges) == 112
    # This sole real physics call uses a designated preflight seed, never a search seed.
    evaluated = evaluate_research_geometry(genome.to_geometry(), "tests.preflight-only")
    assert evaluated.run.reproducibility.seed is None
    assert evaluated.scientific["topology_grid"] is not None
    if evaluated.run.is_valid:
        assert isinstance(evaluated.run, DerivedEvaluationRun)
        assert evaluated.run.derived_quantities["clean_eligible"] == float(
            evaluated.scientific["clean_eligible"]
        )
        assert (
            evaluated.run.derived_quantities["localizer_protection_proxy"]
            == evaluated.scientific["localizer_protection_proxy"]
        )
        protocol = research_protocol("tests.preflight-only", 0.24, (PREFLIGHT_SEEDS[0],))
        initial = create_initial_population([genome] * 2, validity_policy=protocol.validity_policy)
        fitness = evaluate_population_fitness(
            initial, definition=protocol.definition, evaluator=lambda _: evaluated.run
        )
        selection = select_population_members(
            fitness, config=TournamentSelectionConfig(2, 2), seed=11
        )
        request = GenerationReproductionRequest(
            fitness, selection, 1, 1, 12, protocol.validity_policy
        )
        records: list[dict[str, Any]] = []
        child = produce_research_offspring(request, record=records.append)[0].genome
        assert len(child.edges) == 112 and child.n_sites == 64
        assert child.boundary_sites == genome.boundary_sites
        np.testing.assert_array_equal(child.coordinates, genome.coordinates)
        assert child.edges != genome.edges
        assert records[0]["legal_swap_count"] == len(swaps)
        from toposc_lab.search import phase_10_research as physical

        monkeypatch.setattr(physical, "legal_edge_swaps", lambda _: ())
        records.clear()
        noop = produce_research_offspring(request, record=records.append)[0]
        assert noop.genome is genome
        assert records[0]["reason"] == "no_legal_swap"
        assert noop.operator_seed is not None


class QuietMonitor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: list[Any] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def emit(self, event: str, **data: Any) -> None:
        self.events.append((event, data))


@pytest.fixture
def toy_campaign(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        campaign,
        "research_environment",
        lambda: {
            "code_commit": "tests.research",
            "protocol_commit": RESEARCH_PROTOCOL_COMMIT,
        },
    )
    monkeypatch.setattr(campaign, "_output_path", lambda p: p)
    monkeypatch.setattr(campaign, "ResearchMonitor", QuietMonitor)
    monkeypatch.setattr(campaign, "research_protocol", toy_protocol)
    monkeypatch.setattr(campaign, "sample_research_genome", toy_genome)
    monkeypatch.setattr(campaign, "evaluate_research_geometry", toy_evaluation)
    monkeypatch.setattr(campaign, "produce_research_offspring", toy_producer)
    monkeypatch.setattr(campaign, "square_reference", lambda: toy_genome().to_geometry())
    monkeypatch.setattr(campaign, "legal_edge_swaps", lambda _: ())
    monkeypatch.setattr(
        campaign,
        "BUILTIN_GEOMETRY_GENERATORS",
        SimpleNamespace(
            generate=lambda recipe, parameters, seed: toy_genome(seed or 0).to_geometry()
        ),
    )
    return tmp_path / "phase_10_research_test"


def test_user_preflight_full_and_completed_resume(
    toy_campaign: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = campaign.run_research_campaign(toy_campaign, mode="preflight")
    assert path.is_file()
    assert (path.parent / "hit_curves.png").is_file()
    preflight = load_record(path.parent / "analysis.json")
    assert preflight["total_evaluation_attempts"] == 129
    assert len(preflight["arms"]["evolution"]["trials"]) == 2
    assert not (toy_campaign / "full").exists()
    report = campaign.run_research_campaign(toy_campaign, mode="full")
    summary = load_record(report.parent / "analysis.json")
    assert summary["total_evaluation_attempts"] == 2083
    assert tuple(r["seed"] for r in summary["arms"]["random"]["trials"]) == TRIAL_SEEDS
    assert summary["complete_primary_comparison"]
    assert summary["exact_mcnemar_pvalue"] == 1.0
    assert summary["selected_candidates"] == {"evolution": [], "random": []}
    assert len(summary["mutation_statistics"]) == 32
    assert all(
        row["no_legal_swap_count"] == row["offspring_count"]
        for row in summary["mutation_statistics"]
    )
    monkeypatch.setattr(
        campaign,
        "evaluate_research_geometry",
        lambda *a, **k: pytest.fail("completed work repeated"),
    )
    assert campaign.run_research_campaign(toy_campaign, mode="resume") == report


def test_interrupted_pair_replays_same_inputs_and_retains_executions(
    toy_campaign: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def interrupt(geometry: Geometry, code: str, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 12:
            raise KeyboardInterrupt()
        return toy_evaluation(geometry, code, **kwargs)

    monkeypatch.setattr(campaign, "evaluate_research_geometry", interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_research_campaign(toy_campaign, mode="preflight")
    first = toy_campaign / "preflight/trial_00/execution_0000"
    assert len(list(first.glob("evaluation_*_outcome.json"))) == 10
    monkeypatch.setattr(campaign, "evaluate_research_geometry", toy_evaluation)
    report = campaign.run_research_campaign(toy_campaign, mode="resume")
    assert report.is_file()
    assert first.is_dir()
    assert (first.parent / "execution_0001").is_dir()
    restored = load_sealed(first.parent)
    assert restored.seed == PREFLIGHT_SEEDS[0]


def test_progress_event_and_no_implicit_full_mode(tmp_path: Path) -> None:
    from toposc_lab.cli import build_parser

    stream = StringIO()
    with ResearchMonitor(tmp_path / "events.jsonl", stream=stream) as monitor:
        monitor.emit(
            "evaluating",
            stage="preflight",
            completed=1,
            total=129,
            arm="evolution",
            generation=0,
            slot=1,
            last_sealed="references",
        )
    assert "1/129" in stream.getvalue()
    assert '"remaining":128' in (tmp_path / "events.jsonl").read_text()
    parser = build_parser()
    assert parser.parse_args(["phase-10-research", "--preflight"]).research_mode == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-research"])
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-research", "--preflight", "--full"])


def test_seed_collisions_do_not_redraw() -> None:
    registry = campaign.SeedRegistry()
    with pytest.raises(ResearchAbort, match="collision"):
        registry.claim(TRIAL_SEEDS[0], "some_other_role")


def test_deterministic_benchmark_rejects_decorative_seed() -> None:
    def misleading(genome: GeometryGenome, seed: None) -> Any:
        run = toy_evaluation(genome.to_geometry()).run
        assert run.reproducibility is not None
        return replace(run, reproducibility=replace(run.reproducibility, seed=991))

    with pytest.raises(ValueError, match="seed/code version"):
        run_search_benchmark(
            toy_protocol(),
            sampler=toy_genome,
            evaluator=misleading,
            offspring_producer=toy_producer,
        )


def test_callback_failures_remain_in_equal_budget() -> None:
    calls = 0

    def sometimes_failed(genome: GeometryGenome, _: None) -> Any:
        nonlocal calls
        calls += 1
        if calls in (1, 33):
            raise ArithmeticError("synthetic retained failure")
        return toy_evaluation(genome.to_geometry()).run

    result = run_search_benchmark(
        toy_protocol(),
        sampler=toy_genome,
        evaluator=sometimes_failed,
        offspring_producer=toy_producer,
    )
    assert calls == 64
    for arm in (result.trials[0].evolution_arm, result.trials[0].random_arm):
        assert len(arm.successes) == 32
        assert arm.successes[0] is False
        assert arm.fitness_history[0].members[0].failure.error_type == "ArithmeticError"
        assert arm.final_progress.available_count == 31


def test_partial_lexicographic_checkpoint_resumes_exactly(tmp_path: Path) -> None:
    from toposc_lab.search import run_generation_loop

    protocol = toy_protocol()
    initial = create_initial_population(
        [toy_genome(i) for i in range(8)], validity_policy=protocol.validity_policy
    )

    def evaluate(member: Any) -> Any:
        return toy_evaluation(member.genome.to_geometry()).run

    options = {
        "definition": protocol.definition,
        "evaluator": evaluate,
        "seed": 51,
        "offspring_producer": toy_producer,
        "producer_identifier": protocol.producer_identifier,
    }
    partial = run_generation_loop(
        initial, config=replace(protocol.generation_config, generation_count=1), **options
    )
    path = tmp_path / "partial.zip"
    save_search_checkpoint(
        path,
        create_search_checkpoint(
            partial,
            requested_generation_count=3,
            evaluator_identifier=DERIVATION_ID,
            code_version=protocol.code_version,
        ),
    )
    continued = resume_search(
        load_search_checkpoint(path),
        evaluator=evaluate,
        evaluator_identifier=DERIVATION_ID,
        offspring_producer=toy_producer,
        producer_identifier=protocol.producer_identifier,
        code_version=protocol.code_version,
    ).result
    original = run_generation_loop(initial, config=protocol.generation_config, **options)
    assert encode_record(continued) == encode_record(original)


def test_reference_operational_failure_stops_before_search(
    toy_campaign: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ArithmeticError("synthetic reference failure")

    monkeypatch.setattr(campaign, "evaluate_research_geometry", fail)
    with pytest.raises(RuntimeError, match="reference evaluation failed"):
        campaign.run_research_campaign(toy_campaign, mode="preflight")
    assert not (toy_campaign / "preflight/trial_00").exists()
    assert (
        load_record(toy_campaign / "preflight/references/execution_0000/reference_00_outcome.json")[
            "error"
        ]
        == "ArithmeticError"
    )


def test_full_requires_completed_preflight_without_creating_full_folder(toy_campaign: Path) -> None:
    toy_campaign.mkdir()
    with pytest.raises(RuntimeError, match="cannot load"):
        campaign.run_research_campaign(toy_campaign, mode="full")
    assert not (toy_campaign / "full").exists()
