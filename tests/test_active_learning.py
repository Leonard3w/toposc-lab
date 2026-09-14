from __future__ import annotations

import pytest

from toposc_lab.active_learning import Candidate, CandidateSpace, generate_candidate_pool
from toposc_lab.data import GeometryRecord, ModelParametersRecord
from toposc_lab.geometry import chain


def exact_fixture(n=6, mu=0.0, seed=1):
    from toposc_lab.data.dataset_benchmark import _exact_kitaev_record

    return _exact_kitaev_record(n_sites=n, chemical_potential=mu, seed=seed)


def test_exact_verification_positive_negative_and_wrong_identity():
    from toposc_lab.active_learning.exact import verify_exact

    for mu, expected in ((0.0, True), (3.0, False)):
        record = exact_fixture(mu=mu, seed=17)
        proposal = Candidate(record.geometry, record.model)
        calls = []

        def evaluate(c, seed, mu=mu, calls=calls):
            calls.append((c.candidate_id, seed))
            return exact_fixture(mu=mu, seed=seed)

        verified = verify_exact(proposal, seed=17, evaluator=evaluate)
        assert calls == [(proposal.candidate_id, 17)]
        assert verified.record.topology[0].is_topological == expected
        assert verified.failure is None
        assert verify_exact(proposal, seed=18, evaluator=lambda c, s, r=record: r).record is None
        assert verify_exact(candidate(9), seed=17, evaluator=evaluate).record is None
    failed = verify_exact(candidate(), seed=1, evaluator=lambda c, s: 1 / 0)
    assert failed.record is None and "ZeroDivisionError" in failed.failure


def test_atomic_append_rejects_failures_and_duplicate_physics(tmp_path):
    from toposc_lab.active_learning.exact import verify_exact
    from toposc_lab.active_learning.storage import append_verified
    from toposc_lab.data import load_dataset

    record = exact_fixture()
    c = Candidate(record.geometry, record.model)
    path = tmp_path / "labels.json"
    first = verify_exact(c, seed=1, evaluator=lambda c, s: exact_fixture(seed=s))
    assert append_verified(path, first)
    before = path.read_bytes()
    second = verify_exact(c, seed=2, evaluator=lambda c, s: exact_fixture(seed=s))
    assert not append_verified(path, second)
    failed = verify_exact(c, seed=3, evaluator=lambda c, s: None)
    with pytest.raises(ValueError):
        append_verified(path, failed)
    assert path.read_bytes() == before
    assert len(load_dataset(path).records) == 1


def training_config():
    from toposc_lab.active_learning.training import TrainingConfig
    from toposc_lab.ml import RegressionTargetDefinition, RegressionTargetSource

    return TrainingConfig(
        RegressionTargetDefinition(
            name="finite_energy",
            source=RegressionTargetSource.OBSERVABLE,
            result_kind="minimum_absolute_energy",
            result_version="1",
            value_key="gap",
        ),
        ensemble_size=3,
        boosting_estimators=5,
    )


def test_retraining_reproducible_and_holdout_families_frozen():
    from toposc_lab.active_learning.training import retrain
    from toposc_lab.data import ExactPhysicsDataset

    records = tuple(exact_fixture(n=n) for n in (6, 8, 10, 11))
    heldout = (exact_fixture(n=12),)
    pool = tuple(Candidate(r.geometry, r.model) for r in heldout)
    first = retrain(
        ExactPhysicsDataset(records), config=training_config(), seed=1, reserved_records=heldout
    )
    second = retrain(
        ExactPhysicsDataset(tuple(reversed(records))),
        config=training_config(),
        seed=1,
        reserved_records=heldout,
    )
    assert first.predict(pool) == second.predict(pool)
    assert heldout[0].record_id not in first.training_record_ids
    with pytest.raises(ValueError, match="reserved"):
        retrain(
            ExactPhysicsDataset((*records, exact_fixture(n=12, mu=3))),
            config=training_config(),
            seed=1,
            reserved_records=heldout,
        )


def cycle_fixture():
    from toposc_lab.data import ExactPhysicsDataset

    records = tuple(exact_fixture(n=n) for n in range(6, 15))
    space = CandidateSpace(tuple(Candidate(r.geometry, r.model) for r in records))

    def evaluate(c, seed):
        return exact_fixture(n=c.geometry.to_geometry().n_sites, seed=seed)

    return space, ExactPhysicsDataset(records[:4]), evaluate


def test_cycle_runs_exact_stores_and_retrains_reproducibly(tmp_path):
    from toposc_lab.active_learning.acquisition import AcquisitionConfig
    from toposc_lab.active_learning.cycle import CycleConfig, run_cycle
    from toposc_lab.data import dataset_to_bytes, load_dataset

    space, dataset, evaluate = cycle_fixture()
    config = CycleConfig(training_config(), AcquisitionConfig(batch_size=2), pool_size=5)
    first = run_cycle(
        space,
        dataset,
        config=config,
        seed=13,
        cycle_index=0,
        evaluator=evaluate,
        output_path=tmp_path / "labels.json",
    )
    second = run_cycle(space, dataset, config=config, seed=13, cycle_index=0, evaluator=evaluate)
    assert first.exact_attempts == 2
    assert len(first.dataset.records) == 6
    assert first.predictions == second.predictions
    assert first.selected == second.selected
    assert dataset_to_bytes(first.dataset) == dataset_to_bytes(second.dataset)
    assert dataset_to_bytes(load_dataset(tmp_path / "labels.json")) == dataset_to_bytes(
        first.dataset
    )
    assert set(first.surrogate.training_record_ids) == {r.record_id for r in first.dataset.records}
    assert all(v.record is not None for v in first.verifications)


def campaign_config():
    from toposc_lab.active_learning.acquisition import AcquisitionConfig
    from toposc_lab.active_learning.campaign import CampaignConfig
    from toposc_lab.active_learning.cycle import CycleConfig

    return CampaignConfig(
        CycleConfig(training_config(), AcquisitionConfig(batch_size=2), 5),
        seed=13,
        cycles=2,
        exact_budget=4,
        evaluator_identifier="test-exact-v1",
        code_version="test-v1",
    )


def test_campaign_resume_preserves_labels_seeds_and_budget(tmp_path):
    from dataclasses import replace

    from toposc_lab.active_learning.campaign import run_campaign
    from toposc_lab.data import dataset_to_bytes

    space, dataset, evaluate = cycle_fixture()
    config = campaign_config()
    calls = []

    def counted(c, s):
        calls.append((c.candidate_id, s))
        return evaluate(c, s)

    options = {"config": config, "evaluator": counted}
    first = run_campaign(
        space, dataset, **options, checkpoint_path=tmp_path / "resume.json", stop_after=1
    )
    assert first.exact_attempts == 2
    resumed = run_campaign(space, dataset, **options, checkpoint_path=tmp_path / "resume.json")
    assert len(calls) == resumed.exact_attempts == 4
    uninterrupted = run_campaign(space, dataset, **options, checkpoint_path=tmp_path / "full.json")
    assert dataset_to_bytes(resumed.dataset) == dataset_to_bytes(uninterrupted.dataset)
    assert resumed.history == uninterrupted.history
    with pytest.raises(ValueError, match="contract mismatch"):
        run_campaign(
            space,
            dataset,
            config=replace(config, seed=14),
            evaluator=evaluate,
            checkpoint_path=tmp_path / "resume.json",
        )


def candidate(n=6, mu=0.0):
    return Candidate(
        GeometryRecord.from_geometry(chain(n)),
        ModelParametersRecord("GeometryKitaevChain", "1", {"n_sites": n, "chemical_potential": mu}),
    )


def test_pool_reproducibility_constraints_and_exclusion():
    space = CandidateSpace(tuple(candidate(n) for n in range(6, 14)))
    first = generate_candidate_pool(space, count=4, seed=131)
    assert first == generate_candidate_pool(space, count=4, seed=131)
    assert len({c.candidate_id for c in first}) == 4
    remaining = generate_candidate_pool(
        space, count=100, seed=131, excluded_ids=frozenset(c.candidate_id for c in first)
    )
    assert len(remaining) == 4
    assert not {c.candidate_id for c in first} & {c.candidate_id for c in remaining}
    with pytest.raises(ValueError, match="duplicate"):
        CandidateSpace((candidate(), candidate()))
    with pytest.raises(ValueError, match="constraints"):
        CandidateSpace((candidate(),), maximum_sites=5)


def test_prediction_uses_unlabelled_inputs_and_links_training():
    from toposc_lab.active_learning.prediction import predict_pool
    from toposc_lab.data.dataset_benchmark import _exact_kitaev_record
    from toposc_lab.ml import (
        BootstrapUncertaintyRegressor,
        FeatureOODDetector,
        HandcraftedFeatureSchema,
        extract_handcrafted_features,
    )

    records = tuple(
        _exact_kitaev_record(n_sites=n, chemical_potential=0.0, seed=n) for n in (6, 8, 10)
    )
    schema = HandcraftedFeatureSchema.fit(records)
    x = extract_handcrafted_features(records, schema=schema).values
    model = BootstrapUncertaintyRegressor(ensemble_size=3).fit(x, [1.0, 2.0, 3.0])
    pool = tuple(Candidate(r.geometry, r.model) for r in records)
    predictions = predict_pool(
        pool,
        schema=schema,
        model=model,
        ood=FeatureOODDetector().fit(x),
        training_fingerprint="train-v1",
        training_record_ids=tuple(r.record_id for r in records),
    )
    assert len(predictions) == 3
    assert all(p.kind == "predicted" and p.uncertainty >= 0 for p in predictions)
    assert all(p.training_fingerprint == "train-v1" for p in predictions)
    assert [p.candidate_id for p in predictions] == [c.candidate_id for c in pool]


def prediction(identifier, value=1.0, uncertainty=1.0, novelty=1.0, ood=False, valid=True):
    from toposc_lab.active_learning.prediction import Prediction

    return Prediction(
        identifier, value, uncertainty, novelty, ood, float(ood), "train", (), valid=valid
    )


def test_exploit_ties_invalid_and_ood():
    from toposc_lab.active_learning.acquisition import exploit

    predictions = (
        prediction("b"),
        prediction("a"),
        prediction("ood", value=1e9, ood=True),
        prediction("invalid", value=1e10, valid=False),
    )
    assert exploit(predictions, 4) == ("a", "b")
    assert exploit(tuple(reversed(predictions)), 1) == ("a",)


def test_explore_balances_signals_and_permits_ood():
    from toposc_lab.active_learning.acquisition import explore

    rows = (
        prediction("noise", uncertainty=1e99, novelty=0),
        prediction("informative", uncertainty=5, novelty=5, ood=True),
        prediction("known", uncertainty=0, novelty=0),
    )
    assert explore(rows, 2) == ("informative", "noise")
    assert explore(tuple(reversed(rows)), 2) == explore(rows, 2)
    with pytest.raises(ValueError):
        explore(rows, 1, uncertainty_weight=0, novelty_weight=0)


def test_acquisition_reserves_both_strategies_and_handles_all_ood():
    from toposc_lab.active_learning.acquisition import AcquisitionConfig, acquire

    config = AcquisitionConfig(batch_size=2)
    rows = (
        prediction("best", value=10, uncertainty=0, novelty=0),
        prediction("new", value=0, uncertainty=10, novelty=10, ood=True),
    )
    selected = acquire(rows, config)
    assert [(s.candidate_id, s.strategy) for s in selected] == [
        ("best", "exploitation"),
        ("new", "exploration"),
    ]
    assert acquire((rows[1],), config)[0].strategy == "exploration"
    assert acquire((prediction("bad", valid=False),), config) == ()
    with pytest.raises(ValueError, match="duplicate"):
        acquire((rows[0], rows[0]), config)
