"""One bounded generate/predict/select/exact/validate/store/retrain cycle."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from toposc_lab.active_learning.acquisition import Acquisition, AcquisitionConfig, acquire
from toposc_lab.active_learning.exact import ExactEvaluator, Verification, verify_exact
from toposc_lab.active_learning.pool import Candidate, CandidateSpace, generate_candidate_pool
from toposc_lab.active_learning.prediction import Prediction
from toposc_lab.active_learning.training import FittedSurrogate, TrainingConfig, retrain
from toposc_lab.data import DatasetRecord, ExactPhysicsDataset, save_dataset
from toposc_lab.ml import extract_regression_targets


def derived_seed(seed: int, cycle: int, stage: int, index: int = 0) -> int:
    return int(np.random.SeedSequence([seed, cycle, stage, index]).generate_state(1)[0])


@dataclass(frozen=True, slots=True)
class CycleConfig:
    training: TrainingConfig
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    pool_size: int = 64

    def __post_init__(self) -> None:
        if self.pool_size < self.acquisition.batch_size:
            raise ValueError("pool must be at least as large as acquisition batch")


@dataclass(frozen=True, slots=True)
class CycleResult:
    cycle_index: int
    pool_ids: tuple[str, ...]
    predictions: tuple[Prediction, ...]
    selected: tuple[Acquisition, ...]
    verifications: tuple[Verification, ...]
    dataset: ExactPhysicsDataset
    surrogate: FittedSurrogate
    prediction_failure: str | None = None

    @property
    def exact_attempts(self) -> int:
        return sum(v.exact_attempts for v in self.verifications)


def run_cycle(
    space: CandidateSpace,
    dataset: ExactPhysicsDataset,
    *,
    config: CycleConfig,
    seed: int,
    cycle_index: int,
    evaluator: ExactEvaluator,
    reserved_records: tuple[DatasetRecord, ...] = (),
    attempted_ids: frozenset[str] = frozenset(),
    output_path: str | Path | None = None,
) -> CycleResult:
    known = frozenset(Candidate(r.geometry, r.model).candidate_id for r in dataset.records)
    pool = generate_candidate_pool(
        space,
        count=config.pool_size,
        seed=derived_seed(seed, cycle_index, 0),
        excluded_ids=known | attempted_ids,
        excluded_families=frozenset(r.geometry.family_fingerprint for r in reserved_records),
    )
    surrogate = retrain(
        dataset,
        config=config.training,
        seed=derived_seed(seed, cycle_index, 1),
        reserved_records=reserved_records,
    )
    prediction_failure = None
    try:
        predictions = surrogate.predict(pool)
        selected = acquire(
            predictions,
            config.acquisition,
            feature_coordinates=surrogate.feature_coordinates(pool)
            if config.acquisition.diversity_radius_fraction
            else None,
            reference_spacing=surrogate.reference_spacing,
        )
    except (ValueError, FloatingPointError) as error:
        # The pool is already seeded and valid. Explicit random exploration is
        # a safe fallback when numerical predictions cannot be trusted.
        prediction_failure = f"{type(error).__name__}: {error}"
        predictions = ()
        selected = tuple(
            Acquisition(c.candidate_id, "fallback_exploration")
            for c in pool[: config.acquisition.batch_size]
        )
    lookup = {c.candidate_id: c for c in pool}
    verifications = []
    new_records = []
    for index, selection in enumerate(selected):
        result = verify_exact(
            lookup[selection.candidate_id],
            seed=derived_seed(seed, cycle_index, 2, index),
            evaluator=evaluator,
        )
        if result.record is not None:
            targets = extract_regression_targets(
                ExactPhysicsDataset((result.record,)), config.training.target
            ).valid_values_by_id()
            if result.record.record_id not in targets:
                result = replace(result, record=None, failure="unavailable exact training target")
            else:
                new_records.append(result.record)
        verifications.append(result)
    updated = ExactPhysicsDataset((*dataset.records, *new_records))
    # Retrain before publishing: a training failure cannot expose a partial cycle.
    fitted = retrain(
        updated,
        config=config.training,
        seed=derived_seed(seed, cycle_index + 1, 1),
        reserved_records=reserved_records,
    )
    if output_path is not None:
        save_dataset(output_path, updated, overwrite=Path(output_path).exists())
    return CycleResult(
        cycle_index,
        tuple(c.candidate_id for c in pool),
        predictions,
        selected,
        tuple(verifications),
        updated,
        fitted,
        prediction_failure,
    )
