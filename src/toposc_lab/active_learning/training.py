"""Deterministic retraining on exact acquisition labels and a frozen holdout."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from toposc_lab.active_learning.pool import Candidate
from toposc_lab.active_learning.prediction import Prediction, predict_pool
from toposc_lab.data import DatasetRecord, ExactPhysicsDataset, validate_dataset
from toposc_lab.ml import (
    BootstrapUncertaintyRegressor,
    FeatureOODDetector,
    GradientBoostingRegressor,
    HandcraftedFeatureSchema,
    RegressionTargetDefinition,
    dataset_fingerprint,
    extract_handcrafted_features,
    extract_regression_targets,
)
from toposc_lab.ml.features import FeatureInput


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    target: RegressionTargetDefinition
    ensemble_size: int = 8
    boosting_estimators: int = 40
    distance_weight: float = 0.5
    flag_extrapolation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.flag_extrapolation, bool):
            raise TypeError("flag_extrapolation must be boolean")
        if (
            self.ensemble_size < 2
            or self.boosting_estimators < 1
            or not isfinite(self.distance_weight)
            or self.distance_weight < 0
        ):
            raise ValueError("invalid training configuration")


@dataclass(frozen=True, slots=True)
class FittedSurrogate:
    schema: HandcraftedFeatureSchema
    model: BootstrapUncertaintyRegressor
    ood: FeatureOODDetector
    training_fingerprint: str
    training_record_ids: tuple[str, ...]
    seed: int
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    reference_spacing: float = 0.0

    def feature_coordinates(self, pool: tuple[Candidate, ...]) -> dict[str, np.ndarray]:
        if self.feature_mean is None or self.feature_scale is None:
            raise ValueError("feature scaling requires a retrained surrogate")
        inputs = tuple(FeatureInput(c.candidate_id, c.geometry, c.model) for c in pool)
        matrix = extract_handcrafted_features(inputs, schema=self.schema).values
        standardized = (matrix - self.feature_mean) / self.feature_scale
        return {c.candidate_id: standardized[i] for i, c in enumerate(pool)}

    def predict(self, pool: tuple[Candidate, ...]) -> tuple[Prediction, ...]:
        return predict_pool(
            pool,
            schema=self.schema,
            model=self.model,
            ood=self.ood,
            training_fingerprint=self.training_fingerprint,
            training_record_ids=self.training_record_ids,
        )


def retrain(
    dataset: ExactPhysicsDataset,
    *,
    config: TrainingConfig,
    seed: int,
    reserved_records: tuple[DatasetRecord, ...] = (),
) -> FittedSurrogate:
    """Reserved validation/test families never fit features, targets or OOD thresholds.

    Callers may obtain the initial train/validation/test sets via the Phase-11
    split API. Acquisition grows only train; there is no re-splitting per cycle.
    """
    validate_dataset(dataset).raise_for_errors()
    families = {r.geometry.family_fingerprint for r in reserved_records}
    if any(r.geometry.family_fingerprint in families for r in dataset.records):
        raise ValueError("training overlaps a reserved validation/test geometry family")
    # Record IDs include timestamps/source provenance. They must not change the
    # bootstrap row order (and hence acquisition) on an independent rerun.
    records = tuple(
        sorted(dataset.records, key=lambda r: Candidate(r.geometry, r.model).candidate_id)
    )
    if len({Candidate(r.geometry, r.model).candidate_id for r in records}) != len(records):
        raise ValueError("duplicate exact simulations cannot be weighted as independent labels")
    if len(records) < 4:
        raise ValueError("retraining requires at least four exact labels (boosting min_leaf=2)")
    ordered = ExactPhysicsDataset(records)
    targets = extract_regression_targets(ordered, config.target).valid_values_by_id()
    if len(targets) != len(records):
        raise ValueError("training contains an unavailable/invalid exact target")
    schema = HandcraftedFeatureSchema.fit(records)
    features = extract_handcrafted_features(records, schema=schema).values
    model = BootstrapUncertaintyRegressor(
        model_factory=lambda: GradientBoostingRegressor(n_estimators=config.boosting_estimators),
        ensemble_size=config.ensemble_size,
        seed=seed,
        distance_weight=config.distance_weight,
    ).fit(features, [targets[r.record_id] for r in records])
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    distances = np.sqrt(np.mean((standardized[:, None, :] - standardized[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(np.min(distances, axis=1)))
    mean.setflags(write=False)
    scale.setflags(write=False)
    return FittedSurrogate(
        schema,
        model,
        FeatureOODDetector(flag_extrapolation=config.flag_extrapolation).fit(features),
        dataset_fingerprint(ordered),
        tuple(r.record_id for r in records),
        seed,
        mean,
        scale,
        spacing,
    )
