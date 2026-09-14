"""Batched proposals retain training provenance and uncertainty/OOD warnings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.active_learning.pool import Candidate
from toposc_lab.ml import BootstrapUncertaintyRegressor, FeatureOODDetector
from toposc_lab.ml.features import (
    FeatureInput,
    HandcraftedFeatureSchema,
    extract_handcrafted_features,
)


@dataclass(frozen=True, slots=True)
class Prediction:
    candidate_id: str
    value: float
    uncertainty: float
    novelty: float
    is_ood: bool
    ood_score: float
    training_fingerprint: str
    training_record_ids: tuple[str, ...]
    dataset_schema_version: int = 1
    valid: bool = True
    kind: str = "predicted"

    def __post_init__(self) -> None:
        if self.kind != "predicted":
            raise ValueError("surrogate results must remain predicted")
        if self.valid and (
            not np.all(np.isfinite([self.value, self.uncertainty, self.novelty, self.ood_score]))
            or min(self.uncertainty, self.novelty, self.ood_score) < 0
        ):
            raise ValueError("valid predictions require finite values and nonnegative uncertainty")


def predict_pool(
    pool: tuple[Candidate, ...],
    *,
    schema: HandcraftedFeatureSchema,
    model: BootstrapUncertaintyRegressor,
    ood: FeatureOODDetector,
    training_fingerprint: str,
    training_record_ids: tuple[str, ...],
) -> tuple[Prediction, ...]:
    if not pool:
        return ()
    inputs = tuple(FeatureInput(c.candidate_id, c.geometry, c.model) for c in pool)
    features = extract_handcrafted_features(inputs, schema=schema)
    values = model.predict_with_uncertainty(features.values)
    assessment = ood.assess(features.values, record_ids=features.record_ids)
    return tuple(
        Prediction(
            c.candidate_id,
            float(values.mean[i]),
            float(values.standard_deviation[i]),
            float(values.nearest_training_distance[i]),
            bool(assessment.is_ood[i]),
            float(assessment.scores[i]),
            training_fingerprint,
            training_record_ids,
        )
        for i, c in enumerate(pool)
    )
