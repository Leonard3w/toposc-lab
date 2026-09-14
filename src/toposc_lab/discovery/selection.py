"""Generate and select without accessing future exact labels."""

from dataclasses import asdict
from typing import Any, cast

from toposc_lab.active_learning.acquisition import AcquisitionConfig, acquire
from toposc_lab.active_learning.training import FittedSurrogate, TrainingConfig, retrain
from toposc_lab.data import ExactPhysicsDataset
from toposc_lab.discovery.config import DiscoveryConfig, derived_seed
from toposc_lab.generative.generators import Evidence, GeometryGenerator, Method
from toposc_lab.generative.physics import candidate
from toposc_lab.generative.space import NoveltyConfig
from toposc_lab.geometry import Geometry
from toposc_lab.ml import (
    FeatureOODDetector,
    HandcraftedFeatureSchema,
    RegressionTargetDefinition,
    RegressionTargetSource,
    extract_handcrafted_features,
)
from toposc_lab.ml.features import FeatureInput


def plan_cycle(
    config: DiscoveryConfig,
    cycle: int,
    history: tuple[Evidence, ...],
    excluded: tuple[Geometry, ...],
    fitted_cache: dict[int, FittedSurrogate] | None = None,
) -> tuple[tuple[Geometry, ...], dict[str, Any]]:
    # Identical random warm start for every supported generator.
    method = "random" if cycle == 0 else config.resolved_generator
    generator = GeometryGenerator(
        cast(Method, method),
        novelty=NoveltyConfig(config.minimum_distance),
        maximum_attempts=config.maximum_proposals,
    )
    geometries = generator.propose(
        history, seed=derived_seed(config.seed, cycle, 1), count=config.pool_size, excluded=excluded
    )
    proposals = tuple(candidate(g) for g in geometries)
    predictions: list[dict[str, Any]] = []
    training: dict[str, Any] | None = None
    selected = list(range(config.batch_size))
    strategies = {p.candidate_id: "generator_order" for p in proposals[: config.batch_size]}
    records = tuple(e.record for e in history)
    # The frozen reference contains only the first completed warm batch.
    common: dict[str, dict[str, Any]] = {
        p.candidate_id: {
            "is_ood": True,
            "reason": "reference_unavailable",
            "score": None,
            "reference_record_ids": [],
        }
        for p in proposals
    }
    if len(records) >= config.batch_size:
        warm = records[: config.batch_size]
        schema = HandcraftedFeatureSchema.fit(warm)
        matrix = extract_handcrafted_features(
            tuple(FeatureInput(p.candidate_id, p.geometry, p.model) for p in proposals),
            schema=schema,
        )
        assessment = (
            FeatureOODDetector(flag_extrapolation=True)
            .fit(extract_handcrafted_features(warm, schema=schema).values)
            .assess(matrix.values, record_ids=matrix.record_ids)
        )
        common = {
            p.candidate_id: {
                "is_ood": bool(assessment.is_ood[i]),
                "reason": "warm_reference",
                "score": float(assessment.scores[i]),
                "reference_record_ids": [r.record_id for r in warm],
            }
            for i, p in enumerate(proposals)
        }
    if config.surrogate and len(records) >= 4:
        # Cadence freezes a completed-cycle prefix; resuming rebuilds the same model.
        fit_cycle = 1 + ((cycle - 1) // config.retrain_every) * config.retrain_every
        fitted_records = tuple(
            e.record
            for e in history
            if int(cast(int, e.record.provenance.runtime["discovery_cycle"])) < fit_cycle
        )
        if len(fitted_records) >= 4:
            cache = fitted_cache if fitted_cache is not None else {}
            fitted = cache.get(fit_cycle)
            if fitted is None:
                fitted = retrain(
                    ExactPhysicsDataset(fitted_records),
                    config=TrainingConfig(
                        RegressionTargetDefinition(
                            "finite_geometry_quality",
                            RegressionTargetSource.OBSERVABLE,
                            "finite_geometry_quality",
                            "1",
                            "quality",
                        ),
                        flag_extrapolation=True,
                    ),
                    seed=derived_seed(config.seed, fit_cycle, 2),
                )
                cache[fit_cycle] = fitted
            predicted = fitted.predict(proposals)
            acquired = acquire(
                predicted,
                AcquisitionConfig(
                    batch_size=config.batch_size,
                    scaling="target_scale",
                    diversity_radius_fraction=0.5,
                ),
                feature_coordinates=fitted.feature_coordinates(proposals),
                reference_spacing=fitted.reference_spacing,
            )
            lookup = {p.candidate_id: i for i, p in enumerate(proposals)}
            selected = [lookup[a.candidate_id] for a in acquired]
            strategies = {a.candidate_id: a.strategy for a in acquired}
            predictions = [
                {**asdict(p), "kind": "predicted", "strategy": strategies.get(p.candidate_id)}
                for p in predicted
            ]
            training = {
                "fit_cycle": fit_cycle,
                "seed": fitted.seed,
                "record_ids": list(fitted.training_record_ids),
                "fingerprint": fitted.training_fingerprint,
                "calibration_claim": False,
            }
    if len(selected) != config.batch_size:
        raise ValueError("acquisition did not fill the frozen batch")
    return geometries, {
        "cycle": cycle,
        "method": method,
        "selected_indices": selected,
        "candidate_ids": [p.candidate_id for p in proposals],
        "strategies": strategies,
        "common_ood": common,
        "predictions": predictions,
        "training": training,
        "audit": {
            "attempted": generator.audit.attempted,
            "invalid": generator.audit.invalid,
            "duplicate": generator.audit.duplicate,
            "reasons": dict(generator.audit.reasons),
            "events": generator.audit.events,
            "predictions": generator.audit.predictions,
        },
        "stage": "before_exact",
        "history_record_ids": [r.record_id for r in records],
        "generator_update": "exact archive tournament"
        if method == "evolution"
        else "fixed nonlearning",
    }
