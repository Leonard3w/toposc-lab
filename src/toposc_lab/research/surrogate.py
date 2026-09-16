"""Graph-descriptor bootstrap surrogate using the validated Phase-12/13 stack.

The minimal GNN exists but failed its adoption gate. This adapter therefore uses
the established bootstrap gradient-boosting regressors and feature OOD detector.
Prequential errors compare saved predictions to labels obtained later; training
residuals are never presented as held-out calibration.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, TypeGuard

import numpy as np

from toposc_lab.ml.models import GradientBoostingRegressor
from toposc_lab.ml.ood import FeatureOODDetector
from toposc_lab.ml.uncertainty import BootstrapUncertaintyRegressor

SURROGATE_REGISTRY: dict[str, type] = {}


class GraphSurrogate:
    def __init__(self, *, ensemble_size: int = 8, n_estimators: int = 40,
                 seed: int = 0, distance_weight: float = 0.5) -> None:
        if ensemble_size < 2 or n_estimators < 1:
            raise ValueError("surrogate needs >=2 ensemble members and >=1 estimator")
        if not np.isfinite(distance_weight) or distance_weight < 0:
            raise ValueError("distance weight must be finite and nonnegative")
        self.ensemble_size, self.n_estimators, self.seed = ensemble_size, n_estimators, seed
        self.distance_weight = distance_weight
        self.model_version = 0
        self.feature_names: tuple[str, ...] = ()
        self.models: dict[str, BootstrapUncertaintyRegressor] = {}
        self.ood: FeatureOODDetector | None = None
        self.training: list[dict[str, Any]] = []
        self.calibration: list[dict[str, Any]] = []
        self.training_fingerprint: str | None = None

    @property
    def fitted(self) -> bool:
        return bool(self.models)

    def fit(self, candidates: list[dict[str, Any]]) -> GraphSurrogate:
        records = []
        for candidate in candidates:
            if candidate.get("origin") != "exact" or candidate.get("validation_state") not in {
                "EXACT_EVALUATED", "TOPOLOGY_VALIDATED", "MAJORANA_VALIDATED",
                "ROBUSTNESS_VALIDATED", "FINITE_SIZE_VALIDATED",
            }:
                continue
            score = candidate.get("score")
            if not _finite(score):
                continue
            record = {"id": candidate["id"], "descriptors": candidate["descriptors"],
                      "score": float(score), "origin": "exact",
                      "validation_state": candidate.get("validation_state", "EXACT_EVALUATED")}
            raw = candidate.get("raw_metrics", candidate.get("metrics", {}))
            if not isinstance(raw, dict):
                raw = {}
            for name in ("bulk_gap", "edge_localization", "topology_index",
                         "validation_probability", "finite_geometry_quality", "quality",
                         "localizer_gap", "boundary_weight", "minimum_abs_energy",
                         "robustness_success_fraction", "robustness_quality_mean"):
                value = candidate.get(name, raw.get(name))
                if name == "topology_index" and candidate.get("validation_state") == "EXACT_EVALUATED":
                    continue
                if _finite(value):
                    record[name] = float(value)
            prediction = candidate.get("surrogate_prediction", candidate.get("prediction"))
            if prediction:
                self.observe(prediction, float(score))
            records.append(record)
        records.sort(key=lambda record: record["id"])
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("duplicate exact labels cannot train a surrogate")
        if len(records) < 4:
            raise ValueError("surrogate training requires four valid exact labels")
        names = sorted(set.intersection(*(set(r["descriptors"]) for r in records)))
        if not names:
            raise ValueError("surrogate requires shared graph descriptors")
        features = np.array([[r["descriptors"][name] for name in names] for r in records], float)
        if not np.all(np.isfinite(features)):
            raise ValueError("surrogate features must be finite")
        models = {}
        targets = ["score"] + [name for name in records[0] if name not in {
            "score", "id", "descriptors", "origin", "validation_state"
        } and all(name in r for r in records)]
        for name in targets:
            models[name] = BootstrapUncertaintyRegressor(
                model_factory=lambda: GradientBoostingRegressor(n_estimators=self.n_estimators),
                ensemble_size=self.ensemble_size, seed=self.seed,
                distance_weight=self.distance_weight,
            ).fit(features, [r[name] for r in records])
        self.ood = FeatureOODDetector(flag_extrapolation=True).fit(features)
        self.feature_names = tuple(names)
        self.models = models
        self.training = copy.deepcopy(records)
        self.training_fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True,
            allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        self.model_version += 1
        return self

    def predict(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.fitted or self.ood is None:
            raise RuntimeError("surrogate has no fitted exact-label model")
        if not candidates:
            return []
        features = np.array([[r["descriptors"][name] for name in self.feature_names]
                             for r in candidates], float)
        ids = tuple(r["id"] for r in candidates)
        predictions = {name: model.predict_with_uncertainty(features)
                       for name, model in self.models.items()}
        assessment = self.ood.assess(features, record_ids=ids)
        training_ids = [r["id"] for r in self.training]
        results = []
        for index, identifier in enumerate(ids):
            score = predictions["score"]
            results.append({
                "candidate_id": identifier, "origin": "surrogate",
                "score": float(score.mean[index]),
                "uncertainty": float(score.standard_deviation[index]),
                "ensemble_uncertainty": float(score.ensemble_standard_deviation[index]),
                "nearest_training_distance": float(score.nearest_training_distance[index]),
                "ood": bool(assessment.is_ood[index]), "is_ood": bool(assessment.is_ood[index]),
                "ood_score": float(assessment.scores[index]),
                "model_version": self.model_version, "training_ids": training_ids,
                "training_fingerprint": self.training_fingerprint,
                "targets": {name: {"mean": float(p.mean[index]),
                                    "uncertainty": float(p.standard_deviation[index])}
                            for name, p in predictions.items()},
                "calibrated_interval": False,
                "warning": "SURROGATE PREDICTION: exact validation required",
            })
        return results

    def observe(self, prediction: dict[str, Any], score: float | dict[str, Any]) -> None:
        value: Any = score
        if isinstance(score, dict):
            if score.get("origin") != "exact" or score.get("validation_state") in {"FAILED", "REJECTED"}:
                return
            value = score.get("score")
        identifier = prediction.get("candidate_id")
        if not identifier or identifier in prediction.get("training_ids", []):
            return
        if any(r["candidate_id"] == identifier for r in self.calibration):
            return
        if prediction.get("origin") != "surrogate" or not _finite(value):
            return
        mean, uncertainty = prediction.get("score"), prediction.get("uncertainty")
        if not _finite(mean) or not _finite(uncertainty) or uncertainty < 0:
            return
        self.calibration.append({"candidate_id": identifier, "truth": float(value),
            "prediction": mean, "uncertainty": uncertainty,
            "absolute_error": abs(float(value) - mean),
            "covered_2sigma": abs(float(value) - mean) <= 2 * uncertainty,
            "model_version": prediction.get("model_version"), "origin": "prequential"})

    def summarize(self) -> dict[str, Any]:
        errors = np.array([r["absolute_error"] for r in self.calibration], float)
        return {"model": "bootstrap_graph", "estimator": "GradientBoostingRegressor",
                "fitted": self.fitted, "model_version": self.model_version,
                "training_count": len(self.training), "feature_count": len(self.feature_names),
                "training_fingerprint": self.training_fingerprint,
                "targets": list(self.models), "calibration_count": len(errors),
                "mae": float(errors.mean()) if len(errors) else None,
                "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                "coverage_2sigma": float(np.mean([r["covered_2sigma"] for r in self.calibration]))
                    if len(errors) else None,
                "calibration_method": "prequential; heuristic two-sigma coverage, not calibrated intervals",
                "calibration_claim": False, "ensemble_size": self.ensemble_size}

    def checkpoint(self) -> dict[str, Any]:
        return copy.deepcopy({"version": 1, "model": "bootstrap_graph",
            "config": {"ensemble_size": self.ensemble_size, "n_estimators": self.n_estimators,
                       "seed": self.seed, "distance_weight": self.distance_weight},
            "training": self.training, "calibration": self.calibration,
            "model_version": self.model_version, "training_fingerprint": self.training_fingerprint})

    def resume(self, checkpoint: dict[str, Any]) -> GraphSurrogate:
        if checkpoint.get("version") != 1 or checkpoint.get("model") != "bootstrap_graph":
            raise ValueError("incompatible surrogate checkpoint")
        GraphSurrogate.__init__(self, **checkpoint["config"])
        if checkpoint["training"]:
            self.fit(checkpoint["training"])
            if self.training_fingerprint != checkpoint["training_fingerprint"]:
                raise ValueError("corrupt surrogate training fingerprint")
        self.calibration = copy.deepcopy(checkpoint["calibration"])
        self.model_version = checkpoint["model_version"]
        return self


def _finite(value: Any) -> TypeGuard[float | int]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


SURROGATE_REGISTRY["bootstrap_graph"] = GraphSurrogate
