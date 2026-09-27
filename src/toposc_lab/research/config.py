"""Human-readable, versioned experiment configuration without mutable science."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, cast

QUESTION = (
    "Can connectivity optimization alone generate two-dimensional superconducting "
    "geometries with more disorder-robust topological and chiral Majorana edge "
    "signatures than matched conventional geometries?"
)


@dataclass
class ExperimentConfig:
    schema_version: int = 1
    name: str = "Autonomous Experiment 001"
    question: str = QUESTION
    algorithm: str = cast(str, None)
    seed: int = 17001
    cycles: int = 10
    exact_budget: int = 280
    candidate_budget: int = 10000
    pool_size: int = 40
    batch_size: int = 4
    checkpoint_every: int = 250
    retrain_every: int = 1
    wall_seconds: float = 3600.0
    concurrency: int = 1
    blas_threads: int = 1
    retry_limit: int = 1
    output_directory: str = "results/research"
    geometry_space: str = cast(str, None)
    surrogate_model: str = "bootstrap_graph"
    objective: str = cast(str, None)
    baselines: list[str] = field(default_factory=lambda: cast(list[str], None))
    space: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], None))
    search: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], None))
    physics: dict[str, Any] = field(default_factory=dict)
    surrogate: dict[str, Any] = field(default_factory=dict)
    studio: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version == 2:
            from toposc_lab.research import studio_physics  # noqa: F401
            from toposc_lab.research.studio_config import resolve_studio_defaults

            resolve_studio_defaults(self)
        else:
            if self.studio:
                raise ValueError("studio settings require configuration schema_version 2")
            self.studio = None
            if self.algorithm is None:
                self.algorithm = "surrogate_map_elites"
            if self.geometry_space is None:
                self.geometry_space = "fixed_connectivity"
            if self.objective is None:
                self.objective = "robustness_success_fraction"
            if self.baselines is None:
                self.baselines = ["regular", "random_rewired"]
            if self.space is None:
                self.space = {
                    "side": 10,
                    "min_degree": 2,
                    "max_degree": 6,
                    "max_bond_length": 2**0.5,
                    "bond_tolerance": 0.0,
                    "forbid_crossings": True,
                    "minimum_distance": 0.002,
                }
            if self.search is None:
                self.search = {
                    "behavior_descriptors": ["coordination_variance", "clustering_coefficient"],
                    "descriptor_bounds": [[0.0, 4.0], [0.0, 1.0]],
                    "archive_bins": [12, 12],
                    "allocation": {"exploitation": 0.6, "uncertainty": 0.2, "novelty": 0.2},
                    "mutation_rates": {
                        "remove_local_bond": 0.025,
                        "add_local_bond": 0.025,
                        "edge_swap": 0.15,
                        "local_rewiring": 0.35,
                        "diagonal_flip": 0.1,
                        "local_motif_replacement": 0.2,
                        "boundary_rewiring": 0.15,
                    },
                    "family_fraction": 0.5,
                }
        for name in (
            "schema_version",
            "seed",
            "cycles",
            "exact_budget",
            "candidate_budget",
            "pool_size",
            "batch_size",
            "checkpoint_every",
            "retrain_every",
            "concurrency",
            "blas_threads",
            "retry_limit",
        ):
            value = getattr(self, name)
            minimum = 0 if name in ("seed", "retry_limit") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.schema_version not in (1, 2):
            raise ValueError("unsupported configuration version")
        if self.batch_size > self.pool_size:
            raise ValueError("batch_size cannot exceed pool_size")
        if self.concurrency != 1:
            raise ValueError(
                "This deterministic adapter supports one exact worker per experiment; "
                "run independent experiments for parallel comparisons"
            )
        if (
            isinstance(self.wall_seconds, bool)
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0
        ):
            raise ValueError("wall_seconds must be finite and positive")
        for name in (
            "name",
            "question",
            "algorithm",
            "output_directory",
            "geometry_space",
            "surrogate_model",
            "objective",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("space", "search", "physics", "surrogate"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a JSON object")
            setattr(self, name, json.loads(json.dumps(value, allow_nan=False)))
        if not isinstance(self.baselines, (tuple, list)) or len(set(self.baselines)) != len(
            self.baselines
        ):
            raise ValueError("baselines must be distinct names")
        if any(name not in ("regular", "random_rewired") for name in self.baselines):
            raise ValueError(
                "Only regular and random_rewired baselines have matched fixed-site adapters"
            )
        # Materialize defaults before saving or rendering a configuration. The
        # objective is canonical at experiment level and is injected into the
        # selected versioned physical protocol only when constructing it.
        self.physics = asdict(self.physics_protocol())
        self.physics.pop("objective", None)
        from toposc_lab.research.surrogate import SURROGATE_REGISTRY

        if self.surrogate_model not in SURROGATE_REGISTRY:
            raise ValueError(f"unknown surrogate model: {self.surrogate_model}")
        defaults = {
            name: parameter.default
            for name, parameter in signature(
                SURROGATE_REGISTRY[self.surrogate_model]
            ).parameters.items()
            if parameter.default is not Parameter.empty and name != "seed"
        }
        self.surrogate = {**defaults, **self.surrogate}
        self.physics = json.loads(json.dumps(self.physics, allow_nan=False))
        self.surrogate = json.loads(json.dumps(self.surrogate, allow_nan=False))
        if self.schema_version == 2:
            from toposc_lab.research.studio_config import validate_studio

            validate_studio(self)

    def physics_protocol(self) -> Any:
        from toposc_lab.research.physics import create_physics_protocol

        return create_physics_protocol(self.physics, objective=self.objective)

    def validate_plugins(self) -> None:
        if self.schema_version == 2:
            from toposc_lab.research import studio_space  # noqa: F401
            from toposc_lab.research.studio_config import validate_studio

            validate_studio(self)
        from toposc_lab.research.space import SPACE_REGISTRY
        from toposc_lab.research.strategies import create_strategy
        from toposc_lab.research.surrogate import SURROGATE_REGISTRY

        if self.geometry_space not in SPACE_REGISTRY:
            raise ValueError("unknown geometry space")
        if self.surrogate_model not in SURROGATE_REGISTRY:
            raise ValueError("unknown surrogate model")
        self.physics_protocol()
        space = self.geometry_space_instance()
        SURROGATE_REGISTRY[self.surrogate_model](seed=self.seed, **self.surrogate)
        create_strategy(self.algorithm, space, self.seed, **self.search)

    def geometry_space_instance(self) -> Any:
        """Inject the single canonical domain into existing geometry adapters."""
        from toposc_lab.research.space import SPACE_REGISTRY

        if self.schema_version == 2:
            from toposc_lab.research import studio_space  # noqa: F401

        settings = dict(self.space)
        if self.schema_version == 2 and self.geometry_space in {
            "embedded_sampling",
            "fixed_candidates",
        }:
            settings["domain"] = self.physics["domain"]
        return SPACE_REGISTRY[self.geometry_space](**settings)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.schema_version == 1:
            # Historical config hashes cover the exact v1 serialized contract.
            result.pop("studio")
        return result

    @property
    def fingerprint(self) -> str:
        return sha256(
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        return cls(**payload)


def default_config() -> dict[str, Any]:
    return ExperimentConfig().to_dict()
