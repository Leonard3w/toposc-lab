"""Versioned discovery contracts; defaults carry only their validated scope."""

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256

from toposc_lab.generative.physics import PROTOCOL_ID


def default_generator(stratum: str) -> str:
    """A policy decision, not evidence that an unvalidated adapter is supported."""
    return "patch" if stratum == PROTOCOL_ID else "evolution"


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    schema_version: int = 1
    stratum: str = PROTOCOL_ID
    generator: str = "auto"
    seed: int = 15101
    cycles: int = 3
    batch_size: int = 4
    pool_size: int = 8
    maximum_proposals: int = 1000
    minimum_distance: float = 0.06
    surrogate: bool = False
    retrain_every: int = 2
    disorder_samples: int = 4
    onsite_width: float = 0.2
    retry_reserve: int = 8
    plots: bool = True
    # Frozen scientific contract, serialized even though v1 rejects changes.
    model: str = "chiral_p_wave"
    hopping: float = 1.0
    pairing: float = 1.0
    chemical_potential: float = 2.0
    chirality: int = 1
    solver: str = "numpy.linalg.eigh"
    basis: str = "component_major"
    tolerance: float = 1e-10
    success_threshold: float = 0.20
    kappas: tuple[float, ...] = (0.1, 0.2, 0.3)
    probe: tuple[float, float] = (2.5, 2.5)

    def __post_init__(self) -> None:
        # JSON arrays must not leave mutable paths around the frozen contract.
        object.__setattr__(self, "kappas", tuple(self.kappas))
        object.__setattr__(self, "probe", tuple(self.probe))
        if type(self.schema_version) is not int or type(self.chirality) is not int:
            raise ValueError("schema version and chirality must be integers")
        for name in (
            "hopping",
            "pairing",
            "chemical_potential",
            "tolerance",
            "success_threshold",
            "minimum_distance",
            "onsite_width",
        ):
            if isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} cannot be boolean")
        for name in (
            "seed",
            "cycles",
            "batch_size",
            "pool_size",
            "maximum_proposals",
            "retrain_every",
            "disorder_samples",
            "retry_reserve",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("seed", "retry_reserve") else 1):
                raise ValueError(f"invalid {name}")
        if self.batch_size < 4 or self.pool_size < self.batch_size or self.disorder_samples < 2:
            raise ValueError("require batch >=4, pool >=batch and ensemble >=2")
        for name in ("plots", "surrogate"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.generator not in ("auto", "patch", "random", "evolution", "coverage"):
            raise ValueError("unsupported generator; RL/learned models are not justified")
        if self.stratum != PROTOCOL_ID:
            raise ValueError(
                "unvalidated stratum requires a new exact adapter and baseline benchmark; "
                "policy default is Evolution, with Random retained"
            )
        if not math.isfinite(self.minimum_distance) or not 0.06 <= self.minimum_distance < 1:
            raise ValueError("minimum distance cannot weaken the frozen separation rule")
        if not math.isfinite(self.onsite_width) or self.onsite_width < 0:
            raise ValueError("onsite width must be finite and nonnegative")
        fixed = (
            self.schema_version,
            self.model,
            self.hopping,
            self.pairing,
            self.chemical_potential,
            self.chirality,
            self.solver,
            self.basis,
            self.tolerance,
            self.success_threshold,
            tuple(self.kappas),
            tuple(self.probe),
        )
        if fixed != (
            1,
            "chiral_p_wave",
            1.0,
            1.0,
            2.0,
            1,
            "numpy.linalg.eigh",
            "component_major",
            1e-10,
            0.20,
            (0.1, 0.2, 0.3),
            (2.5, 2.5),
        ):
            raise ValueError("v1 scientific definitions are frozen; use a versioned adapter")

    @property
    def resolved_generator(self) -> str:
        return default_generator(self.stratum) if self.generator == "auto" else self.generator

    @property
    def exact_attempt_cap(self) -> int:
        return self.cycles * self.batch_size * (2 + self.disorder_samples) + self.retry_reserve

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def derived_seed(seed: int, *coordinates: int) -> int:
    import numpy as np

    return int(np.random.SeedSequence([seed, *coordinates]).generate_state(1)[0])
