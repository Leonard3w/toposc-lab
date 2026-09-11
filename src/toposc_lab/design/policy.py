"""Small shared action-scoring network and an interchangeable random baseline.

REINFORCE uses one terminal return, a baseline from PREVIOUS episodes, and
the sum of log-policy gradients along the trajectory. No physics surrogate.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .environment import FEATURE_COUNT

Array = NDArray[np.float64]


class Policy(Protocol):
    def choose(self, features: Array, rng: np.random.Generator) -> int: ...

    def finish(self, reward: float | None) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


class RandomPolicy:
    """Uniform over the supplied feasible actions, not over final geometries."""

    def choose(self, features: Array, rng: np.random.Generator) -> int:
        return int(rng.integers(len(features)))

    def finish(self, reward: float | None) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {"kind": "random_v1"}


class NeuralPolicy:
    """One tanh hidden layer, shared across a variable number of legal actions.

    No autograd dependency: analytic derivatives are covered by finite differences.
    Pending gradients are bounded in model size, not trajectory length.
    """

    def __init__(self, *, seed: int, hidden: int = 16, learning_rate: float = 0.01) -> None:
        if type(hidden) is not int or hidden < 1:
            raise ValueError("hidden must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        rng = np.random.Generator(np.random.PCG64(seed))
        self.weights = rng.normal(0, 0.15, (FEATURE_COUNT, hidden))
        self.bias = np.zeros(hidden)
        self.output = rng.normal(0, 0.15, hidden)
        self.learning_rate = learning_rate
        self.baseline = 0.0
        self.updates = 0
        self._reset_gradient()

    def _reset_gradient(self) -> None:
        self._dw = np.zeros_like(self.weights)
        self._db = np.zeros_like(self.bias)
        self._dv = np.zeros_like(self.output)

    def probabilities(self, features: Array) -> Array:
        if (
            features.ndim != 2
            or features.shape[1] != FEATURE_COUNT
            or len(features) == 0
            or not np.isfinite(features).all()
        ):
            raise ValueError("invalid action features")
        logits = np.tanh(features @ self.weights + self.bias) @ self.output
        weights = np.exp(logits - np.max(logits))
        return np.asarray(weights / weights.sum(), dtype=float)

    def log_gradient(self, features: Array, action: int) -> tuple[Array, Array, Array]:
        hidden = np.tanh(features @ self.weights + self.bias)
        residual = -self.probabilities(features)
        residual[action] += 1.0
        hidden_gradient = residual[:, None] * self.output * (1 - hidden**2)
        return features.T @ hidden_gradient, hidden_gradient.sum(axis=0), hidden.T @ residual

    def choose(self, features: Array, rng: np.random.Generator) -> int:
        action = int(rng.choice(len(features), p=self.probabilities(features)))
        dw, db, dv = self.log_gradient(features, action)
        self._dw += dw
        self._db += db
        self._dv += dv
        return action

    def finish(self, reward: float | None) -> None:
        """Unavailable evaluator results discard the trajectory, not invent a reward."""
        if reward is not None:
            if not np.isfinite(reward):
                raise ValueError("reward must be finite")
            advantage = reward - self.baseline
            gradient = [advantage * g for g in (self._dw, self._db, self._dv)]
            norm = float(np.sqrt(sum(float(np.sum(g**2)) for g in gradient)))
            scale = self.learning_rate / max(1.0, norm / 5.0)
            self.weights += scale * gradient[0]
            self.bias += scale * gradient[1]
            self.output += scale * gradient[2]
            self.baseline = 0.95 * self.baseline + 0.05 * reward
            self.updates += 1
        self._reset_gradient()

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "shared_mlp_reinforce_v1",
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "output": self.output.tolist(),
            "learning_rate": self.learning_rate,
            "baseline": self.baseline,
            "updates": self.updates,
        }


def restore_policy(record: dict[str, Any]) -> Policy:
    if record.get("kind") == "frozen_shared_mlp_v1":
        from .benchmark import FrozenNeuralPolicy

        restored = dict(record)
        restored["kind"] = "shared_mlp_reinforce_v1"
        return FrozenNeuralPolicy(restore_policy(restored))
    if record == {"kind": "random_v1"}:
        return RandomPolicy()
    if record.get("kind") != "shared_mlp_reinforce_v1":
        raise ValueError("unknown policy version")
    policy = NeuralPolicy(seed=0, hidden=len(record["bias"]), learning_rate=record["learning_rate"])
    for name in ("weights", "bias", "output"):
        values = np.asarray(record[name], dtype=float)
        if values.shape != getattr(policy, name).shape or not np.isfinite(values).all():
            raise ValueError("invalid policy checkpoint")
        setattr(policy, name, values.copy())
    policy.baseline = float(record["baseline"])
    policy.updates = int(record["updates"])
    if not np.isfinite(policy.baseline) or policy.updates < 0:
        raise ValueError("invalid policy state")
    policy._reset_gradient()
    return policy
