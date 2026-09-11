"""User-started three-arm engineering pilot; no physics and no implicit seed selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .environment import BuildRules
from .policy import Array, NeuralPolicy, Policy, RandomPolicy
from .training import (
    CompactWiringDemo,
    TrainingConfig,
    _code_identity,
    load_sealed,
    publish_live,
    run_training,
    save_sealed,
)

DEFAULT_SEEDS = tuple(range(20261001, 20261011))
ARMS = ("neural", "frozen", "random")


class FrozenNeuralPolicy:
    """Same initial action probabilities as the learner, without learning or gradients."""

    def __init__(self, policy: Policy) -> None:
        if not isinstance(policy, NeuralPolicy):
            raise TypeError("frozen policy requires a NeuralPolicy")
        self.policy = policy

    def choose(self, features: Array, rng: np.random.Generator) -> int:
        return int(rng.choice(len(features), p=self.policy.probabilities(features)))

    def finish(self, reward: float | None) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        record = self.policy.snapshot()
        record["kind"] = "frozen_shared_mlp_v1"
        return record


def summarize_trial(result: dict[str, Any]) -> dict[str, Any]:
    history = result["history"]
    rewards = [row["reward"] for row in history]
    valid = [
        row["reward"] for row in history if row["geometry_valid"] and row["reward"] is not None
    ]
    return {
        "episodes": len(history),
        # Operational failures make this metric unavailable, never disappear from denominator.
        "mean_reward": None if any(r is None for r in rewards) else float(np.mean(rewards)),
        "valid_fraction": sum(row["geometry_valid"] for row in history) / len(history),
        "mean_valid_reward": float(np.mean(valid)) if valid else None,
        "best_valid_reward": max(valid) if valid else None,
        "evaluator_calls": result["evaluator_calls_in_sealed_episodes"],
        "cache_hits": result["cache_hits"],
        "unavailable_rewards": result["unavailable_rewards"],
        "additional_executions": result["additional_executions"],
    }


def summarize_pairs(trials: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for opponent in ("frozen", "random"):
        differences = [
            None
            if trial["arms"]["neural"]["mean_reward"] is None
            or trial["arms"][opponent]["mean_reward"] is None
            else trial["arms"]["neural"]["mean_reward"] - trial["arms"][opponent]["mean_reward"]
            for trial in trials
        ]
        available = [d for d in differences if d is not None]
        comparisons["neural_minus_" + opponent] = {
            "paired_differences": differences,
            "available_pairs": len(available),
            "mean_difference": float(np.mean(available)) if available else None,
            "wins": sum(d > 1e-12 for d in available),
            "ties": sum(abs(d) <= 1e-12 for d in available),
            "losses": sum(d < -1e-12 for d in available),
        }
    return {
        "kind": "descriptive_engineering_pilot",
        "trials": trials,
        "comparisons": comparisons,
        "scientific_superiority_claim": False,
    }


def run_benchmark(
    output: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    episodes: int = 100,
    resume: bool = False,
) -> dict[str, Any]:
    """Three independent policies per seed; paired actor initialization and RNG schedules.

    No shared reward cache across arms, no hyperparameter tuning or early stopping.
    A suite-level live snapshot follows whichever arm is currently running.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    for seed in seeds:
        TrainingConfig(episodes, seed)
    output = Path(output)
    manifest = {
        "protocol": "constructive_three_arm_engineering_pilot_v1",
        "seeds": list(seeds),
        "episodes_per_arm": episodes,
        "arms": list(ARMS),
        "total_episodes": len(seeds) * len(ARMS) * episodes,
        "core_code_sha256": _code_identity(),
        "benchmark_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_metric": "paired_mean_reward_including_geometry_rejections",
    }
    if resume:
        if load_sealed(output / "benchmark_manifest.json") != manifest:
            raise ValueError("benchmark protocol or code changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_sealed(output / "benchmark_manifest.json", manifest)
    if (output / "benchmark_complete.json").exists():
        load_sealed(output / "benchmark_complete.json")
        if any(
            not (output / f"seed_{seed}" / arm / "complete.json").exists()
            for seed in seeds
            for arm in ARMS
        ):
            raise ValueError("completed benchmark is missing an arm; refusing to modify it")
    lock = output / "benchmark.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        trials: list[dict[str, Any]] = []
        for index, seed in enumerate(seeds):
            metrics: dict[str, Any] = {}
            # Rotate execution order to avoid always assigning the first arm to the learner.
            order = ARMS[index % 3 :] + ARMS[: index % 3]
            for arm in order:
                policy: Policy = (
                    RandomPolicy()
                    if arm == "random"
                    else FrozenNeuralPolicy(NeuralPolicy(seed=seed))
                    if arm == "frozen"
                    else NeuralPolicy(seed=seed)
                )
                destination = output / f"seed_{seed}" / arm

                def observe(
                    event: dict[str, Any], current_arm: str = arm, current_seed: int = seed
                ) -> None:
                    event["benchmark_arm"] = current_arm
                    event["benchmark_seed"] = current_seed
                    publish_live(output, event)
                    if event["stage"] == "episode_complete":
                        print(
                            f"Seed {current_seed} | {current_arm} | "
                            f"Episode {event['episode'] + 1}/{episodes} | "
                            f"{event['evaluation']['reward']}",
                            flush=True,
                        )

                result = run_training(
                    BuildRules(),
                    TrainingConfig(episodes, seed),
                    policy,
                    CompactWiringDemo(),
                    destination,
                    resume=resume and destination.exists(),
                    observer=observe,
                )
                metrics[arm] = summarize_trial(result)
            trials.append({"seed": seed, "arms": metrics})
        summary = summarize_pairs(trials)
        complete = output / "benchmark_complete.json"
        if complete.exists():
            if load_sealed(complete) != summary:
                raise ValueError("benchmark summary differs from verified arms")
        else:
            save_sealed(complete, summary)
        return summary
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dreiarmiger Graphbauer-Pilot: keine Physik")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if sys.version_info[:2] != (3, 14) or not sys.dont_write_bytecode:
            raise ValueError("use Python 3.14 with -B")
        for name in (
            "PYTHONDONTWRITEBYTECODE",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            if os.environ.get(name) != "1":
                raise ValueError(f"set {name}=1 before starting Python")
        result = run_benchmark(
            args.output, seeds=tuple(args.seeds), episodes=args.episodes, resume=args.resume
        )
        print(json.dumps(result["comparisons"], indent=2))
        print("Pilot abgeschlossen. Deskriptiver Vergleich, keine Physikaussage.")
    except KeyboardInterrupt:
        print("Unterbrochen. Gleichen Befehl mit --resume fortsetzen.")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Graphbauer-Benchmark: {error}\n")


if __name__ == "__main__":
    main()
