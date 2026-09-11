"""Bounded constructive training with pluggable evaluation and immutable checkpoints.

The shipped evaluator is geometric engineering only. Scientific evaluators must
provide their own versioned boundary, measurement, eligibility and reward contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .environment import BuildRules, BuildState, GraphBuilder
from .policy import Policy, restore_policy


@dataclass(frozen=True)
class Evaluation:
    reward: float | None
    status: str
    quantities: dict[str, float]

    def __post_init__(self) -> None:
        if not self.status or (self.reward is not None and not np.isfinite(self.reward)):
            raise ValueError("evaluation needs a status and finite reward or None")
        if any(not np.isfinite(v) for v in self.quantities.values()):
            raise ValueError("raw quantities must be finite")


class Evaluator(Protocol):
    identifier: str

    def __call__(self, state: BuildState, rules: BuildRules, seed: int) -> Evaluation: ...


class CompactWiringDemo:
    """Reward short wiring, explicitly not topology, L or superconductivity."""

    identifier = "engineering_mean_edge_length_v1"
    deterministic = True

    def __call__(self, state: BuildState, rules: BuildRules, seed: int) -> Evaluation:
        points = np.asarray(state.points)
        mean = float(np.mean([np.linalg.norm(points[a] - points[b]) for a, b in state.edges]))
        return Evaluation(
            1.0 - mean / rules.maximum_edge_length, "engineering_only", {"mean_edge_length": mean}
        )


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 100
    seed: int = 20260911

    def __post_init__(self) -> None:
        if type(self.episodes) is not int or self.episodes < 1:
            raise ValueError("episodes must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def save_sealed(path: Path, value: dict[str, Any]) -> None:
    """Publish exclusively; an interrupted file is never a completed checkpoint."""
    payload = _bytes(value)
    document = _bytes({"sha256": hashlib.sha256(payload).hexdigest(), "payload": value})
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    with temporary.open("xb") as stream:
        stream.write(document)
        stream.flush()
        os.fsync(stream.fileno())
    # Hard-link publication refuses to overwrite an existing checkpoint on Windows/POSIX.
    os.link(temporary, path)
    temporary.unlink()


def load_sealed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if hashlib.sha256(_bytes(value["payload"])).hexdigest() != value["sha256"]:
        raise ValueError(f"checksum mismatch: {path}")
    result: dict[str, Any] = value["payload"]
    return result


def _code_identity() -> str:
    digest = hashlib.sha256()
    for name in ("environment.py", "policy.py", "training.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def publish_live(output: Path, record: dict[str, Any]) -> bool:
    """Best-effort display publication; a Windows reader can temporarily block replace.

    Authoritative event traces and checkpoints remain strict. Only PermissionError
    on this derived display is tolerated; the next event tries the latest snapshot.
    """
    temporary = output / "live.partial"
    try:
        temporary.write_bytes(_bytes(record))
        os.replace(temporary, output / "live.json")
    except PermissionError:
        return False
    return True


def run_training(
    rules: BuildRules,
    config: TrainingConfig,
    policy: Policy,
    evaluator: Evaluator,
    output: Path,
    *,
    resume: bool = False,
    observer: Callable[[dict[str, Any]], None] | None = None,
    builder: GraphBuilder | None = None,
) -> dict[str, Any]:
    """Run only when explicitly called; fixed episode budget, no retries or repair.

    Sealed episode checkpoints hold the full policy and RNG schedule inputs.
    Resume repeats only an interrupted episode, in a new execution directory.
    Every transition is retained for replay. Observers get JSON copies and no RNG.
    """
    output = Path(output)
    builder = GraphBuilder(rules) if builder is None else builder
    if builder.rules != rules:
        raise ValueError("builder and training rules disagree")
    manifest = json.loads(
        _bytes(
            {
                "schema": "toposc_constructive_design_v1",
                "rules": asdict(rules),
                "config": asdict(config),
                "initial_policy": policy.snapshot(),
                "evaluator": evaluator.identifier,
                "code_sha256": _code_identity(),
                "deterministic_evaluator": bool(getattr(evaluator, "deterministic", False)),
                "builder": builder.identifier,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "threads": {
                    n: os.environ.get(n)
                    for n in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "BLIS_NUM_THREADS",
                    )
                },
            }
        )
    )
    if resume:
        archived_manifest = load_sealed(output / "manifest.json")
        # Exact known predecessor: only display publication changed, no search semantics.
        comparable = dict(archived_manifest)
        if comparable.get("code_sha256") == (
            "2b4a4fdccd514989be5562534199fb4381909ac5d609946c73cb3c88df161aae"
        ):
            comparable["code_sha256"] = manifest["code_sha256"]
        if comparable != manifest:
            raise ValueError("resume requires identical rules, budget, policy, evaluator and code")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_sealed(output / "manifest.json", manifest)
    # Exclusive writer lock; stale locks require explicit inspection after a hard crash.
    lock = output / "writer.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        return _run(rules, config, policy, evaluator, output, observer, builder)
    finally:
        lock.unlink()


def _run(
    rules: BuildRules,
    config: TrainingConfig,
    policy: Policy,
    evaluator: Evaluator,
    output: Path,
    observer: Callable[[dict[str, Any]], None] | None,
    builder: GraphBuilder,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    cache: dict[str, Evaluation] = {}
    started = time.monotonic()
    previous_hash = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    checkpoints = sorted(output.glob("episode_*.json"))
    if [p.name for p in checkpoints] != [f"episode_{i:06d}.json" for i in range(len(checkpoints))]:
        raise ValueError("episode checkpoint prefix has a gap")
    if len(checkpoints) > config.episodes:
        raise ValueError("archive exceeds episode budget")
    if (output / "complete.json").exists():
        load_sealed(output / "complete.json")
        if len(checkpoints) != config.episodes:
            raise ValueError("completed archive is missing an episode; refusing to modify it")
    for episode in range(config.episodes):
        checkpoint = output / f"episode_{episode:06d}.json"
        if checkpoint.exists():
            saved = load_sealed(checkpoint)
            if saved["episode"] != episode or saved["previous_sha256"] != previous_hash:
                raise ValueError("invalid episode chain")
            execution = output / saved["execution"]
            if not execution.resolve().is_relative_to(output.resolve()):
                raise ValueError("episode trace must stay inside its archive")
            if hashlib.sha256(execution.read_bytes()).hexdigest() != saved["trace_sha256"]:
                raise ValueError("episode trace checksum mismatch")
            policy = restore_policy(saved["policy"])
            history.append(saved["summary"])
            if saved["cache_key"] is not None and saved["evaluation"]["reward"] is not None:
                cache[saved["cache_key"]] = Evaluation(**saved["evaluation"])
            previous_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            continue
        executions = output / f"attempts_{episode:06d}"
        executions.mkdir(exist_ok=True)
        attempt = 0
        while (executions / f"execution_{attempt:04d}").exists():
            attempt += 1
        directory = executions / f"execution_{attempt:04d}"
        directory.mkdir()
        trace_path = directory / "events.jsonl"
        state = builder.initial()
        sequence = np.random.SeedSequence([config.seed, episode]).spawn(3)
        proposal_rng, action_rng = (np.random.default_rng(s) for s in sequence[:2])
        evaluation_seed = int(sequence[2].generate_state(1, dtype=np.uint64)[0])
        with trace_path.open("x", encoding="utf-8") as trace:
            step = 0
            actions = 0

            def emit(
                current: BuildState,
                status: str,
                evaluation: Evaluation | None = None,
                episode_index: int = episode,
                attempt_index: int = attempt,
            ) -> None:
                nonlocal step
                record = {
                    "episode": episode_index,
                    "attempt": attempt_index,
                    "step": step,
                    "stage": status,
                    "state": asdict(current),
                    "evaluation": None if evaluation is None else asdict(evaluation),
                    "history": history[-200:],
                    "rules": asdict(rules),
                    "evaluator": evaluator.identifier,
                    "elapsed": time.monotonic() - started,
                }
                trace.write(_bytes(record).decode())
                trace.flush()
                # Only this derived live snapshot is replaceable, never sealed episodes.
                publish_live(output, record)
                if observer is not None:
                    observer(json.loads(_bytes(record)))
                step += 1

            emit(state, builder.stage(state))
            while builder.stage(state) not in ("complete", "failed"):
                choices = builder.choices(state, proposal_rng)
                if not choices:
                    state = BuildState(state.points, state.edges, "no_feasible_action")
                else:
                    action = policy.choose(builder.features(state, choices), action_rng)
                    if type(action) is not int or not 0 <= action < len(choices):
                        raise ValueError("policy returned an invalid action")
                    state = builder.step(state, choices[action])
                    actions += 1
                emit(state, builder.stage(state))
            issues = builder.validate_final(state)
            key: str | None = None
            geometry_key: str | None = None
            cache_hit = False
            if issues:
                evaluation = Evaluation(-1.0, "geometry_rejected", {})
            else:
                geometry_key = hashlib.sha256(
                    _bytes({"points": state.points, "edges": sorted(state.edges)})
                ).hexdigest()
                # Evaluator seed is part of identity: stochastic evaluation is never conflated.
                key = hashlib.sha256(
                    _bytes(
                        {
                            "points": state.points,
                            "edges": sorted(state.edges),
                            "evaluator": evaluator.identifier,
                            "seed": None
                            if getattr(evaluator, "deterministic", False)
                            else evaluation_seed,
                        }
                    )
                ).hexdigest()
                cache_hit = key in cache
                emit(state, "evaluating")
                if cache_hit:
                    evaluation = cache[key]
                else:
                    emit(state, "evaluator_started")
                    try:
                        evaluation = evaluator(state, rules, evaluation_seed)
                        if not isinstance(evaluation, Evaluation):
                            raise TypeError("evaluator must return Evaluation")
                    except Exception as error:  # noqa: BLE001 -- retain callback failures, not interrupts
                        evaluation = Evaluation(
                            None, f"evaluator_error:{type(error).__name__}:{error}", {}
                        )
                    emit(state, "evaluator_finished", evaluation)
                    if evaluation.reward is not None:
                        cache[key] = evaluation
            policy.finish(evaluation.reward)
            summary = {
                "episode": episode,
                "reward": evaluation.reward,
                "status": evaluation.status,
                "geometry_valid": not issues,
                "cache_hit": cache_hit,
                "evaluator_called": not issues and not cache_hit,
                "attempt": attempt,
                "steps": actions,
                "geometry_key": geometry_key,
            }
            history.append(summary)
            emit(state, "episode_complete", evaluation)
        saved = {
            "execution_code_sha256": _code_identity(),
            "episode": episode,
            "previous_sha256": previous_hash,
            "summary": summary,
            "state": asdict(state),
            "policy": policy.snapshot(),
            "cache_key": key,
            "evaluation_seed": evaluation_seed,
            "evaluation": asdict(evaluation),
            "execution": trace_path.relative_to(output).as_posix(),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        }
        save_sealed(checkpoint, saved)
        previous_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "episodes": len(history),
        "history": history,
        "last_checkpoint_sha256": previous_hash,
        "evaluator": evaluator.identifier,
        "unique_valid_geometries": len(
            {row["geometry_key"] for row in history if row["geometry_key"] is not None}
        ),
        "evaluator_calls_in_sealed_episodes": sum(row["evaluator_called"] for row in history),
        "cache_hits": sum(row["cache_hit"] for row in history),
        "geometry_rejections": sum(not row["geometry_valid"] for row in history),
        "unavailable_rewards": sum(row["reward"] is None for row in history),
        "additional_executions": sum(row["attempt"] for row in history),
    }
    complete = output / "complete.json"
    if complete.exists():
        if load_sealed(complete) != result:
            raise ValueError("completed run differs from reconstructed history")
    else:
        save_sealed(complete, result)
    return result
