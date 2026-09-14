"""Frozen small autonomy gate, including a real killed numerical worker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.data import load_dataset, record_to_dict, save_dataset, validate_dataset
from toposc_lab.discovery import DiscoveryConfig, DiscoveryEngine
from toposc_lab.discovery.config import derived_seed
from toposc_lab.discovery.engine import THREADS, scientific_record
from toposc_lab.discovery.storage import atomic_json, read_json
from toposc_lab.generative.physics import ExactGeometryEvaluator, candidate
from toposc_lab.generative.space import GeometrySearchSpace

GATE_CONFIG = DiscoveryConfig(seed=15201)


def science(directory: Path) -> list[dict[str, Any]]:
    return [scientific_record(r) for r in load_dataset(directory / "dataset.json").records]


def audit(directory: Path) -> dict[str, Any]:
    summary = read_json(directory / "summary.json")
    manifest = read_json(directory / "manifest.json")
    config = DiscoveryConfig(**manifest["config"])
    dataset = load_dataset(directory / "dataset.json")
    validate_dataset(dataset).raise_for_errors()
    assert len(dataset.records) == config.cycles * config.batch_size
    assert summary["minimum_pairwise_distance"] > config.minimum_distance
    assert not summary["candidate_failures"]
    assert summary["exact_attempts"] <= config.exact_attempt_cap
    selected = []
    for cycle in range(config.cycles):
        folder = directory / f"cycle-{cycle:04d}"
        plan = read_json(folder / "plan.json")
        for index in plan["selected_indices"]:
            outcome = read_json(folder / f"candidate-{index:04d}.json")
            assert outcome["exact_confirmation"]
            base = read_json(folder / f"base-{index:04d}.json")["payload"]["record"]
            confirmation = read_json(folder / f"confirmation-{index:04d}.json")["payload"]["record"]
            assert base["provenance"]["seed"] == derived_seed(config.seed, cycle, index, 3)
            assert confirmation["provenance"]["seed"] == derived_seed(config.seed, cycle, index, 4)
            for member in range(config.disorder_samples):
                realization = read_json(folder / f"disorder-{index:04d}-{member:04d}.json")[
                    "payload"
                ]
                assert realization["seed"] == derived_seed(config.seed, cycle, index, 5, member)
            for payload in (base, confirmation):
                payload.pop("record_id")
                payload.pop("provenance")
            assert base == confirmation
            selected.append(outcome["record"]["record_id"])
        assert not any(p["is_ood"] and p["strategy"] == "exploitation" for p in plan["predictions"])
        if plan["training"]:
            assert set(plan["training"]["record_ids"]) <= set(plan["history_record_ids"])
    assert selected == [r.record_id for r in dataset.records]
    for record in dataset.records:
        assert record.provenance.runtime["source_sha256"] == manifest["source_sha256"]
        assert record.provenance.runtime["config_sha256"] == config.fingerprint
        assert record.result_kind == "exact"
        assert len(record.robustness[0].seeds) == config.disorder_samples
        assert not record.observables[1].values["majorana_claim"]
        assert record.observables[-1].values["status"] == "unavailable"
    inventory = {
        p.relative_to(directory).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "writer.lock"
    }
    return {"passed": True, "record_count": len(dataset.records), "inventory": inventory}


def worker(directory: Path) -> None:
    def pause(event: str, payload: dict[str, Any]) -> None:
        if event == "after_simulation":
            atomic_json(directory / "kill-ready.json", payload)
            while True:
                time.sleep(0.1)

    DiscoveryEngine(directory, GATE_CONFIG, hook=pause).run()


def kill_worker(process: subprocess.Popen[Any]) -> None:
    """Windows venv launchers can have a separate interpreter child."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=True,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        process.kill()
    process.wait(timeout=10)


def run(directory: Path, *, resume: bool = False) -> None:
    if any(os.environ.get(n) != "1" for n in THREADS):
        raise ValueError("gate requires all four numerical thread limits set to 1")
    directory.mkdir(parents=True, exist_ok=resume)
    driver = Path(__file__).read_bytes()
    (directory / "gate-driver.py").write_bytes(driver)
    summaries = {}
    for name in ("reference", "resumed", "repeat", "random", "evolution", "surrogate"):
        config = (
            replace(GATE_CONFIG, generator=name) if name in ("random", "evolution") else GATE_CONFIG
        )
        if name == "surrogate":
            config = replace(config, surrogate=True)
        campaign = directory / name
        if name == "resumed" and not (directory / "single-cycle.json").exists():
            single = DiscoveryEngine(campaign, config).run(iterations=1)
            atomic_json(directory / "single-cycle.json", single)
            assert single["completed_cycles"] == 1
        summaries[name] = DiscoveryEngine(campaign, config).run()
        print(
            f"{name}: {summaries[name]['validated_dataset_records']} exact validated candidates",
            flush=True,
        )
    crashed = directory / "crashed"
    previous_attempts = len(list((crashed / "attempts").glob("*.json")))
    (crashed / "kill-ready.json").unlink(missing_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(crashed)],
            stdout=log,
            stderr=log,
            creationflags=flags,
        )
        try:
            deadline = time.monotonic() + 60
            while not (crashed / "kill-ready.json").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("worker did not reach its exact-simulation kill boundary")
                time.sleep(0.05)
            kill_worker(process)
        finally:
            if process.poll() is None:
                kill_worker(process)
    assert process.returncode != 0
    summaries["crashed"] = DiscoveryEngine(crashed, GATE_CONFIG).run()
    reference = science(directory / "reference")
    for name in ("resumed", "repeat", "crashed"):
        assert science(directory / name) == reference, name
    assert summaries["crashed"]["exact_attempts"] == 73 + previous_attempts
    # Known positive/trivial exact references, additional to every winner confirmation.
    space = GeometrySearchSpace()
    xy = space.coordinates
    grid = space.build(
        frozenset(e for e in space.edge_pool if np.linalg.norm(xy[e[0]] - xy[e[1]]) == 1)
    )
    evaluator = ExactGeometryEvaluator(source_provenance(Path.cwd()))
    references = tuple(evaluator.evaluate(candidate(grid, mu), 15299)[0] for mu in (2.0, 8.0))
    assert references[0].observables[0].values["eligible"] == 1
    assert references[1].observables[0].values["eligible"] == 0
    save_dataset(directory / "physics-references.json", references)
    audits = {name: audit(directory / name) for name in summaries}
    atomic_json(directory / "audit.json", audits)
    report = {
        "status": "EXPERIMENT_PASS_FULL_SUITE_PENDING",
        "summaries": summaries,
        "reproduction": {name: len(reference) for name in ("resumed", "repeat", "crashed")},
        "real_kill_exit_code": process.returncode,
        "prior_interrupted_attempts": previous_attempts,
        "gate_driver_sha256": sha256(driver).hexdigest(),
        "total_exact_attempts": sum(s["exact_attempts"] for s in summaries.values()) + 2,
        "physics_references": [record_to_dict(r) for r in references],
    }
    atomic_json(directory / "gate.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("summaries", "physics_references")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.directory:
        run(args.directory, resume=args.resume)
    else:
        parser.error("directory required")
