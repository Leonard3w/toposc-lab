"""Affordable exact-physics workbench gate, including a hard process interruption.

This is an engineering demonstration, not the full Experiment 001 research run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/phase17-smoke"))
    parser.add_argument("--crash-worker", type=Path)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.engine import ResearchEngine
    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.storage import ResearchStore, atomic_text

    if args.crash_worker:
        def crash(event, payload):
            if event == "after_simulation":
                os._exit(73)
        ResearchEngine(args.crash_worker, hook=crash).run()
        raise RuntimeError("Crash boundary was not reached")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Smoke output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(name="Phase 17 affordable engineering smoke", seed=17051,
        cycles=3, exact_budget=60, candidate_budget=400, pool_size=12, batch_size=4,
        checkpoint_every=8, wall_seconds=600,
        space={"side": 4, "min_degree": 2, "max_degree": 6,
               "max_bond_length": 2**0.5, "bond_tolerance": 0.0,
               "forbid_crossings": True, "minimum_distance": .002},
        physics={"disorder_widths": [.2], "disorder_seeds": [17051, 17052]},
        surrogate={"ensemble_size": 4, "n_estimators": 10}, output_directory=str(output))
    path = ResearchService.create(config, output / "resumed")
    def pause(event, payload):
        if event == "cycle_committed" and payload["cycle"] == 1:
            ResearchStore(path).request("pause")
    first = ResearchEngine(path, hook=pause).run()
    assert first["status"] == "PAUSED" and first["cycle"] == 1
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    crashed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
        "--crash-worker", str(path)], env=env, check=False, capture_output=True, text=True)
    atomic_text(output / "crash-worker.log", crashed.stdout + crashed.stderr)
    assert crashed.returncode == 73, crashed.stderr
    interrupted = ResearchService.snapshot(path)
    assert interrupted["state"]["status"] == "INTERRUPTED"
    assert any(a["status"] == "started" for a in interrupted["attempts"])
    # Same public detached launch API as both interfaces, then bounded wait.
    pid = ResearchService.launch(path)
    import time
    deadline = time.monotonic() + 300
    final = None
    while time.monotonic() < deadline:
        final = ResearchService.snapshot(path)
        if final["state"]["status"] == "COMPLETED":
            break
        time.sleep(.2)
    assert final is not None and final["state"]["status"] == "COMPLETED"
    # Worker may publish the report after its terminal state transaction.
    deadline = time.monotonic() + 10
    while not final["report"] and time.monotonic() < deadline:
        time.sleep(.1)
        final = ResearchService.snapshot(path)
    assert final["report"]
    reference = ResearchService.create(config, output / "uninterrupted")
    ResearchEngine(reference).run()
    expected = ResearchService.snapshot(reference)
    def signature(snapshot):
        return [(c["id"], c["score"], c["validation_state"], c["raw_metrics"],
                 c.get("acquisition"), c["geometry"]["edges"])
                for c in snapshot["candidates"] if c.get("observed")]
    assert signature(final) == signature(expected)
    assert final["state"]["exact_evaluations"] == expected["state"]["exact_evaluations"] + 1
    before = len(final["attempts"])
    ResearchEngine(path).run()
    assert len(ResearchStore(path).attempts()) == before
    ResearchStore(path).integrity_check()
    report = {"gate": "PASS", "engineering_smoke_only": True, "worker_pid": pid,
              "pause_verified": True, "hard_crash_exit": crashed.returncode,
              "restart_detected": True, "resume_science_identical": True,
              "completed_replay_additional_exact_calls": 0,
              "resumed_state": final["state"], "reference_state": expected["state"],
              "model_versions": final["model_versions"], "selection_counts": final["selection_counts"],
              "diagnostics": final["diagnostics"], "manifest": final["manifest"],
              "total_charged_attempts": before + len(expected["attempts"]),
              "report": str(path / "final_report.md")}
    atomic_text(output / "smoke_gate.json", json.dumps(report, indent=2))
    print(json.dumps({"gate": report["gate"], "attempts": report["total_charged_attempts"],
                      "directory": str(output)}, indent=2))


if __name__ == "__main__":
    main()
