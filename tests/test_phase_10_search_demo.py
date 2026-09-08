from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

from toposc_lab.search import (
    create_search_checkpoint,
    load_search_checkpoint,
    save_search_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO = PROJECT_ROOT / "examples" / "phase_10_search_demo.py"


def test_demo_runs_reproducibly_without_output_files(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    first = subprocess.run(
        [sys.executable, "-B", str(DEMO)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    second = subprocess.run(
        [sys.executable, "-B", str(DEMO)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert first.stdout == second.stdout
    assert "Attempts per arm and trial: 30" in first.stdout
    assert "NOT a superconductivity/discovery benchmark" in first.stdout
    assert "None = no hit/no available scalar, not zero" in first.stdout
    assert list(tmp_path.iterdir()) == []


def test_demo_retains_resources_and_compatible_evolution_checkpoints(tmp_path: Path) -> None:
    demo = runpy.run_path(str(DEMO))
    result = demo["run_demo"]()
    assert len(result.trials) == 5
    for trial in result.trials:
        for arm in (trial.evolution_arm, trial.random_arm):
            assert arm.final_progress.attempt_count == 30
            assert arm.final_progress.available_count == 30
            for fitness in arm.fitness_history:
                for member in fitness.population.members:
                    genome = member.genome
                    assert genome.n_sites == 6
                    assert len(genome.edges) == 8
                    assert set(demo["RING"]) <= {(e.source, e.target) for e in genome.edges}
    checkpoint = create_search_checkpoint(
        result.trials[0].evolution,
        evaluator_identifier=result.protocol.evaluator_identifier,
        code_version=result.protocol.code_version,
    )
    path = tmp_path / "evolution.zip"
    save_search_checkpoint(path, checkpoint)
    loaded = load_search_checkpoint(path)
    assert loaded.result.config == checkpoint.result.config
    assert loaded.result.seed == checkpoint.result.seed
    assert loaded.requested_generation_count == 4
