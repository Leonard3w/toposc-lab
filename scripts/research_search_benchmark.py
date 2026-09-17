"""Compare proposal generation against the checksummed source of a saved run.

Runs archived Python definitions from a trusted local experiment, with current
unchanged geometry code. No evaluator or research writer is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

from toposc_lab.research.space import FixedConnectivitySpace
from toposc_lab.research.storage import ResearchStore, dumps
from toposc_lab.research.strategies import create_strategy


def benchmark(directory: Path, count: int = 8) -> dict:
    store = ResearchStore(directory)
    saved, config, manifest = store.get("strategy"), store.get("config"), store.get("manifest")
    archive = directory / "source.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["source_zip_sha256"]:
        raise ValueError("Source archive checksum mismatch")
    with zipfile.ZipFile(archive) as stream:
        source = stream.read("src/toposc_lab/research/strategies.py").decode()
        current_space = Path(__file__).resolve().parents[1] / "src/toposc_lab/research/space.py"
        if stream.read("src/toposc_lab/research/space.py") != current_space.read_bytes():
            raise ValueError("Geometry implementation differs; this isolated benchmark is not matched")
    namespace = {"__name__": "archived_search_benchmark"}
    # Deliberately execute verified, trusted local archive definitions for comparison.
    exec(compile(source, "archived_strategies.py", "exec"), namespace)  # noqa: S102
    report = {"exact_calls": 0, "proposals": count}
    sequences = []
    for name, factory in [("archived", namespace["create_strategy"]), ("optimized", create_strategy)]:
        strategy = factory(config["algorithm"], FixedConnectivitySpace(**config["space"]),
                           config["seed"], **config["search"]).resume(saved)
        start = time.perf_counter()
        proposals = strategy.propose(count)
        seconds = time.perf_counter() - start
        sequences.append([(c["id"], c["seed"], c["mutation"]) for c in proposals])
        report[name] = {"proposal_seconds": seconds,
                        "checkpoint_bytes": len(dumps(strategy.checkpoint()).encode())}
        del strategy
    report["identical_proposals"] = sequences[0] == sequences[1]
    if not report["identical_proposals"]:
        raise ValueError("Optimized default policy did not reproduce archived proposals")
    report["speedup"] = report["archived"]["proposal_seconds"] / report["optimized"]["proposal_seconds"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.directory.resolve()):
        raise ValueError("Output must be outside the archived experiment")
    result = benchmark(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
