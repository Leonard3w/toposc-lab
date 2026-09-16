"""CLI and detached UI worker using the same durable experiment API."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("directory", type=Path)
    create.add_argument("--config", type=Path)
    for name in ("run", "inspect", "pause", "stop", "checkpoint", "resume", "archive"):
        command = sub.add_parser(name)
        command.add_argument("directory", type=Path)
        if name == "run":
            command.add_argument("--max-cycles", type=int)
    args = parser.parse_args()
    config_path = args.config if args.command == "create" else args.directory / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(raw.get("blas_threads", 1))
    from toposc_lab.research.config import ExperimentConfig
    config = ExperimentConfig(**raw)
    from toposc_lab.research.service import ResearchService
    if args.command == "create":
        print(ResearchService.create(config, args.directory))
    elif args.command == "run":
        from toposc_lab.research.engine import ResearchEngine
        print(json.dumps(ResearchEngine(args.directory).run(max_cycles=args.max_cycles), indent=2))
    elif args.command == "inspect":
        print(json.dumps(ResearchService.snapshot(args.directory)["state"], indent=2))
    else:
        ResearchService.control(args.directory, args.command)


if __name__ == "__main__":
    main()
