"""Explicit user-started engineering training and separate read-only live viewer."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .environment import BuildRules
from .policy import NeuralPolicy, RandomPolicy
from .training import CompactWiringDemo, TrainingConfig, run_training
from .viewer import make_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modularer Graphbauer: Geometrie-Demo, keine Physik"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("config", help="Standardregeln als JSON ausgeben")
    template.add_argument("--square", type=int, help="Feste Quadratpunkte statt freier Platzierung")
    train = commands.add_parser("train-demo", help="Explizit geometrisches REINFORCE-Training")
    train.add_argument("--rules", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--episodes", type=int, default=100)
    train.add_argument("--seed", type=int, default=20260911)
    train.add_argument("--policy", choices=("neural", "random"), default="neural")
    train.add_argument("--resume", action="store_true")
    watch = commands.add_parser("watch", help="Liveansicht starten; trainiert nicht")
    watch.add_argument("--output", type=Path, required=True)
    watch.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    try:
        if args.command == "config":
            rules = BuildRules()
            if args.square is not None:
                n = args.square
                if n < 2:
                    raise ValueError("square must be at least 2")
                rules = BuildRules(
                    site_count=n * n,
                    edge_count=2 * n * (n - 1),
                    width=float(n - 1),
                    height=float(n - 1),
                    maximum_edge_length=1.75,
                    minimum_degree=2,
                    maximum_degree=min(4, n * n - 1),
                    fixed_points=tuple((float(x), float(y)) for x in range(n) for y in range(n)),
                )
            print(json.dumps(asdict(rules), indent=2))
        elif args.command == "watch":
            server = make_server(args.output, args.port)
            print(
                f"Liveansicht: http://127.0.0.1:{server.server_port} (Ctrl+C beendet)", flush=True
            )
            try:
                server.serve_forever()
            finally:
                server.server_close()
        else:
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
            ):
                if os.environ.get(name) != "1":
                    raise ValueError(f"set {name}=1 before starting Python")
            import sys

            if sys.version_info[:2] != (3, 14) or not sys.dont_write_bytecode:
                raise ValueError("use Python 3.14 with -B")
            if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
                raise ValueError("set PYTHONDONTWRITEBYTECODE=1")
            rules = (
                BuildRules(**json.loads(args.rules.read_text(encoding="utf-8-sig")))
                if args.rules
                else BuildRules()
            )
            policy = NeuralPolicy(seed=args.seed) if args.policy == "neural" else RandomPolicy()

            def progress(event: dict[str, object]) -> None:
                if event["stage"] == "episode_complete":
                    print(
                        f"Episode {int(str(event['episode'])) + 1}/{args.episodes}: "
                        f"{event['evaluation']}",
                        flush=True,
                    )

            run_training(
                rules,
                TrainingConfig(args.episodes, args.seed),
                policy,
                CompactWiringDemo(),
                args.output,
                resume=args.resume,
                observer=progress,
            )
            print("Festes Budget abgeschlossen. Geometrie-Demo; keine Physikaussage.", flush=True)
    except KeyboardInterrupt:
        print("Unterbrochen. Versiegelte Episoden bleiben erhalten; --resume setzt fort.")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Graphbauer: {error}\n")


if __name__ == "__main__":
    main()
