"""Run or resume a bounded discovery campaign from its frozen JSON config."""

import argparse
import json
from pathlib import Path

from toposc_lab.discovery import DiscoveryConfig, DiscoveryEngine
from toposc_lab.discovery.storage import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    if args.config:
        config = DiscoveryConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    elif (args.directory / "manifest.json").exists():
        config = DiscoveryConfig(**read_json(args.directory / "manifest.json")["config"])
    else:
        config = DiscoveryConfig()
    summary = DiscoveryEngine(args.directory, config).run(iterations=args.iterations)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
