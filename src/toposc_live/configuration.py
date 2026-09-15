"""Thin adapter to the engine's official configuration and resume contracts."""

import json
import os
import platform
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from toposc_lab.discovery.config import DiscoveryConfig
from toposc_lab.discovery.storage import read_json


@dataclass(frozen=True)
class LaunchPlan:
    name: str
    directory: Path
    config: DiscoveryConfig
    resume: bool = False
    iterations: int | None = None


class ConfigurationAdapter:
    """Only this boundary knows the currently supported discovery engine."""

    def defaults(self) -> DiscoveryConfig:
        return DiscoveryConfig()

    def build(
        self, *, generator: str, seed: int, cycles: int, batch_size: int, pool_size: int
    ) -> DiscoveryConfig:
        return replace(
            self.defaults(),
            generator=generator,
            seed=seed,
            cycles=cycles,
            batch_size=batch_size,
            pool_size=pool_size,
        )

    def plans(
        self,
        name: str,
        directory: Path,
        config: DiscoveryConfig,
        seed_count: int = 1,
        one_cycle: bool = False,
    ) -> tuple[LaunchPlan, ...]:
        # Revalidate through the same constructor used by the official CLI.
        config = DiscoveryConfig(**asdict(config))
        if not name.strip():
            raise ValueError("Enter a campaign name.")
        if type(seed_count) is not int or not 1 <= seed_count <= 100:
            raise ValueError("Number of independent seeds must be between 1 and 100.")
        root = directory.expanduser().resolve()
        plans = tuple(
            LaunchPlan(
                name if seed_count == 1 else f"{name} / seed {config.seed + i}",
                root if seed_count == 1 else root / f"seed-{config.seed + i}",
                replace(config, seed=config.seed + i),
                iterations=1 if one_cycle else None,
            )
            for i in range(seed_count)
        )
        for plan in plans:
            if plan.directory.exists():
                raise ValueError(
                    f"New campaign location must not exist: {plan.directory}. "
                    "Open it and use Resume for an existing compatible campaign."
                )
        return plans

    def resume_plan(self, directory: Path, *, one_cycle: bool = False) -> LaunchPlan:
        directory = directory.resolve()
        manifest = read_json(directory / "manifest.json")
        config = DiscoveryConfig(**manifest["config"])
        if manifest.get("config_sha256") != config.fingerprint:
            raise ValueError("Resume configuration fingerprint mismatch.")
        if manifest.get("external_record_ids") != []:
            raise ValueError(
                "Official CLI cannot restore external exclusion archives; resume "
                "this campaign through its original API driver."
            )
        import toposc_lab.discovery.engine as engine_module
        from toposc_lab.active_learning.benchmark import source_provenance

        source = source_provenance(Path(engine_module.__file__).resolve().parents[3])
        if manifest.get("source_sha256") != source.runtime["source_sha256"]:
            raise ValueError(
                "Resume source mismatch. Inspect read-only or use the original "
                "frozen source environment; the scientific guard is not bypassed."
            )
        environment = {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "platform": platform.platform(),
            "threads": {n: os.environ.get(n) for n in engine_module.THREADS},
        }
        if manifest.get("environment") != environment:
            raise ValueError(
                "Resume runtime/thread environment mismatch; restore original runtime."
            )
        if sha256((directory / "source.zip").read_bytes()).hexdigest() != manifest.get(
            "source_zip_sha256"
        ):
            raise ValueError("Resume source archive integrity mismatch.")
        checkpoint = (
            read_json(directory / "checkpoint.json")
            if (directory / "checkpoint.json").exists()
            else {}
        )
        if checkpoint and checkpoint.get("config_sha256") != config.fingerprint:
            raise ValueError("Resume checkpoint configuration mismatch.")
        completed = checkpoint.get("completed_cycles", 0)
        if completed >= config.cycles:
            raise ValueError("Campaign is already complete; open it for inspection.")
        return LaunchPlan(directory.name, directory, config, True, 1 if one_cycle else None)

    def summary(self, plans: tuple[LaunchPlan, ...]) -> str:
        first = plans[0]
        config = first.config
        return (
            f"{len(plans)} independent campaign process(es); seeds "
            f"{', '.join(str(p.config.seed) for p in plans)}\n"
            f"Protocol: {config.stratum}\nGenerator: {config.resolved_generator}\n"
            f"Frozen success threshold: {config.success_threshold}\n"
            f"Exact attempt cap: {config.exact_attempt_cap} per seed; "
            f"{sum(p.config.exact_attempt_cap for p in plans)} total (including retry reserve).\n"
            f"Invocation: {'one additional cycle, then checkpoint and exit' if first.iterations else 'remaining configured cycles'}\n"
            "Budget is derived by DiscoveryConfig; unused reserve is not a request for more work.\n"
            "Finite validated wiring stratum only. No Majorana or thermodynamic claim.\n"
            "This launcher does not implement a Campaign-1A or Phase-16B protocol.\n"
            "Each seed is independent; this is not a generator-superiority benchmark.\n"
            "Closing the GUI leaves processes running. Safe Stop/Pause are unavailable.\n\n"
            + "\n".join(str(p.directory) for p in plans)
            + "\n\nOfficial configuration (first seed):\n"
            + json.dumps(asdict(config), indent=2)
        )
