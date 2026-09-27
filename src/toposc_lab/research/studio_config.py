"""Studio metadata and validation over the existing canonical ExperimentConfig.

Presets are ordinary configurations. This module never generates a geometry,
starts a worker, opens a results database, or evaluates physics.
"""

from __future__ import annotations

import copy
import json
from dataclasses import MISSING, asdict, fields
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from toposc_lab.research.config import ExperimentConfig

FAMILIES = ("regular", "rewired_square", "amorphous_planar", "constrained_embedded")
VARIABLES = ("connectivity", "coordinates", "hopping", "chemical_potential", "pairing")
OBSERVABLES = (
    "Q",
    "interior_localizer_gap",
    "edge_weight",
    "center_weight",
    "chern",
    "spectrum",
    "majorana",
)
PRESETS = (
    "quick_test",
    "phase19_like",
    "disorder_scan",
    "candidate_validation",
    "broad_random_search",
)
STUDIO_ADAPTER = "phase20.configurable-chiral-p-wave.v1"
_MANDATORY_OUTPUT = (
    "save_every_candidate",
    "save_rejected_candidates",
    "save_eigenvalues",
    "save_raw_realizations",
    "generate_report",
)


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def resolve_studio_defaults(config: ExperimentConfig) -> None:
    """Resolve v2 only; legacy serialization and scientific contracts stay intact."""
    from toposc_lab.geometry.generators.hard_core_planar import HardCorePlanarConfig
    from toposc_lab.research.embedded import EmbeddedDomain
    from toposc_lab.research.space import FixedConnectivitySpace
    from toposc_lab.research.studio_physics import FROZEN_NUMERICAL_SETTINGS

    if config.geometry_space is None:
        config.geometry_space = "embedded_sampling"
    if config.algorithm is None:
        config.algorithm = {
            "fixed_candidates": "fixed_candidates",
            "fixed_connectivity": "random",
        }.get(config.geometry_space, "embedded_random")
    if config.objective is None:
        config.objective = "robustness_quality_mean"
    if config.baselines is None:
        config.baselines = []
    if config.search is None:
        config.search = {}
    if config.space is None:
        config.space = {}
    if not isinstance(config.space, dict) or not isinstance(config.physics, dict):
        raise TypeError("space and physics must be JSON objects")
    config.space = _plain(config.space)
    if config.geometry_space == "embedded_sampling":
        from toposc_lab.research.studio_space import EmbeddedSamplingSpace

        defaults = {}
        for item in fields(EmbeddedSamplingSpace):
            if item.name == "domain":
                continue
            if item.default is not MISSING:
                defaults[item.name] = copy.deepcopy(item.default)
            elif item.default_factory is not MISSING:
                defaults[item.name] = item.default_factory()
        config.space = {
            **defaults,
            "families": ["regular"],
            "side": 8,
            "generator": asdict(HardCorePlanarConfig()),
            "rewires": [1, 4, 8],
            **config.space,
        }
        config.space["generator"] = asdict(HardCorePlanarConfig(**config.space["generator"]))
    elif config.geometry_space == "fixed_candidates":
        config.space = {"candidates": [], **config.space}
    elif config.geometry_space == "fixed_connectivity":
        config.space = {**asdict(FixedConnectivitySpace()), **config.space}
        from toposc_lab.research.config import ExperimentConfig
        from toposc_lab.research.strategies import SearchStrategy

        supplied_search = dict(config.search)
        config.search = {
            **{
                key: copy.deepcopy(parameter.default)
                for key, parameter in signature(SearchStrategy).parameters.items()
                if parameter.default is not Parameter.empty
                and key not in {"seed", "experiment_id", "code_version"}
            },
            **ExperimentConfig().search,
            **supplied_search,
        }
    if "domain" in config.space:
        raise ValueError(
            "The canonical physical domain belongs in physics.domain, not space.domain"
        )
    domain = config.physics.get("domain")
    if domain is None and config.geometry_space == "fixed_connectivity":
        side = config.space["side"]
        domain = asdict(
            EmbeddedDomain(bounds=(0, side - 1, 0, side - 1), bulk_inset=min(2.0, (side - 1) / 3))
        )
    if domain is None:
        side = config.space.get("side", 8)
        domain = asdict(
            EmbeddedDomain(bounds=(0, side - 1, 0, side - 1), bulk_inset=min(2.0, (side - 1) / 3))
        )
    config.physics = {
        "adapter_id": STUDIO_ADAPTER,
        "confirmation": False,
        "domain": domain,
        **config.physics,
    }
    families = config.space.get("families", [])
    variable_defaults = dict.fromkeys(VARIABLES, "locked")
    if config.geometry_space == "fixed_connectivity" or any(f != "regular" for f in families):
        variable_defaults["connectivity"] = "variable"
    if any(f in ("amorphous_planar", "constrained_embedded") for f in families):
        variable_defaults["coordinates"] = "variable"
    supplied = {} if config.studio is None else _plain(config.studio)
    if not isinstance(supplied, dict):
        raise TypeError("studio must be a JSON object")
    if not isinstance(supplied.get("variables", {}), dict):
        raise TypeError("studio.variables must be a JSON object")
    if not isinstance(supplied.get("output", {}), dict):
        raise TypeError("studio.output must be a JSON object")
    config.studio = {
        "description": "",
        "tags": [],
        "preset": "",
        "objective_mode": "none",
        "observables": list(OBSERVABLES),
        "definitions": copy.deepcopy(FROZEN_NUMERICAL_SETTINGS),
        **supplied,
        "variables": {**variable_defaults, **supplied.get("variables", {})},
        "output": {
            **dict.fromkeys(_MANDATORY_OUTPUT, True),
            "save_plots": False,
            **supplied.get("output", {}),
        },
    }
    config.space = _plain(config.space)
    config.search = _plain(config.search)
    config.studio = _plain(config.studio)


def validate_studio(config: ExperimentConfig) -> None:
    """Reject unsupported capabilities instead of accepting inert science settings."""
    if config.schema_version != 2:
        return
    studio = config.studio
    if not isinstance(studio, dict):
        raise TypeError("schema 2 requires studio settings")
    allowed = {
        "description",
        "tags",
        "preset",
        "objective_mode",
        "variables",
        "observables",
        "output",
        "source",
        "definitions",
    }
    if set(studio) - allowed:
        raise ValueError(f"unknown studio settings: {sorted(set(studio) - allowed)}")
    for key in ("description", "preset"):
        if not isinstance(studio[key], str):
            raise TypeError(f"studio.{key} must be text")
    tags = studio["tags"]
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("studio.tags must be a list of strings")
    if studio["objective_mode"] not in ("none", "single"):
        raise ValueError("weighted and Pareto objectives are not supported by these strategies")
    variables = studio["variables"]
    if set(variables) != set(VARIABLES) or any(
        v not in ("locked", "variable") for v in variables.values()
    ):
        raise ValueError("search variables must explicitly be locked or variable")
    if any(variables[key] != "locked" for key in ("hopping", "chemical_potential", "pairing")):
        raise ValueError("current strategies cannot vary physics parameters; keep them locked")
    observables = studio["observables"]
    if (
        not isinstance(observables, list)
        or not observables
        or any(item not in OBSERVABLES for item in observables)
        or len(set(observables)) != len(observables)
    ):
        raise ValueError("observables must be distinct supported observable names")
    output = studio["output"]
    if set(output) - {*_MANDATORY_OUTPUT, "save_plots"}:
        raise ValueError("unsupported output setting; eigenvector archives are not implemented")
    if any(output.get(key) is not True for key in _MANDATORY_OUTPUT):
        raise ValueError(
            "candidate, rejection, eigenvalue, raw realization and report saves are required"
        )
    if type(output.get("save_plots")) is not bool:
        raise ValueError("save_plots must be boolean")
    if config.geometry_space == "fixed_connectivity":
        supported = {"random", "evolution", "map_elites", "surrogate_map_elites"}
        if config.algorithm not in supported:
            raise ValueError(
                "fixed_connectivity requires a registered connectivity search strategy"
            )
        if variables["coordinates"] != "locked" or variables["connectivity"] != "variable":
            raise ValueError("fixed_connectivity searches connectivity and locks coordinates")
    elif config.geometry_space == "embedded_sampling":
        if config.algorithm != "embedded_random":
            raise ValueError("embedded_sampling currently supports embedded_random only")
        families = config.space.get("families")
        if (
            not isinstance(families, list)
            or not families
            or len(set(families)) != len(families)
            or any(family not in FAMILIES for family in families)
        ):
            raise ValueError("select distinct supported geometry families")
        if any(f != "regular" for f in families) and variables["connectivity"] == "locked":
            raise ValueError("selected geometry families change locked connectivity")
        if any(f in ("amorphous_planar", "constrained_embedded") for f in families):
            if variables["coordinates"] == "locked":
                raise ValueError("selected geometry families change locked coordinates")
        elif variables["coordinates"] != "locked":
            raise ValueError("square families have fixed coordinates")
        if families == ["regular"] and variables["connectivity"] != "locked":
            raise ValueError("regular square has fixed connectivity")
        if config.baselines:
            raise ValueError(
                "embedded sampling includes regular as a family; baselines must be empty"
            )
    elif config.geometry_space == "fixed_candidates":
        if config.algorithm != "fixed_candidates":
            raise ValueError("fixed candidate follow-ups require the fixed_candidates strategy")
        if any(v != "locked" for v in variables.values()):
            raise ValueError("fixed candidate follow-ups must lock every variable")
        if not isinstance(config.space.get("candidates"), list) or not config.space["candidates"]:
            raise ValueError("select at least one exact candidate for a follow-up")
        if config.baselines:
            raise ValueError(
                "fixed candidate follow-ups use selected references; baselines must be empty"
            )
    else:
        raise ValueError("unsupported studio geometry space")
    optimizing = config.algorithm in {"evolution", "map_elites", "surrogate_map_elites"}
    if optimizing and studio["objective_mode"] != "single":
        raise ValueError("optimization strategies require single objective mode")
    if optimizing and "variable" not in variables.values():
        raise ValueError("optimization requires at least one searchable variable")
    if config.geometry_space != "fixed_connectivity" and config.search:
        raise ValueError("the selected sampling strategy has no algorithm-specific search settings")
    if config.physics.get("adapter_id") != STUDIO_ADAPTER:
        raise ValueError(
            "schema 2 uses the configurable versioned adapter; legacy configs remain schema 1"
        )
    if config.physics.get("finite_size"):
        raise ValueError("finite-size extension is not implemented by the active studio adapter")
    from toposc_lab.research.studio_physics import FROZEN_NUMERICAL_SETTINGS

    if studio["definitions"] != FROZEN_NUMERICAL_SETTINGS:
        raise ValueError("validated numerical definitions are immutable")
    protocol = config.physics_protocol()
    stages = _stage_count(config)
    if config.exact_budget < stages:
        raise ValueError("Exact budget cannot cover one complete candidate.")
    if _memory_estimate(config) > 8 * 1024**3:
        raise ValueError("Dense localizer working-set estimate exceeds 8 GiB; reduce site count.")
    if config.geometry_space == "fixed_connectivity":
        side = config.space["side"]
        domain = protocol.domain
        if domain.bounds != (0.0, float(side - 1), 0.0, float(side - 1)):
            raise ValueError("fixed_connectivity requires domain bounds [0, side-1]²")
        if domain.boundary_shell >= 1:
            raise ValueError("fixed_connectivity boundary shell must be below unit site spacing")


def switch_geometry_space(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Explicit UI switch to a compatible draft; no candidate is generated.

    Switching replaces geometry/search settings and applicable permissions. The
    caller displays the resulting JSON before any preview or start action.
    """
    if name not in {"embedded_sampling", "fixed_connectivity", "fixed_candidates"}:
        raise ValueError("unsupported studio geometry space")
    if raw.get("schema_version") != 2:
        raise ValueError("clone a schema 2 configuration to switch geometry space")
    result = copy.deepcopy(raw)
    previous = result.get("space", {})
    bounds = result.get("physics", {}).get("domain", {}).get("bounds", [0, 7, 0, 7])
    inferred = int(bounds[1] - bounds[0] + 1)
    side = previous.get("side", inferred)
    result.update(geometry_space=name, search={}, baselines=[])
    result["studio"]["objective_mode"] = "none"
    result["studio"]["variables"] = dict.fromkeys(VARIABLES, "locked")
    if name == "fixed_connectivity":
        from toposc_lab.research.space import FixedConnectivitySpace

        result["space"] = _plain(asdict(FixedConnectivitySpace(side=side)))
        result["algorithm"] = "random"
        result["studio"]["variables"]["connectivity"] = "variable"
    elif name == "embedded_sampling":
        from toposc_lab.research.config import ExperimentConfig

        result["space"] = ExperimentConfig(schema_version=2, space={"side": side}).space
        result["algorithm"] = "embedded_random"
    else:
        result["space"] = {"candidates": previous.get("candidates", [])}
        result["algorithm"] = "fixed_candidates"
    return result


def preset(name: str) -> dict[str, Any]:
    """Return an independent resolved normal config; no run or geometry is created."""
    from toposc_lab.research.config import ExperimentConfig

    if name not in PRESETS:
        raise ValueError(f"unknown preset: {name}")
    raw: dict[str, Any] = {
        "schema_version": 2,
        "name": name,
        "output_directory": f"results/research/{name}",
        "seed": 20001,
        "cycles": 1,
        "candidate_budget": 1,
        "pool_size": 1,
        "batch_size": 1,
        "exact_budget": 3,
        "wall_seconds": 300.0,
        "checkpoint_every": 1,
        "space": {"families": ["regular"], "side": 8},
        "physics": {
            "disorder_widths": [3.0],
            "disorder_seeds": [200101, 200102],
            "clean_seed": 200100,
            "confirmation": False,
        },
        "studio": {
            "preset": name,
            "description": "Editable preset; starts only after preview confirmation.",
        },
    }
    if name == "phase19_like":
        raw.update(
            cycles=4,
            candidate_budget=16,
            pool_size=4,
            batch_size=4,
            exact_budget=160,
            wall_seconds=1800.0,
        )
        raw["space"]["families"] = list(FAMILIES)
        raw["physics"].update(
            disorder_widths=[3.0, 6.0, 9.0], disorder_seeds=[200101, 200102, 200103]
        )
    elif name == "disorder_scan":
        raw.update(exact_budget=7)
        raw["physics"]["disorder_widths"] = [3.0, 6.0, 9.0]
    elif name == "candidate_validation":
        # A standalone reusable validation recipe. Selecting candidates replaces
        # this regular reference with fixed_candidates and its exact geometry.
        raw.update(exact_budget=7)
        raw["physics"]["disorder_widths"] = [3.0, 6.0, 9.0]
        raw["studio"]["description"] = (
            "Validation recipe; select stored candidates to create a fixed follow-up."
        )
    elif name == "broad_random_search":
        raw.update(
            cycles=10,
            candidate_budget=40,
            pool_size=4,
            batch_size=4,
            exact_budget=280,
            checkpoint_every=20,
            wall_seconds=1800.0,
        )
        raw["space"]["families"] = list(FAMILIES)
        raw["physics"]["disorder_widths"] = [3.0, 6.0, 9.0]
    return ExperimentConfig.from_dict(raw).to_dict()


def _site_count(config: ExperimentConfig) -> int:
    if config.geometry_space == "fixed_candidates":
        return max((int(c["geometry"]["n_sites"]) for c in config.space["candidates"]), default=0)
    counts = [int(config.space.get("side", 8)) ** 2]
    if any(
        f in ("amorphous_planar", "constrained_embedded") for f in config.space.get("families", [])
    ):
        counts.append(int(config.space["generator"]["n_sites"]))
    return max(counts)


def _memory_estimate(config: ExperimentConfig) -> int:
    return 8 * (4 * _site_count(config)) ** 2 * 16


def _stage_count(config: ExperimentConfig) -> int:
    protocol = config.physics_protocol()
    stages = int(getattr(protocol, "clean_reference", True)) + int(protocol.confirmation)
    if "robustness" in protocol.validators:
        stages += len(protocol.disorder_widths) * len(protocol.disorder_seeds)
    return stages


def planned_evaluations(config: ExperimentConfig) -> int:
    """Upper bound on scheduled complete stages, excluding retry attempts.

    Rejected/duplicate proposals can reduce the actual stochastic candidate
    count. Exact attempts have a separate cap and include interrupted retries.
    """
    stages = _stage_count(config)
    if not stages:
        return 0
    slots = config.exact_budget // stages
    baseline_count = min(len(config.baselines), slots)
    candidate_count = min(
        config.candidate_budget, config.cycles * config.batch_size, slots - baseline_count
    )
    if config.geometry_space == "fixed_candidates":
        candidate_count = min(candidate_count, len(config.space["candidates"]))
    elif config.geometry_space == "embedded_sampling" and config.space["families"] == ["regular"]:
        candidate_count = min(candidate_count, 1)
    return (baseline_count + candidate_count) * stages


def preview_config(raw: dict[str, Any], check_output: bool = True) -> dict[str, Any]:
    """Resolve once and return the exact configuration to display and then create."""
    from toposc_lab.research.config import ExperimentConfig

    result: dict[str, Any] = {
        "config": None,
        "config_sha256": None,
        "errors": [],
        "warnings": [],
        "planned_evaluations": 0,
        "summary": "",
    }
    try:
        config = ExperimentConfig.from_dict(raw)
        config.validate_plugins()
        result["config"], result["config_sha256"] = config.to_dict(), config.fingerprint
        if check_output:
            directory = Path(config.output_directory).expanduser()
            if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
                result["errors"].append(
                    "Output directory exists and is not empty; choose a new run directory."
                )
        protocol = config.physics_protocol()
        stages = _stage_count(config)
        if config.geometry_space == "fixed_candidates":
            candidates = len(config.space["candidates"])
        else:
            candidates = min(config.candidate_budget, config.cycles * config.batch_size)
            candidates += len(config.baselines)
        requested = candidates * stages
        result["planned_evaluations"] = planned_evaluations(config)
        result["requested_evaluations"] = requested
        if requested > config.exact_budget:
            result["warnings"].append(
                f"Requested schedule needs {requested} exact stages; the {config.exact_budget} attempt budget may truncate it."
            )
        if config.exact_budget < stages:
            result["errors"].append("Exact budget cannot cover one complete candidate.")
        sites = _site_count(config)
        # Conservative working-set estimate, including several complex 4N localizers.
        memory = _memory_estimate(config)
        result["estimated_memory_bytes"] = memory
        if memory > 8 * 1024**3:
            result["errors"].append(
                "Dense localizer working-set estimate exceeds 8 GiB; reduce site count."
            )
        elif memory > 1024**3:
            result["warnings"].append("Dense localizer working-set estimate exceeds 1 GiB.")
        if requested > 1000:
            result["warnings"].append(
                "This configuration requests over 1,000 exact stages; review the runtime budget."
            )
        if (
            config.geometry_space == "embedded_sampling"
            and config.space["families"] == ["regular"]
            and candidates > 1
        ):
            result["warnings"].append(
                "Regular square is one unique geometry; duplicate proposals do not create new candidates."
            )
        if config.geometry_space != "fixed_candidates" and not (
            config.geometry_space == "embedded_sampling" and config.space["families"] == ["regular"]
        ):
            result["warnings"].append(
                "Planned evaluations are an upper bound; invalid or duplicate proposals may reduce the schedule."
            )
        variables = (config.studio or {}).get("variables", {"connectivity": "variable"})
        fixed = ", ".join(
            f"{k}={config.physics[k]}" for k in ("hopping", "chemical_potential", "pairing")
        )
        families = ", ".join(config.space.get("families", [config.geometry_space]))
        result["summary"] = (
            f"Experiment: {config.name}\nGeometry: {families}; maximum N={sites}\n"
            f"Fixed physics: {fixed}\nSearch variables: "
            + (", ".join(k for k, v in variables.items() if v == "variable") or "none")
            + f"\nSearch: {config.algorithm}; objective mode: {(config.studio or {}).get('objective_mode', 'single')}"
            + f"\nDisorder W: {list(protocol.disorder_widths)}; seeds: {list(protocol.disorder_seeds)}"
            + f"\nPlanned exact stages (upper bound): {result['planned_evaluations']} (requested {requested})"
            + f"\nAttempt cap: {config.exact_budget}; runtime cap: {config.wall_seconds:g} s"
            + f"\nDense working-set estimate: {memory / 1024**2:.1f} MiB (not a guarantee)"
            + "\nRuntime estimate: unavailable before measurements"
            + f"\nOutput: {config.output_directory}\nConfig SHA256: {config.fingerprint}"
        )
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        result["errors"].append(str(error))
    return result


_DESCRIPTIONS = {
    "seed": "Master seed for deterministic geometry proposals and search state.",
    "exact_budget": "Hard cap on attempted exact stages, including failures and retries.",
    "candidate_budget": "Hard cap on generated proposals, including rejected proposals.",
    "cycles": "Maximum number of search cycles.",
    "pool_size": "Maximum proposal pool prepared per cycle.",
    "batch_size": "Maximum candidates selected for exact evaluation per cycle.",
    "checkpoint_every": "Persist a recovery checkpoint after this many exact attempts.",
    "wall_seconds": "Pause cooperatively after the current exact stage when elapsed time reaches this cap.",
    "concurrency": "The durable deterministic engine currently supports exactly one exact worker.",
    "blas_threads": "BLAS thread count set before the detached worker imports numerical libraries.",
    "retry_limit": "Maximum retries for an interrupted stage; each attempt consumes budget.",
    "objective": "Existing exact scalar observable used only when an active strategy supports optimization.",
    "physics.disorder_widths": "Increasing unique W values; uniform onsite offsets lie in [-W/2,W/2].",
    "physics.disorder_seeds": "Distinct explicit nonnegative seeds, retained with every realization.",
    "physics.paired_seeds": "Use the declared same seed for every candidate at each disorder width.",
    "physics.clean_reference": "Evaluate the exact clean reference stage.",
    "physics.confirmation": "Repeat the clean calculation as an independent reproducibility stage.",
    "physics.kappas": "Positive localizer scale values; raw localizer diagnostics remain separate from Q.",
    "physics.energy_cutoff": "Absolute energy window used by existing localization diagnostics.",
    "physics.group_tolerance": "Near-degenerate states are grouped before localization statistics.",
    "physics.boundary_widths": "Boundary strips for existing low-energy state probability summaries.",
    "physics.domain": "Canonical physical box and boundary definition shared by geometry and diagnostics.",
    "studio.objective_mode": "Pure sampling or an existing single scalar objective; weighted/Pareto unsupported.",
    "studio.observables": "Observables selected for inspection; exact raw diagnostics are always preserved.",
    "space.candidates": "Lossless selected candidate payloads and provenance for an immutable follow-up cohort.",
    "space.rewires": "Allowed initialization rewire counts for the rewired square family.",
    "space.generator.minimum_separation": "Minimum Euclidean site distance in the physical coordinate box.",
    "space.generator.maximum_edge_length": "Maximum allowed physical edge length.",
    "space.generator.max_point_proposals": "Hard bound on proposed sites during graph construction.",
    "space.generator.max_complete_attempts": "Hard bound on complete graph construction attempts.",
}


def _category(path: str) -> str:
    if path.startswith("studio.definitions"):
        return "Numerical Settings"
    if path.startswith("studio.variables"):
        return "Search Space"
    if path.startswith("studio.output") or path == "output_directory":
        return "Output / Storage"
    if path.startswith("studio.objective") or path in ("objective", "studio.observables"):
        return "Objectives"
    if path.startswith("space") or path in ("geometry_space", "baselines"):
        return "Geometry"
    if path.startswith("physics."):
        if path.split(".")[1] in {
            "disorder_widths",
            "disorder_seeds",
            "clean_seed",
            "confirmation",
            "paired_seeds",
            "clean_reference",
        }:
            return "Disorder / Ensemble"
        return "Physics"
    if path in ("algorithm", "surrogate_model"):
        return "Search Method"
    if path.startswith(("search.", "surrogate.")) or path in {
        "cycles",
        "candidate_budget",
        "pool_size",
        "batch_size",
        "retrain_every",
    }:
        return "Search Settings"
    if path in {
        "concurrency",
        "blas_threads",
        "checkpoint_every",
        "wall_seconds",
        "exact_budget",
        "retry_limit",
    }:
        return "Numerical Settings"
    return "Experiment"


def field_specs(config: dict[str, Any] | ExperimentConfig) -> list[dict[str, Any]]:
    """Render metadata for valid configurations and partially edited drafts.

    Presentation does not validate scientific consistency. A temporarily invalid
    family/lock combination must remain editable until the user previews it.
    """
    from toposc_lab.research.config import ExperimentConfig

    if isinstance(config, ExperimentConfig):
        raw = config.to_dict()
    else:
        if not isinstance(config, dict):
            raise TypeError("configuration draft must be an object")
        raw = copy.deepcopy(config)
    resolved = SimpleNamespace(
        algorithm=raw.get("algorithm", "embedded_random"),
        geometry_space=raw.get("geometry_space", "embedded_sampling"),
        schema_version=raw.get("schema_version", 2),
    )
    common = {
        "name",
        "studio.description",
        "studio.tags",
        "algorithm",
        "seed",
        "space.families",
        "space.side",
        "output_directory",
        "exact_budget",
        "candidate_budget",
        "cycles",
        "physics.hopping",
        "physics.chemical_potential",
        "physics.pairing",
        "physics.disorder_widths",
        "physics.disorder_seeds",
        "studio.objective_mode",
        "objective",
    }
    specs: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], prefix: str = "") -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and value:
                visit(value, path)
                continue
            if path.startswith("surrogate.") and resolved.algorithm != "surrogate_map_elites":
                continue
            if path.startswith("space.generator.") and not any(
                f in ("amorphous_planar", "constrained_embedded")
                for f in raw["space"].get("families", [])
            ):
                continue
            if (
                path
                in {
                    "space.min_degree",
                    "space.max_degree",
                    "space.max_bond_length",
                    "space.bond_tolerance",
                    "space.forbid_crossings",
                    "space.site_crossings",
                }
                and resolved.geometry_space == "embedded_sampling"
                and "rewired_square" not in raw["space"]["families"]
            ):
                continue
            if path == "space.rewires" and "rewired_square" not in raw["space"].get("families", []):
                continue
            kind = (
                "bool"
                if type(value) is bool
                else "int"
                if type(value) is int
                else (
                    "float"
                    if isinstance(value, float)
                    else "list"
                    if isinstance(value, list)
                    else "dict"
                    if isinstance(value, dict)
                    else "str"
                )
            )
            editable = path not in {
                "schema_version",
                "physics.adapter_id",
                "physics.model",
                "physics.basis",
                "physics.solver",
                "physics.probe_rule",
                "physics.success_threshold",
                "physics.finite_size",
                "concurrency",
            }
            editable = editable and not path.startswith(("studio.source.", "studio.definitions."))
            if path.startswith("studio.output.") and path != "studio.output.save_plots":
                editable = False
            if resolved.schema_version == 1 and path.startswith("physics."):
                editable = editable and path in {
                    "physics.disorder_widths",
                    "physics.disorder_seeds",
                    "physics.clean_seed",
                    "physics.confirmation",
                    "physics.validators",
                    "physics.finite_size",
                }
            spec = {
                "path": path,
                "label": key.replace("_", " ").capitalize(),
                "category": _category(path),
                "type": kind,
                "default": copy.deepcopy(value),
                "description": _DESCRIPTIONS.get(
                    path, f"Serialized {key.replace('_', ' ')} setting; backend validation applies."
                ),
                "expert": path not in common and not path.startswith("studio.variables."),
                "editable": editable,
                "searchable": path
                in {
                    "physics.hopping",
                    "physics.chemical_potential",
                    "physics.pairing",
                    "studio.variables.connectivity",
                    "studio.variables.coordinates",
                },
            }
            if path.startswith("studio.variables."):
                spec.update(
                    choices=["locked", "variable"],
                    description="Locked values are never changed by search; unsupported variability is rejected.",
                )
            elif path == "studio.objective_mode":
                spec["choices"] = ["none", "single"]
            elif path == "algorithm":
                spec["choices"] = (
                    ["random", "evolution", "map_elites", "surrogate_map_elites"]
                    if resolved.geometry_space == "fixed_connectivity"
                    else ["fixed_candidates"]
                    if resolved.geometry_space == "fixed_candidates"
                    else ["embedded_random"]
                )
            elif path == "geometry_space":
                spec["choices"] = ["embedded_sampling", "fixed_connectivity", "fixed_candidates"]
            elif path == "objective":
                from toposc_lab.research.physics import OBJECTIVE_REGISTRY

                spec["choices"] = list(OBJECTIVE_REGISTRY)
            if path.startswith("studio.definitions."):
                spec["description"] = (
                    "Validated immutable algorithm definition retained with this configuration."
                )
            if kind == "int":
                spec["min"] = 0 if "seed" in key or key == "retry_limit" else 1
                spec["max"] = 2147483647
            if path == "concurrency":
                spec.update(min=1, max=1)
            if path == "physics.chirality":
                spec.update(
                    min=-1,
                    max=1,
                    description="Directional p+ip chirality: +1 or -1; zero is rejected.",
                )
            if path in {
                "physics.tolerance",
                "physics.energy_cutoff",
                "physics.group_tolerance",
                "wall_seconds",
            }:
                spec["min"] = 1e-15
            specs.append(spec)

    visit(raw)
    return specs
