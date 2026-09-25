"""Bounded random-family study using the existing fixed-cohort execution loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.embedded import EMBEDDED_ADAPTER_ID, EmbeddedValidationEvaluator
from toposc_lab.research.embedded_cohort import (
    CohortPreparationError,
    prepare_cohort,
    validate_cohort,
)
from toposc_lab.research.storage import atomic_text, dumps
from toposc_lab.research.validation import ValidationStudy

VERSION = "phase19.embedded-exploration.v1"


def settings_for(cohort: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "mode": "exploratory",
        "cohort_sha256": cohort["sha256"],
        "widths": [3.0, 6.0, 9.0],
        "seeds": [191001, 191002, 191003],
        "clean_seed": 191000,
        "exact_budget": 1600,
        "wall_seconds": 1800.0,
        "retry_limit": 1,
        "checkpoint_every": 20,
        "protocol_document": "docs/decisions/phase19_exploratory_protocol.md",
    }


def make_config(settings: dict[str, Any], cohort: dict[str, Any]) -> ExperimentConfig:
    validate_cohort(cohort)
    if settings.get("version") != VERSION or settings.get("mode") != "exploratory":
        raise ValueError("Unsupported embedded study")
    if settings.get("cohort_sha256") != cohort["sha256"]:
        raise ValueError("Cohort checksum mismatch")
    if any(w <= 0 for w in settings["widths"]):
        raise ValueError("Separate single clean stage required")
    if settings["exact_budget"] > 1600 or settings["wall_seconds"] > 1800:
        raise ValueError("This exploratory study is capped at 1600 attempts / 1800 seconds")
    return ExperimentConfig(
        name="Phase 19 embedded exploration",
        algorithm="random",
        exact_budget=settings["exact_budget"],
        wall_seconds=settings["wall_seconds"],
        retry_limit=settings["retry_limit"],
        checkpoint_every=settings["checkpoint_every"],
        objective="robustness_quality_mean",
        geometry_space="embedded_fixed_cohort",
        space={"domain": cohort["domain"], "policy": cohort["policy"]},
        baselines=(),
        physics={
            "adapter_id": EMBEDDED_ADAPTER_ID,
            "domain": cohort["domain"],
            "disorder_widths": settings["widths"],
            "disorder_seeds": settings["seeds"],
            "clean_seed": settings["clean_seed"],
            "confirmation": False,
        },
    )


class EmbeddedStudy(ValidationStudy):
    configure = staticmethod(make_config)
    study_version = VERSION
    build_evaluator = staticmethod(EmbeddedValidationEvaluator)

    def _stage(self, candidate: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any] | None:
        result = super()._stage(candidate, stage)
        if result is not None and (
            result.get("status") != "completed" or not result.get("primary_valid")
        ):
            raise RuntimeError(
                "Embedded study stopped: failed numerical/primary validation; inspect saved evidence"
            )
        return result

    def _report(self) -> None:
        # Raw export remains a generic operation, with bounded memory usage.
        from toposc_lab.research.embedded_reporting import export_study

        report = export_study(self.directory)
        self.store.save("report", {"markdown": report})
        self._hook("report_saved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "create", "run", "report"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--max-stages", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.directory.exists():
            raise FileExistsError("Refuse overwriting prepared cohort")
        try:
            cohort = prepare_cohort(per_family=args.per_family)
        except CohortPreparationError as error:
            atomic_text(args.directory.with_suffix(".rejected.json"), dumps(error.proposals))
            raise
        atomic_text(args.directory, json.dumps(cohort, indent=2))
        print(
            dumps(
                {
                    "cohort": str(args.directory),
                    "candidates": len(cohort["candidates"]),
                    "sha256": cohort["sha256"],
                }
            )
        )
    elif args.action == "create":
        if args.cohort is None:
            parser.error("--cohort is required")
        cohort = json.loads(args.cohort.read_text())
        print(EmbeddedStudy.create_study(args.directory, settings_for(cohort), cohort))
    elif args.action == "run":
        print(dumps(EmbeddedStudy(args.directory).run(max_stages=args.max_stages)))
    else:
        from toposc_lab.research.embedded_reporting import export_study

        export_study(args.directory)


if __name__ == "__main__":
    main()
