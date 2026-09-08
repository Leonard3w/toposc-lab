"""Checked continuation from a completed-generation search checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TypeAlias

from toposc_lab.search._checkpoint_codec import decode_checkpoint, encode_checkpoint
from toposc_lab.search.checkpoint import SearchCheckpoint, _current_runtime_versions, _label
from toposc_lab.search.diversity_preservation import DiversityFamilyClassifier
from toposc_lab.search.generation_loop import (
    GenerationLoopResult,
    OffspringProducer,
    _continue_generation_loop,
)
from toposc_lab.search.population_fitness import PopulationEvaluator

SEARCH_RESUME_VERSION = 1
ResumeCheckpointCallback: TypeAlias = Callable[[SearchCheckpoint], None]


class IncompatibleSearchCheckpointError(ValueError):
    """Resume policy or runtime provenance differs from the stored experiment."""

    def __init__(self, mismatches: tuple[str, ...]) -> None:
        self.mismatches = tuple(mismatches)
        if not self.mismatches or any(
            not isinstance(item, str) or not item.strip() for item in self.mismatches
        ):
            raise ValueError("mismatches must contain nonempty messages")
        super().__init__("cannot resume checkpoint: " + "; ".join(self.mismatches))


def resume_search(
    checkpoint: SearchCheckpoint,
    *,
    evaluator: PopulationEvaluator,
    evaluator_identifier: str,
    offspring_producer: OffspringProducer,
    producer_identifier: str,
    code_version: str,
    family_classifier: DiversityFamilyClassifier | None = None,
    family_classifier_identifier: str | None = None,
    checkpoint_callback: ResumeCheckpointCallback | None = None,
) -> SearchCheckpoint:
    """Continue only unfinished generations under the checkpoint's frozen policy.

    The returned checkpoint contains the entire history and unchanged provenance
    and novelty reports. The optional callback receives a new SearchCheckpoint
    after each newly completed generation; the stored boundary is not emitted
    again. A completed checkpoint returns unchanged after compatibility checks.

    Identifiers are caller assertions about deterministic external policies,
    not code hashes. Matching recorded runtime versions cannot guarantee
    bitwise numerical equivalence across hardware or numerical backends.
    """
    if not isinstance(checkpoint, SearchCheckpoint):
        raise TypeError("checkpoint must be SearchCheckpoint")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not callable(offspring_producer):
        raise TypeError("offspring_producer must be callable")
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError("checkpoint_callback must be callable or None")
    mismatches: list[str] = []
    for name, provided, stored in (
        ("evaluator_identifier", evaluator_identifier, checkpoint.evaluator_identifier),
        ("producer_identifier", producer_identifier, checkpoint.result.producer_identifier),
        ("code_version", code_version, checkpoint.code_version),
    ):
        if _label(provided, name) != stored:
            mismatches.append(f"{name} differs from the checkpoint")
    if checkpoint.result.config.diversity is None:
        if family_classifier is not None or family_classifier_identifier is not None:
            mismatches.append("family classifier inputs require the stored diversity policy")
    else:
        if not callable(family_classifier):
            raise TypeError("stored diversity policy requires a callable family_classifier")
        if _label(family_classifier_identifier, "family_classifier_identifier") != (
            checkpoint.result.family_classifier_identifier
        ):
            mismatches.append("family_classifier_identifier differs from the checkpoint")
    current_runtime = _current_runtime_versions()
    for name in sorted(set(current_runtime) | set(checkpoint.runtime_versions)):
        if current_runtime.get(name) != checkpoint.runtime_versions.get(name):
            mismatches.append(f"runtime version {name!r} differs from the checkpoint")
    if mismatches:
        raise IncompatibleSearchCheckpointError(tuple(mismatches))

    # Revalidate the complete graph with the Phase-10.18 schema and constructors,
    # including existing RNG, identity, fitness, and derived-field checks. Keep
    # the caller's original prefix objects for the actual continuation.
    decode_checkpoint(encode_checkpoint(checkpoint))
    if checkpoint.completed_generation_index == checkpoint.requested_generation_count:
        return checkpoint

    def completed(prefix: GenerationLoopResult) -> None:
        assert checkpoint_callback is not None
        checkpoint_callback(replace(checkpoint, result=prefix))

    result = _continue_generation_loop(
        checkpoint.result,
        config=replace(
            checkpoint.result.config,
            generation_count=checkpoint.requested_generation_count,
        ),
        evaluator=evaluator,
        offspring_producer=offspring_producer,
        family_classifier=family_classifier,
        checkpoint_callback=completed if checkpoint_callback is not None else None,
    )
    return replace(checkpoint, result=result)
