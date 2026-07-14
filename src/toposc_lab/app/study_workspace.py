"""Pure helper functions for the Streamlit study explorer.

Keeping compatibility checks out of the UI makes them testable and ensures
that a visual comparison is only offered for studies with the same scan axis.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from toposc_lab.data.studies import StudyData


def scan_parameter_name(studies: Mapping[str, StudyData]) -> str:
    """Return the shared scan parameter or reject incomparable studies."""
    if not studies:
        raise ValueError("at least one study is required")

    names = {
        str(study.metadata.scan_parameters.get("parameter_name", ""))
        for study in studies.values()
    }

    if "" in names:
        raise ValueError("every study needs metadata.scan_parameters['parameter_name']")

    if len(names) != 1:
        joined = ", ".join(sorted(names))
        raise ValueError(f"studies use different scan parameters: {joined}")

    return names.pop()


def common_scalar_observables(studies: Mapping[str, StudyData]) -> tuple[str, ...]:
    """List scalar scan observables that can be compared point by point."""
    if not studies:
        return ()

    common_names: set[str] | None = None

    for study in studies.values():
        parameter_values = np.asarray(study.arrays.get("parameter_values"))

        if parameter_values.ndim != 1:
            raise ValueError("each study needs one-dimensional parameter_values")

        names = {
            name
            for name, values in study.arrays.items()
            if name != "parameter_values"
            and np.asarray(values).ndim == 1
            and np.asarray(values).shape == parameter_values.shape
        }
        common_names = names if common_names is None else common_names & names

    return tuple(sorted(common_names or ()))


def study_summary(study: StudyData) -> dict[str, Any]:
    """Create a compact serializable description for the explorer table."""
    parameter_values = np.asarray(study.arrays.get("parameter_values", ()))
    return {
        "study": study.metadata.study_name,
        "model": study.metadata.model_name,
        "scan parameter": study.metadata.scan_parameters.get("parameter_name", "-"),
        "scan points": int(parameter_values.size),
        "created (UTC)": study.metadata.created_at.isoformat(),
        "arrays": ", ".join(sorted(study.arrays)),
    }
