"""Neutral visualization of explicitly ranked Phase-9 candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toposc_lab.search import CandidateRankingEntry, CandidateRankingResult
from toposc_lab.visualization.geometry_plots import plot_geometry
from toposc_lab.visualization.style import paper_style

_VISUALIZATION_WARNINGS = (
    "Displayed ranks are an engineering ordering, not evidence of topology, "
    "robustness, novelty, or scientific discovery.",
    "The maximum displayed rank controls presentation only and is not a scientific "
    "eligibility or success threshold.",
    "Circular layouts for geometries without coordinates are schematic; x-y views of "
    "higher-dimensional coordinates are projections.",
)


@dataclass(frozen=True, slots=True)
class TopCandidateVisualization:
    """Figure and exact ranked entries displayed by one visualization call."""

    figure: Figure
    axes: tuple[Axes, ...]
    entries: tuple[CandidateRankingEntry, ...]
    maximum_rank: int
    columns: int
    warnings: tuple[str, ...] = field(default=_VISUALIZATION_WARNINGS, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.figure, Figure):
            raise TypeError("figure must be a matplotlib Figure")
        axes = tuple(self.axes)
        entries = tuple(self.entries)
        if not axes or not all(isinstance(item, Axes) for item in axes):
            raise TypeError("axes must contain matplotlib Axes values")
        if not entries or not all(isinstance(item, CandidateRankingEntry) for item in entries):
            raise TypeError("entries must contain ranked CandidateRankingEntry values")
        if len(axes) != len(entries):
            raise ValueError("axes and entries must have equal length")
        if any(entry.rank is None for entry in entries):
            raise ValueError("visualized entries must be ranked")
        maximum_rank = _positive_integer(self.maximum_rank, name="maximum_rank")
        columns = _positive_integer(self.columns, name="columns")
        if any(entry.rank is None or entry.rank > maximum_rank for entry in entries):
            raise ValueError("entries must not exceed maximum_rank")
        if columns > len(entries):
            raise ValueError("columns must not exceed the number of displayed entries")

        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "maximum_rank", maximum_rank)
        object.__setattr__(self, "columns", columns)


def plot_top_ranked_candidates(
    ranking: CandidateRankingResult,
    *,
    maximum_rank: int,
    columns: int = 3,
    show_site_indices: bool = False,
    show_boundary_sites: bool = True,
    show_periodic_edges: bool = False,
    show_edge_orientation: bool = True,
    site_size: float = 48.0,
    show: bool = True,
) -> TopCandidateVisualization:
    """Plot all ranked candidates at or above the explicit rank cutoff.

    Competition-rank ties are complete because selection uses ``rank <=
    maximum_rank`` rather than a candidate count. Candidate geometry and ranking
    records are only read; no figure is saved and no scientific label is inferred.
    """
    if not isinstance(ranking, CandidateRankingResult):
        raise TypeError("ranking must be CandidateRankingResult")
    prepared_maximum_rank = _positive_integer(maximum_rank, name="maximum_rank")
    requested_columns = _positive_integer(columns, name="columns")
    entries = tuple(
        entry
        for entry in ranking.ranked_entries
        if entry.rank is not None and entry.rank <= prepared_maximum_rank
    )
    if not entries:
        raise ValueError("ranking has no candidates within maximum_rank")
    display_columns = min(requested_columns, len(entries))
    rows = math.ceil(len(entries) / display_columns)

    with paper_style():
        figure, axes_grid = plt.subplots(
            rows,
            display_columns,
            figsize=(5.2 * display_columns, 4.4 * rows),
            constrained_layout=True,
            squeeze=False,
        )
        figure.suptitle("Top-ranked candidates — engineering visualization")

    all_axes = tuple(
        cast(Axes, axes_grid[row, column])
        for row in range(rows)
        for column in range(display_columns)
    )
    displayed_axes = all_axes[: len(entries)]
    try:
        for axes, entry in zip(displayed_axes, entries, strict=True):
            plot_geometry(
                entry.member.candidate.geometry,
                axes=axes,
                title=_candidate_title(entry, ranking=ranking),
                show_site_indices=show_site_indices,
                show_boundary_sites=show_boundary_sites,
                show_periodic_edges=show_periodic_edges,
                show_edge_orientation=show_edge_orientation,
                site_size=site_size,
                show=False,
            )
        for axes in all_axes[len(entries) :]:
            axes.set_axis_off()
    except BaseException:
        plt.close(figure)
        raise

    if show:
        plt.show()

    return TopCandidateVisualization(
        figure=figure,
        axes=displayed_axes,
        entries=entries,
        maximum_rank=prepared_maximum_rank,
        columns=display_columns,
    )


def _candidate_title(
    entry: CandidateRankingEntry,
    *,
    ranking: CandidateRankingResult,
) -> str:
    assert entry.rank is not None
    values = "; ".join(
        f"{criterion.name}={_format_value(entry.values[criterion.name])}"
        for criterion in ranking.config.criteria
    )
    geometry = entry.member.candidate.geometry
    if geometry.coordinates is None:
        layout_note = "schematic circular layout; no coordinates"
    elif geometry.coordinates.shape[1] > 2:
        layout_note = "x-y projection"
    else:
        layout_note = "stored coordinates"
    return f"Rank {entry.rank} · candidate {entry.candidate_index}\n{values}\n[{layout_note}]"


def _format_value(value: bool | int | float) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return format(value, ".6g")


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result
