"""Reproducible independent removal and reindexing of geometry sites."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryEdge,
    GeometryFace,
)
from toposc_lab.robustness.disorder import (
    DisorderParameterValue,
    DisorderProvenance,
    DisorderRealization,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    realize_disorder,
)

RANDOM_NODE_REMOVAL_KEY = "random_node_removal"
RANDOM_NODE_REMOVAL_VERSION = 1


@dataclass(frozen=True, slots=True)
class NodeRemovalRealization:
    """A geometry realization plus its explicit original-site reindexing."""

    realization: DisorderRealization
    surviving_sites: tuple[int, ...]
    removed_sites: tuple[int, ...]
    old_to_new: Mapping[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.realization, DisorderRealization):
            raise TypeError("realization must be DisorderRealization")
        if not isinstance(self.realization.state, Geometry):
            raise TypeError("node removal realization must contain a Geometry")

        surviving_sites = _site_tuple(
            self.surviving_sites,
            name="surviving_sites",
        )
        removed_sites = _site_tuple(self.removed_sites, name="removed_sites")
        if not surviving_sites:
            raise ValueError("at least one site must survive node removal")

        input_site_count = self.realization.provenance.parameters.get(
            "input_site_count"
        )
        if isinstance(input_site_count, bool) or not isinstance(
            input_site_count,
            Integral,
        ):
            raise ValueError("provenance must record an integer input_site_count")
        expected_sites = tuple(range(int(input_site_count)))
        if tuple(sorted(surviving_sites + removed_sites)) != expected_sites:
            raise ValueError(
                "surviving_sites and removed_sites must partition the original sites"
            )

        if not isinstance(self.old_to_new, Mapping):
            raise TypeError("old_to_new must be a mapping")
        old_to_new = {
            _site_index(old, name="old_to_new key"): _site_index(
                new,
                name="old_to_new value",
            )
            for old, new in self.old_to_new.items()
        }
        expected_mapping = {
            old_site: new_site
            for new_site, old_site in enumerate(surviving_sites)
        }
        if old_to_new != expected_mapping:
            raise ValueError(
                "old_to_new must map surviving sites to contiguous indices in order"
            )
        if self.realization.state.n_sites != len(surviving_sites):
            raise ValueError("result geometry size does not match surviving_sites")

        object.__setattr__(self, "surviving_sites", surviving_sites)
        object.__setattr__(self, "removed_sites", removed_sites)
        object.__setattr__(self, "old_to_new", MappingProxyType(old_to_new))

    @property
    def state(self) -> Geometry:
        """Return the reindexed geometry from the common disorder realization."""
        state = self.realization.state
        assert isinstance(state, Geometry)
        return state

    @property
    def provenance(self) -> DisorderProvenance:
        """Return the common immutable disorder provenance."""
        return self.realization.provenance


def apply_random_node_removal(
    geometry: Geometry,
    *,
    removal_probability: float,
    seed: int,
) -> NodeRemovalRealization:
    """Remove sites independently and return their explicit compact reindexing."""
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    probability = _unit_interval_real(
        removal_probability,
        name="removal_probability",
    )
    request = DisorderRequest(
        seed=seed,
        parameters={
            "removal_probability": probability,
            "geometry_id": exact_geometry_id(geometry),
            "input_site_count": geometry.n_sites,
            "input_edge_count": geometry.n_edges,
            "sampling_rule": "independent_bernoulli_remove_if_draw_below_probability",
            "site_iteration_order": "ascending_original_site_index",
            "minimum_survivor_policy": "retain_largest_draw_then_lowest_site_on_tie",
            "reindexing_rule": "surviving_original_order_to_contiguous_indices",
            "edge_policy": "retain_if_both_endpoints_survive_and_preserve_orientation",
            "face_policy": "retain_if_all_sites_survive_and_preserve_order",
            "boundary_policy": "remap_survivors_and_drop_empty_components",
            "rooted_tree_policy": "clear_if_any_site_is_removed",
        },
    )
    sampled_outcome: list[
        tuple[Geometry, tuple[int, ...], tuple[int, ...], Mapping[int, int]]
    ] = []

    def transform(
        source: DisorderState,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        assert isinstance(source, Geometry)
        parameter_probability = parameters["removal_probability"]
        if isinstance(parameter_probability, bool) or not isinstance(
            parameter_probability,
            Real,
        ):
            raise TypeError("recorded removal_probability must be a real number")
        outcome = _remove_sampled_sites(
            source,
            removal_probability=float(parameter_probability),
            rng=rng,
        )
        sampled_outcome.append(outcome)
        return outcome[0]

    disorder_transform = FunctionDisorderTransform(
        key=RANDOM_NODE_REMOVAL_KEY,
        version=RANDOM_NODE_REMOVAL_VERSION,
        target=DisorderTarget.GEOMETRY,
        function=transform,
    )
    realization = realize_disorder(
        geometry,
        transform=disorder_transform,
        request=request,
    )
    if len(sampled_outcome) != 1:
        raise RuntimeError("node removal transform must run exactly once")
    _, surviving_sites, removed_sites, old_to_new = sampled_outcome[0]
    return NodeRemovalRealization(
        realization=realization,
        surviving_sites=surviving_sites,
        removed_sites=removed_sites,
        old_to_new=old_to_new,
    )


def _remove_sampled_sites(
    geometry: Geometry,
    *,
    removal_probability: float,
    rng: np.random.Generator,
) -> tuple[Geometry, tuple[int, ...], tuple[int, ...], Mapping[int, int]]:
    draws = rng.random(geometry.n_sites)
    surviving_sites = tuple(
        site
        for site, draw in enumerate(draws)
        if draw >= removal_probability
    )
    if not surviving_sites:
        largest_draw_site = max(
            range(geometry.n_sites),
            key=lambda site: (float(draws[site]), -site),
        )
        surviving_sites = (largest_draw_site,)
    surviving_sites = tuple(sorted(surviving_sites))
    surviving_set = frozenset(surviving_sites)
    removed_sites = tuple(
        site for site in range(geometry.n_sites) if site not in surviving_set
    )
    old_to_new = {
        old_site: new_site
        for new_site, old_site in enumerate(surviving_sites)
    }
    if not removed_sites:
        return geometry, surviving_sites, removed_sites, old_to_new

    edges = tuple(
        GeometryEdge(
            source=old_to_new[edge.source],
            target=old_to_new[edge.target],
            edge_type=edge.edge_type,
            boundary_crossing=edge.boundary_crossing,
            displacement=edge.displacement,
            metadata=edge.metadata,
        )
        for edge in geometry.edges
        if edge.source in surviving_set and edge.target in surviving_set
    )
    coordinates = (
        None
        if geometry.coordinates is None
        else geometry.coordinates[np.asarray(surviving_sites, dtype=np.intp)]
    )
    boundary_sites = frozenset(
        old_to_new[site]
        for site in geometry.boundary_sites
        if site in surviving_set
    )
    boundary_components = tuple(
        GeometryBoundaryComponent(
            kind=component.kind,
            component_index=component.component_index,
            sites=frozenset(
                old_to_new[site]
                for site in component.sites
                if site in surviving_set
            ),
        )
        for component in geometry.boundary_components
        if component.sites & surviving_set
    )
    site_types = (
        None
        if geometry.site_types is None
        else tuple(geometry.site_types[site] for site in surviving_sites)
    )
    faces = tuple(
        GeometryFace(
            sites=tuple(old_to_new[site] for site in face.sites),
            face_type=face.face_type,
            metadata=face.metadata,
        )
        for face in geometry.faces
        if all(site in surviving_set for site in face.sites)
    )
    result = Geometry(
        n_sites=len(surviving_sites),
        edges=edges,
        coordinates=coordinates,
        embedding_dimension=geometry.embedding_dimension,
        boundary_sites=boundary_sites,
        boundary_components=boundary_components,
        site_types=site_types,
        dimension_records=geometry.dimension_records,
        rooted_tree=None,
        metadata=geometry.metadata,
        faces=faces,
    )
    return result, surviving_sites, removed_sites, old_to_new


def _site_tuple(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    normalized = tuple(_site_index(value, name=name) for value in values)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{name} must contain unique sites in ascending order")
    return normalized


def _site_index(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must contain only integers")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must contain only non-negative sites")
    return result


def _unit_interval_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result
