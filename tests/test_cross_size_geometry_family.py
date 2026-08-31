from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import BUILTIN_GEOMETRY_GENERATORS, square
from toposc_lab.robustness import (
    GEOMETRY_FAMILY_CONTRACT_VERSION,
    CrossSizeGeometryFamily,
    DisorderEnsembleRequest,
    FiniteSizeRobustnessPoint,
    FiniteSizeScalingResult,
    FiniteSizeScalingSpec,
    GeometryFamilyMember,
    GeometryFamilySeedPolicy,
    GeometryFamilySpec,
    RobustnessFractionMetric,
    create_cross_size_geometry_family,
    estimate_robustness_uncertainty,
    fit_finite_size_scaling,
)


def _scaling(
    sizes: tuple[float, ...] = (2.0, 3.0, 4.0),
) -> FiniteSizeScalingResult:
    points = tuple(
        FiniteSizeRobustnessPoint(
            system_size=size,
            uncertainty=estimate_robustness_uncertainty(
                RobustnessFractionMetric(
                    criterion_key="fixed_family_success",
                    criterion_description="One fixed success policy for the family.",
                    request=DisorderEnsembleRequest(seeds=tuple(range(10))),
                    successes=(True,) * successful_count
                    + (False,) * (10 - successful_count),
                )
            ),
        )
        for size, successful_count in zip(sizes, (7, 8, 9), strict=True)
    )
    return fit_finite_size_scaling(
        FiniteSizeScalingSpec(
            size_key="linear_size",
            size_description="Explicit square side length.",
            correction_exponent=1.0,
        ),
        points=points,
    )


def _square_geometries(*, spacings: tuple[float, ...] = (1.0, 1.0, 1.0)):
    return tuple(
        BUILTIN_GEOMETRY_GENERATORS.generate(
            "square",
            parameters={"n_x": size, "n_y": size, "spacing": spacing},
        )
        for size, spacing in zip((2, 3, 4), spacings, strict=True)
    )


def _square_spec(
    *,
    size_key: str = "linear_size",
    generator_version: int = 1,
    seed_policy: GeometryFamilySeedPolicy = GeometryFamilySeedPolicy.NO_SEED,
) -> GeometryFamilySpec:
    return GeometryFamilySpec(
        family_key="square_open_family",
        family_version=1,
        description="Protocol-generated open square patches at increasing side length.",
        size_key=size_key,
        generator_key="square",
        generator_version=generator_version,
        varying_parameter_keys=("n_x", "n_y"),
        seed_policy=seed_policy,
    )


def _random_graph_geometries(*, seeds: tuple[int, int, int]):
    return tuple(
        BUILTIN_GEOMETRY_GENERATORS.generate(
            "random_graph",
            parameters={"n_sites": n_sites, "edge_probability": 0.25},
            seed=seed,
        )
        for n_sites, seed in zip((8, 10, 12), seeds, strict=True)
    )


def _random_graph_spec(seed_policy: GeometryFamilySeedPolicy) -> GeometryFamilySpec:
    return GeometryFamilySpec(
        family_key="random_graph_size_family",
        family_version=1,
        description="Seed-policy fixture for random graphs across site counts.",
        size_key="linear_size",
        generator_key="random_graph",
        generator_version=1,
        varying_parameter_keys=("n_sites",),
        seed_policy=seed_policy,
    )


def test_protocol_generated_family_retains_exact_construction_provenance() -> None:
    scaling = _scaling()
    geometries = _square_geometries()
    original_edges = tuple(geometry.edges for geometry in geometries)

    family = create_cross_size_geometry_family(
        _square_spec(),
        scaling=scaling,
        geometries=geometries,
    )

    assert isinstance(family, CrossSizeGeometryFamily)
    assert family.scaling is scaling
    assert family.contract_version == GEOMETRY_FAMILY_CONTRACT_VERSION
    assert family.system_sizes == (2.0, 3.0, 4.0)
    assert family.site_counts == (4, 9, 16)
    assert family.geometry_ids == tuple(map(exact_geometry_id, geometries))
    assert len(set(family.geometry_ids)) == 3
    assert family.generation_seeds == (None, None, None)
    assert family.fixed_generation_parameters == {"spacing": 1.0}
    assert family.varying_generation_parameters == (
        {"n_x": 2, "n_y": 2},
        {"n_x": 3, "n_y": 3},
        {"n_x": 4, "n_y": 4},
    )
    assert tuple(request.parameters for request in family.generation_requests) == (
        {"n_x": 2, "n_y": 2, "spacing": 1.0},
        {"n_x": 3, "n_y": 3, "spacing": 1.0},
        {"n_x": 4, "n_y": 4, "spacing": 1.0},
    )
    assert tuple(geometry.edges for geometry in geometries) == original_edges
    assert all(member.generator_key == "square" for member in family.members)
    assert all(member.generator_version == 1 for member in family.members)


@pytest.mark.parametrize(
    ("policy", "seeds"),
    [
        (GeometryFamilySeedPolicy.COMMON_SEED, (17, 17, 17)),
        (GeometryFamilySeedPolicy.DISTINCT_PER_SIZE, (17, 18, 19)),
    ],
)
def test_stochastic_family_requires_an_explicit_matching_seed_policy(
    policy: GeometryFamilySeedPolicy,
    seeds: tuple[int, int, int],
) -> None:
    family = create_cross_size_geometry_family(
        _random_graph_spec(policy),
        scaling=_scaling((8.0, 10.0, 12.0)),
        geometries=_random_graph_geometries(seeds=seeds),
    )

    assert family.generation_seeds == seeds


@pytest.mark.parametrize(
    ("policy", "seeds"),
    [
        (GeometryFamilySeedPolicy.NO_SEED, (17, 18, 19)),
        (GeometryFamilySeedPolicy.COMMON_SEED, (17, 18, 19)),
        (GeometryFamilySeedPolicy.DISTINCT_PER_SIZE, (17, 17, 17)),
    ],
)
def test_seed_policy_mismatches_are_rejected(
    policy: GeometryFamilySeedPolicy,
    seeds: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="declared seed policy"):
        create_cross_size_geometry_family(
            _random_graph_spec(policy),
            scaling=_scaling((8.0, 10.0, 12.0)),
            geometries=_random_graph_geometries(seeds=seeds),
        )


def test_family_requires_matching_size_key() -> None:
    with pytest.raises(ValueError, match="size keys must match"):
        create_cross_size_geometry_family(
            _square_spec(size_key="site_count"),
            scaling=_scaling(),
            geometries=_square_geometries(),
        )


def test_family_requires_declared_generator_key_and_version() -> None:
    with pytest.raises(ValueError, match="generator key and version"):
        create_cross_size_geometry_family(
            _square_spec(generator_version=2),
            scaling=_scaling(),
            geometries=_square_geometries(),
        )

    mixed = (*_square_geometries()[:2], BUILTIN_GEOMETRY_GENERATORS.generate(
        "chain", parameters={"n_sites": 4, "spacing": 1.0}
    ))
    with pytest.raises(ValueError, match="generator key and version"):
        create_cross_size_geometry_family(
            _square_spec(),
            scaling=_scaling(),
            geometries=mixed,
        )


def test_non_varying_generator_parameters_must_remain_fixed() -> None:
    with pytest.raises(ValueError, match="non-varying generator parameters"):
        create_cross_size_geometry_family(
            _square_spec(),
            scaling=_scaling(),
            geometries=_square_geometries(spacings=(1.0, 1.0, 2.0)),
        )


def test_declared_varying_parameters_must_identify_sizes_uniquely() -> None:
    geometries = tuple(
        BUILTIN_GEOMETRY_GENERATORS.generate(
            "random_graph",
            parameters={"n_sites": 8, "edge_probability": 0.25},
            seed=seed,
        )
        for seed in (17, 18, 19)
    )

    with pytest.raises(ValueError, match="identify each size uniquely"):
        create_cross_size_geometry_family(
            _random_graph_spec(GeometryFamilySeedPolicy.DISTINCT_PER_SIZE),
            scaling=_scaling((8.0, 10.0, 12.0)),
            geometries=geometries,
        )


def test_exact_duplicate_snapshots_are_rejected_without_graph_hashing() -> None:
    geometry = _square_geometries()[0]
    with pytest.raises(ValueError, match="distinct exact geometry snapshots"):
        create_cross_size_geometry_family(
            _square_spec(),
            scaling=_scaling(),
            geometries=(geometry, geometry, geometry),
        )


def test_direct_geometry_without_phase_6_provenance_is_rejected() -> None:
    with pytest.raises(ValueError, match="Phase-6 generation metadata"):
        create_cross_size_geometry_family(
            _square_spec(),
            scaling=_scaling(),
            geometries=(square(2, 2), square(3, 3), square(4, 4)),
        )


def test_family_requires_one_geometry_per_scaling_point() -> None:
    with pytest.raises(ValueError, match="one value per scaling point"):
        create_cross_size_geometry_family(
            _square_spec(),
            scaling=_scaling(),
            geometries=_square_geometries()[:2],
        )


@pytest.mark.parametrize("family_key", ["", "SquareFamily", "square-family"])
def test_family_spec_requires_a_stable_technical_key(family_key: str) -> None:
    with pytest.raises(ValueError, match="family_key"):
        GeometryFamilySpec(
            family_key=family_key,
            family_version=1,
            description="Valid description.",
            size_key="linear_size",
            generator_key="square",
            generator_version=1,
            varying_parameter_keys=("n_x", "n_y"),
            seed_policy=GeometryFamilySeedPolicy.NO_SEED,
        )


def test_family_contracts_are_immutable_and_validate_types() -> None:
    family = create_cross_size_geometry_family(
        _square_spec(),
        scaling=_scaling(),
        geometries=_square_geometries(),
    )
    with pytest.raises(FrozenInstanceError):
        family.members = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="seed_policy must be"):
        GeometryFamilySpec(
            family_key="square_open_family",
            family_version=1,
            description="Valid description.",
            size_key="linear_size",
            generator_key="square",
            generator_version=1,
            varying_parameter_keys=("n_x", "n_y"),
            seed_policy="no_seed",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="geometry must be"):
        GeometryFamilyMember(
            geometry=None,  # type: ignore[arg-type]
            scaling_point=_scaling().points[0],
        )
