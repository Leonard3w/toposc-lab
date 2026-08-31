from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.robustness import (
    DISORDER_RNG_ALGORITHM,
    HAMILTONIAN_ID_SCHEME,
    DisorderParameterValue,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    exact_hamiltonian_id,
    realize_disorder,
)


def _diagonal_noise(
    source: DisorderState,
    parameters: Mapping[str, DisorderParameterValue],
    rng: np.random.Generator,
) -> DisorderState:
    assert isinstance(source, np.ndarray)
    scale = parameters["scale"]
    assert isinstance(scale, float)
    noise = np.diag(rng.normal(scale=scale, size=source.shape[0]))
    return np.asarray(source, dtype=float) + noise


def _identity_geometry(
    source: DisorderState,
    parameters: Mapping[str, DisorderParameterValue],
    rng: np.random.Generator,
) -> DisorderState:
    del parameters, rng
    assert isinstance(source, Geometry)
    return source


def _hamiltonian_transform() -> FunctionDisorderTransform:
    return FunctionDisorderTransform(
        key="neutral_diagonal_test_noise",
        target=DisorderTarget.HAMILTONIAN,
        function=_diagonal_noise,
    )


def test_request_requires_explicit_nonnegative_integer_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        DisorderRequest(seed=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed must be an integer"):
        DisorderRequest(seed=True)
    with pytest.raises(ValueError, match="non-negative"):
        DisorderRequest(seed=-1)


def test_request_parameters_are_validated_sorted_and_deeply_frozen() -> None:
    nested = {"values": [2, 3]}
    request = DisorderRequest(
        seed=np.int64(17),  # type: ignore[arg-type]
        parameters={"scale": 0.25, "nested": nested},  # type: ignore[dict-item]
    )
    nested["values"].append(4)

    assert request.seed == 17
    assert tuple(request.parameters) == ("nested", "scale")
    assert request.parameters["nested"] == {"values": (2, 3)}
    with pytest.raises(TypeError):
        request.parameters["scale"] = 1.0  # type: ignore[index]


def test_same_request_reproduces_state_and_complete_audit_record() -> None:
    source = np.asarray([[1.0, 0.2], [0.2, -1.0]])
    request = DisorderRequest(seed=1729, parameters={"scale": 0.1})

    first = realize_disorder(
        source,
        transform=_hamiltonian_transform(),
        request=request,
    )
    second = realize_disorder(
        source,
        transform=_hamiltonian_transform(),
        request=request,
    )

    assert isinstance(first.state, np.ndarray)
    assert isinstance(second.state, np.ndarray)
    assert np.array_equal(first.state, second.state)
    assert first.provenance == second.provenance
    assert first.provenance.disorder_key == "neutral_diagonal_test_noise"
    assert first.provenance.disorder_version == 1
    assert first.provenance.seed == 1729
    assert first.provenance.rng_algorithm == DISORDER_RNG_ALGORITHM
    assert first.provenance.parameters == {"scale": 0.1}
    assert first.provenance.source.identifier == exact_hamiltonian_id(source)
    assert first.provenance.source.scheme == HAMILTONIAN_ID_SCHEME
    assert first.provenance.result.identifier == exact_hamiltonian_id(first.state)
    assert first.provenance.result.identifier != first.provenance.source.identifier


def test_different_seeds_change_the_realization() -> None:
    source = np.eye(4)
    first = realize_disorder(
        source,
        transform=_hamiltonian_transform(),
        request=DisorderRequest(seed=1, parameters={"scale": 0.2}),
    )
    second = realize_disorder(
        source,
        transform=_hamiltonian_transform(),
        request=DisorderRequest(seed=2, parameters={"scale": 0.2}),
    )

    assert isinstance(first.state, np.ndarray)
    assert isinstance(second.state, np.ndarray)
    assert not np.array_equal(first.state, second.state)


def test_execution_does_not_read_or_modify_global_numpy_random_state() -> None:
    source = np.eye(3)
    np.random.seed(9182)
    expected = np.random.random(4)
    np.random.seed(9182)

    realize_disorder(
        source,
        transform=_hamiltonian_transform(),
        request=DisorderRequest(seed=7, parameters={"scale": 0.2}),
    )

    assert np.array_equal(np.random.random(4), expected)


def test_hamiltonian_input_is_not_mutated_and_result_is_read_only() -> None:
    source = np.eye(3)
    clean_copy = source.copy()
    realization = realize_disorder(
        source,
        transform=_hamiltonian_transform(),
        request=DisorderRequest(seed=5, parameters={"scale": 0.2}),
    )

    assert isinstance(realization.state, np.ndarray)
    assert np.array_equal(source, clean_copy)
    assert not realization.state.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        realization.state[0, 0] = 0.0


def test_geometry_snapshots_reuse_exact_phase_7_11_identity_and_orientation() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(1, 0, edge_type="oriented_test"),),
    )
    realization = realize_disorder(
        geometry,
        transform=FunctionDisorderTransform(
            key="neutral_geometry_identity",
            target=DisorderTarget.GEOMETRY,
            function=_identity_geometry,
        ),
        request=DisorderRequest(seed=11),
    )

    assert isinstance(realization.state, Geometry)
    assert realization.state.edges[0].source == 1
    assert realization.state.edges[0].target == 0
    assert realization.provenance.source == realization.provenance.result
    assert realization.provenance.source.identifier.startswith(
        "toposc-geometry-archive-v1-sha256:"
    )


def test_transform_target_and_hamiltonian_shape_are_enforced() -> None:
    geometry_transform = FunctionDisorderTransform(
        key="neutral_geometry_identity",
        target=DisorderTarget.GEOMETRY,
        function=_identity_geometry,
    )
    with pytest.raises(TypeError, match="requires and returns Geometry"):
        realize_disorder(
            np.eye(2),
            transform=geometry_transform,
            request=DisorderRequest(seed=1),
        )

    def wrong_shape(
        source: DisorderState,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        del source, parameters, rng
        return np.eye(3)

    with pytest.raises(ValueError, match="preserve the matrix shape"):
        realize_disorder(
            np.eye(2),
            transform=FunctionDisorderTransform(
                key="wrong_shape",
                target=DisorderTarget.HAMILTONIAN,
                function=wrong_shape,
            ),
            request=DisorderRequest(seed=1),
        )


@pytest.mark.parametrize("key", ["", "UPPER", "contains-hyphen"])
def test_transform_rejects_unstable_keys(key: str) -> None:
    with pytest.raises(ValueError, match="disorder key"):
        FunctionDisorderTransform(
            key=key,
            target=DisorderTarget.GEOMETRY,
            function=_identity_geometry,
        )


def test_hamiltonian_snapshot_distinguishes_dtype_and_shape() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    assert exact_hamiltonian_id(values) != exact_hamiltonian_id(
        values.astype(np.complex128)
    )
    assert exact_hamiltonian_id(values) != exact_hamiltonian_id(
        np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    )
