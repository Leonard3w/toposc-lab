"""Concrete uniform disorder for explicit superconducting pairing channels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry
from toposc_lab.hamiltonians.disorder import (
    sample_uniform_edge_disorder,
    sample_uniform_site_disorder,
)
from toposc_lab.hamiltonians.nambu import NambuBasis
from toposc_lab.hamiltonians.pairing import (
    build_chiral_p_wave_pairing,
    build_d_wave_pairing,
    build_onsite_s_wave_pairing,
    build_spinless_p_wave_pairing,
)
from toposc_lab.robustness._matrix_disorder import nonnegative_finite_real
from toposc_lab.robustness.disorder import (
    DisorderParameterValue,
    DisorderRealization,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    realize_disorder,
)

UNIFORM_PAIRING_DISORDER_KEY = "uniform_pairing_disorder"
UNIFORM_PAIRING_DISORDER_VERSION = 1


class PairingDisorderChannel(str, Enum):
    """Explicit existing pairing convention receiving amplitude disorder."""

    SPINLESS_P_WAVE = "spinless_p_wave"
    ONSITE_S_WAVE = "onsite_s_wave"
    CHIRAL_P_WAVE = "chiral_p_wave"
    D_WAVE = "d_wave"


@dataclass(frozen=True, slots=True)
class _PairingChannelConfig:
    channel: PairingDisorderChannel
    support: str
    chirality: int | None
    plane_axes: tuple[int, int] | None


def apply_uniform_pairing_disorder(
    geometry: Geometry,
    hamiltonian: np.ndarray,
    *,
    width: float,
    seed: int,
    nambu_basis: NambuBasis,
    channel: PairingDisorderChannel,
    chirality: int | None = None,
    plane_axes: tuple[int, int] | None = None,
) -> DisorderRealization:
    r"""Add uniform amplitude disorder in one explicit pairing channel.

    The selected existing pairing builder converts sampled site or oriented-
    edge amplitudes into an antisymmetric normal-state ``Delta`` block. The
    block is then added to the supplied full BdG Hamiltonian using its explicit
    ``NambuBasis`` convention.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    if not isinstance(nambu_basis, NambuBasis):
        raise TypeError("nambu_basis must be NambuBasis")
    if nambu_basis.n_sites != geometry.n_sites:
        raise ValueError("nambu_basis site count must match geometry")
    width = nonnegative_finite_real(width, name="width")
    config = _resolve_channel_config(
        geometry,
        nambu_basis=nambu_basis,
        channel=channel,
        chirality=chirality,
        plane_axes=plane_axes,
    )
    request = DisorderRequest(
        seed=seed,
        parameters={
            "distribution": "uniform",
            "width": width,
            "geometry_id": exact_geometry_id(geometry),
            "n_sites": geometry.n_sites,
            "n_edges": geometry.n_edges,
            "channel": config.channel.value,
            "support": config.support,
            "normal_components_per_site": (
                nambu_basis.normal_components_per_site
            ),
            "basis_ordering": nambu_basis.ordering,
            "chirality": config.chirality,
            "plane_axes": config.plane_axes,
            "pairing_embedding": (
                "upper_delta_lower_negative_conjugate"
            ),
        },
    )

    def transform(
        source: DisorderState,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        assert isinstance(source, np.ndarray)
        parameter_width = parameters["width"]
        if isinstance(parameter_width, bool) or not isinstance(parameter_width, Real):
            raise TypeError("recorded pairing width must be a real number")
        pairing_block = _sample_pairing_block(
            geometry,
            width=float(parameter_width),
            rng=rng,
            config=config,
        )
        return _apply_pairing_block(
            source,
            pairing_block,
            basis=nambu_basis,
        )

    disorder_transform = FunctionDisorderTransform(
        key=UNIFORM_PAIRING_DISORDER_KEY,
        version=UNIFORM_PAIRING_DISORDER_VERSION,
        target=DisorderTarget.HAMILTONIAN,
        function=transform,
    )
    return realize_disorder(
        hamiltonian,
        transform=disorder_transform,
        request=request,
    )


def _resolve_channel_config(
    geometry: Geometry,
    *,
    nambu_basis: NambuBasis,
    channel: PairingDisorderChannel,
    chirality: int | None,
    plane_axes: tuple[int, int] | None,
) -> _PairingChannelConfig:
    if not isinstance(channel, PairingDisorderChannel):
        raise TypeError("channel must be PairingDisorderChannel")
    expected_components = (
        2
        if channel in {
            PairingDisorderChannel.ONSITE_S_WAVE,
            PairingDisorderChannel.D_WAVE,
        }
        else 1
    )
    if nambu_basis.normal_components_per_site != expected_components:
        raise ValueError(
            f"{channel.value} pairing disorder requires exactly "
            f"{expected_components} normal component(s) per site"
        )

    if channel is PairingDisorderChannel.CHIRAL_P_WAVE:
        if chirality is None or plane_axes is None:
            raise ValueError("chiral p-wave disorder requires chirality and plane_axes")
        zero_offsets = {edge: 0.0 for edge in geometry.edges}
        build_chiral_p_wave_pairing(
            geometry,
            pairing=zero_offsets,
            chirality=chirality,
            plane_axes=plane_axes,
        )
        return _PairingChannelConfig(
            channel=channel,
            support="edge",
            chirality=int(chirality),
            plane_axes=(int(plane_axes[0]), int(plane_axes[1])),
        )

    if channel is PairingDisorderChannel.D_WAVE:
        if chirality is not None:
            raise ValueError("d-wave disorder does not accept chirality")
        if plane_axes is None:
            raise ValueError("d-wave disorder requires plane_axes")
        zero_offsets = {edge: 0.0 for edge in geometry.edges}
        build_d_wave_pairing(
            geometry,
            pairing=zero_offsets,
            plane_axes=plane_axes,
        )
        return _PairingChannelConfig(
            channel=channel,
            support="edge",
            chirality=None,
            plane_axes=(int(plane_axes[0]), int(plane_axes[1])),
        )

    if chirality is not None or plane_axes is not None:
        raise ValueError(
            f"{channel.value} pairing disorder does not accept chirality or plane_axes"
        )
    if channel is PairingDisorderChannel.ONSITE_S_WAVE:
        build_onsite_s_wave_pairing(
            geometry,
            pairing={site: 0.0 for site in geometry.site_indices},
        )
        return _PairingChannelConfig(
            channel=channel,
            support="site",
            chirality=None,
            plane_axes=None,
        )
    build_spinless_p_wave_pairing(
        geometry,
        pairing={edge: 0.0 for edge in geometry.edges},
    )
    return _PairingChannelConfig(
        channel=channel,
        support="edge",
        chirality=None,
        plane_axes=None,
    )


def _sample_pairing_block(
    geometry: Geometry,
    *,
    width: float,
    rng: np.random.Generator,
    config: _PairingChannelConfig,
) -> np.ndarray:
    if config.channel is PairingDisorderChannel.ONSITE_S_WAVE:
        site_offsets = sample_uniform_site_disorder(
            geometry,
            width=width,
            rng=rng,
        )
        return build_onsite_s_wave_pairing(geometry, pairing=site_offsets)

    edge_offsets = sample_uniform_edge_disorder(
        geometry,
        width=width,
        rng=rng,
    )
    if config.channel is PairingDisorderChannel.SPINLESS_P_WAVE:
        return build_spinless_p_wave_pairing(geometry, pairing=edge_offsets)
    if config.channel is PairingDisorderChannel.CHIRAL_P_WAVE:
        assert config.chirality is not None
        assert config.plane_axes is not None
        return build_chiral_p_wave_pairing(
            geometry,
            pairing=edge_offsets,
            chirality=config.chirality,
            plane_axes=config.plane_axes,
        )
    assert config.channel is PairingDisorderChannel.D_WAVE
    assert config.plane_axes is not None
    return build_d_wave_pairing(
        geometry,
        pairing=edge_offsets,
        plane_axes=config.plane_axes,
    )


def _apply_pairing_block(
    source: np.ndarray,
    pairing_block: np.ndarray,
    *,
    basis: NambuBasis,
) -> np.ndarray:
    if source.shape != (basis.dimension, basis.dimension):
        raise ValueError("BdG Hamiltonian shape does not match nambu_basis")
    expected_pairing_shape = (basis.normal_dimension, basis.normal_dimension)
    if pairing_block.shape != expected_pairing_shape:
        raise ValueError("pairing channel block does not match nambu_basis")
    if not np.all(np.isfinite(pairing_block)):
        raise ValueError("pairing disorder block must contain only finite values")
    if not np.allclose(pairing_block, -pairing_block.T, rtol=0.0, atol=1e-12):
        raise ValueError("pairing disorder block must be antisymmetric")
    if not np.any(pairing_block):
        return source.copy()

    result = np.array(
        source,
        dtype=np.result_type(source.dtype, np.complex128),
        copy=True,
    )
    particle_indices = np.asarray(basis.particle_indices, dtype=np.intp)
    hole_indices = np.asarray(basis.hole_indices, dtype=np.intp)
    result[np.ix_(particle_indices, hole_indices)] += pairing_block
    result[np.ix_(hole_indices, particle_indices)] -= pairing_block.conj()
    return result
