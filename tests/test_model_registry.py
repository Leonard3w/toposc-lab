from __future__ import annotations

import pytest

from toposc_lab.app.registry import MODEL_REGISTRY, ModelRegistry, ModelSpec
from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters


def test_every_registered_model_has_valid_default_parameters() -> None:
    specifications = MODEL_REGISTRY.specifications()

    assert len(specifications) == 7

    for specification in specifications:
        model = specification.build(specification.validated_default_parameters())

        assert model.model_name
        assert model.basis_layout.dimension == model.hamiltonian().shape[0]


def test_registry_rejects_duplicate_model_keys() -> None:
    specification = ModelSpec(
        key="ssh",
        display_name="SSH",
        category="test",
        description="test model",
        parameter_model=SSHChainParameters,
        factory=SSHChain,
        default_parameters={"n_cells": 3, "v": 0.5, "w": 1.0},
    )
    registry = ModelRegistry((specification,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(specification)


def test_registry_reports_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown model key"):
        MODEL_REGISTRY.get("not-a-model")
