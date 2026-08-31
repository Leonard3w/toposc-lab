"""Reproducible robustness-testing contracts."""

from toposc_lab.robustness.disorder import (
    DISORDER_RNG_ALGORITHM,
    HAMILTONIAN_ID_SCHEME,
    DisorderParameterValue,
    DisorderProvenance,
    DisorderRealization,
    DisorderRequest,
    DisorderSnapshot,
    DisorderState,
    DisorderTarget,
    DisorderTransform,
    FunctionDisorderTransform,
    exact_hamiltonian_id,
    realize_disorder,
)
from toposc_lab.robustness.edge_removal import (
    RANDOM_EDGE_REMOVAL_KEY,
    RANDOM_EDGE_REMOVAL_VERSION,
    apply_random_edge_removal,
)
from toposc_lab.robustness.hopping import (
    UNIFORM_HOPPING_DISORDER_KEY,
    UNIFORM_HOPPING_DISORDER_VERSION,
    apply_uniform_hopping_disorder,
)
from toposc_lab.robustness.node_removal import (
    RANDOM_NODE_REMOVAL_KEY,
    RANDOM_NODE_REMOVAL_VERSION,
    NodeRemovalRealization,
    apply_random_node_removal,
)
from toposc_lab.robustness.onsite import (
    UNIFORM_ONSITE_DISORDER_KEY,
    UNIFORM_ONSITE_DISORDER_VERSION,
    apply_uniform_onsite_disorder,
)
from toposc_lab.robustness.pairing import (
    UNIFORM_PAIRING_DISORDER_KEY,
    UNIFORM_PAIRING_DISORDER_VERSION,
    PairingDisorderChannel,
    apply_uniform_pairing_disorder,
)

__all__ = [
    "DISORDER_RNG_ALGORITHM",
    "HAMILTONIAN_ID_SCHEME",
    "RANDOM_EDGE_REMOVAL_KEY",
    "RANDOM_EDGE_REMOVAL_VERSION",
    "RANDOM_NODE_REMOVAL_KEY",
    "RANDOM_NODE_REMOVAL_VERSION",
    "UNIFORM_HOPPING_DISORDER_KEY",
    "UNIFORM_HOPPING_DISORDER_VERSION",
    "UNIFORM_ONSITE_DISORDER_KEY",
    "UNIFORM_ONSITE_DISORDER_VERSION",
    "UNIFORM_PAIRING_DISORDER_KEY",
    "UNIFORM_PAIRING_DISORDER_VERSION",
    "DisorderParameterValue",
    "DisorderProvenance",
    "DisorderRealization",
    "DisorderRequest",
    "DisorderSnapshot",
    "DisorderState",
    "DisorderTarget",
    "DisorderTransform",
    "FunctionDisorderTransform",
    "NodeRemovalRealization",
    "PairingDisorderChannel",
    "apply_random_edge_removal",
    "apply_random_node_removal",
    "apply_uniform_hopping_disorder",
    "apply_uniform_onsite_disorder",
    "apply_uniform_pairing_disorder",
    "exact_hamiltonian_id",
    "realize_disorder",
]
