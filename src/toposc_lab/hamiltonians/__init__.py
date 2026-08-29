"""Hamiltonian builders that are independent of particular solvers."""

from toposc_lab.hamiltonians.bdg import build_bdg_hamiltonian
from toposc_lab.hamiltonians.disorder import (
    uniform_edge_disorder,
    uniform_site_disorder,
)
from toposc_lab.hamiltonians.nambu import NambuBasis, NambuSector, NambuState
from toposc_lab.hamiltonians.pairing import (
    EdgePairingMap,
    EdgePairingTerm,
    OnsitePairingMap,
    OnsitePairingTerm,
    PairingValue,
    build_chiral_p_wave_pairing,
    build_d_wave_pairing,
    build_onsite_s_wave_pairing,
    build_spinless_p_wave_pairing,
)
from toposc_lab.hamiltonians.rashba import (
    PlaneAxes,
    RashbaCoupling,
    RashbaCouplingMap,
    build_rashba_term,
    rashba_hopping_matrix,
)
from toposc_lab.hamiltonians.tight_binding import (
    HoppingMap,
    HoppingTerm,
    OnsiteMap,
    OnsiteTerm,
    TermValue,
    build_tight_binding_hamiltonian,
)
from toposc_lab.hamiltonians.zeeman import (
    ZeemanField,
    ZeemanFieldMap,
    ZeemanTerm,
    build_zeeman_term,
    zeeman_matrix,
)

__all__ = [
    "EdgePairingMap",
    "EdgePairingTerm",
    "HoppingMap",
    "HoppingTerm",
    "NambuBasis",
    "NambuSector",
    "NambuState",
    "OnsiteMap",
    "OnsitePairingMap",
    "OnsitePairingTerm",
    "OnsiteTerm",
    "PairingValue",
    "PlaneAxes",
    "RashbaCoupling",
    "RashbaCouplingMap",
    "TermValue",
    "ZeemanField",
    "ZeemanFieldMap",
    "ZeemanTerm",
    "build_bdg_hamiltonian",
    "build_chiral_p_wave_pairing",
    "build_d_wave_pairing",
    "build_onsite_s_wave_pairing",
    "build_rashba_term",
    "build_spinless_p_wave_pairing",
    "build_tight_binding_hamiltonian",
    "build_zeeman_term",
    "uniform_edge_disorder",
    "uniform_site_disorder",
    "rashba_hopping_matrix",
    "zeeman_matrix",
]
