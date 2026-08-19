"""Bosonic many-body systems and statistical-mechanics learning modules."""

from toposc_lab.bosons.ideal_bose_gas import (
    BoseClassicalComparison,
    IdealBoseEinsteinCondensationParameters,
    IdealBoseEinsteinCondensationResult,
    IdealBoseGasParameters,
    IdealBoseGasResult,
    analyze_ideal_bose_gas,
    analyze_ideal_bose_einstein_condensation,
    bose_einstein_occupation,
    compare_bose_to_classical,
    ideal_bec_condensate_fraction,
    ideal_bec_critical_temperature,
    maxwell_boltzmann_occupation,
)

__all__ = [
    "BoseClassicalComparison",
    "IdealBoseEinsteinCondensationParameters",
    "IdealBoseEinsteinCondensationResult",
    "IdealBoseGasParameters",
    "IdealBoseGasResult",
    "analyze_ideal_bose_gas",
    "analyze_ideal_bose_einstein_condensation",
    "bose_einstein_occupation",
    "compare_bose_to_classical",
    "ideal_bec_condensate_fraction",
    "ideal_bec_critical_temperature",
    "maxwell_boltzmann_occupation",
]
