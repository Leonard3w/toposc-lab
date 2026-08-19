from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.gases.ensembles import (
    BoseGrandCanonicalParameters,
    BoseMicrocanonicalParameters,
    ClassicalCanonicalParameters,
    ClassicalGrandCanonicalParameters,
    ClassicalMicrocanonicalParameters,
    analyze_bose_grand_canonical,
    analyze_bose_microcanonical,
    analyze_classical_canonical,
    analyze_classical_grand_canonical,
    analyze_classical_microcanonical,
    sample_classical_particles,
)


def test_classical_ensembles_fix_and_fluctuate_the_correct_variables() -> None:
    canonical = analyze_classical_canonical(ClassicalCanonicalParameters())
    grand = analyze_classical_grand_canonical(
        ClassicalGrandCanonicalParameters(chemical_potential=-200.0e-9 * 1.380649e-23)
    )
    microcanonical = analyze_classical_microcanonical(ClassicalMicrocanonicalParameters())

    assert canonical.number_variance == 0.0
    assert canonical.energy_variance > 0.0
    assert grand.number_variance == pytest.approx(grand.particle_number_mean)
    assert grand.energy_variance > 0.0
    assert microcanonical.number_variance == 0.0
    assert microcanonical.energy_variance == 0.0
    assert microcanonical.temperature > 0.0


def test_microcanonical_particle_sample_has_exact_visible_energy_per_particle() -> None:
    result = analyze_classical_microcanonical(
        ClassicalMicrocanonicalParameters(particle_number=20)
    )
    sample = sample_classical_particles(result, visible_particle_count=10, seed=5)
    visible_energy = 0.5 * result.particle_mass_kg * np.sum(sample.velocities**2)

    assert visible_energy == pytest.approx(result.total_energy_mean * 10 / 20)
    assert np.all(sample.positions >= 0.0)
    assert np.all(sample.positions < result.box_length)


def test_grand_canonical_bose_number_fluctuation_exceeds_mean() -> None:
    result = analyze_bose_grand_canonical(BoseGrandCanonicalParameters())

    assert result.mean_particle_number > 0.0
    assert result.number_variance > result.mean_particle_number
    assert result.ground_state_occupation > 0.0


def test_microcanonical_bose_enumeration_preserves_exact_N_and_energy() -> None:
    result = analyze_bose_microcanonical(
        BoseMicrocanonicalParameters(
            particle_number=4,
            energy_quanta=2,
            maximum_mode_index=2,
        )
    )

    assert result.microstate_count > 0
    assert np.sum(result.mean_occupations) == pytest.approx(4.0)
    assert np.dot(result.mode_energy_quanta, result.mean_occupations) == pytest.approx(2.0)
