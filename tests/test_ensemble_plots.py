from __future__ import annotations

from toposc_lab.gases.ensembles import (
    BoseGrandCanonicalParameters,
    BoseMicrocanonicalParameters,
    ClassicalCanonicalParameters,
    analyze_bose_grand_canonical,
    analyze_bose_microcanonical,
    analyze_classical_canonical,
    sample_classical_particles,
)
from toposc_lab.visualization.ensemble_plots import (
    bose_grand_canonical_figure,
    bose_microcanonical_figure,
    classical_ensemble_motion_figure,
)


def test_classical_motion_figure_has_animation_and_distribution() -> None:
    result = analyze_classical_canonical(ClassicalCanonicalParameters())
    sample = sample_classical_particles(result, visible_particle_count=20)
    figure = classical_ensemble_motion_figure(result, sample, n_frames=4)

    assert len(figure.data) == 3
    assert len(figure.frames) == 4
    assert figure.layout.sliders


def test_bose_ensemble_figures_contain_expected_data() -> None:
    grand = bose_grand_canonical_figure(
        analyze_bose_grand_canonical(BoseGrandCanonicalParameters()),
        maximum_mode_index=4,
    )
    micro = bose_microcanonical_figure(
        analyze_bose_microcanonical(BoseMicrocanonicalParameters())
    )

    assert len(grand.data) == 2
    assert len(micro.data) == 2
