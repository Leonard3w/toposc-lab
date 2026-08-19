from __future__ import annotations

import json
import re

from toposc_lab.app.landau_live_material import (
    landau_live_configuration,
    landau_live_material_html,
)
from toposc_lab.quantum_hall.landau_levels import LandauLevelParameters


def test_live_configuration_preserves_physics_and_safe_control_ranges() -> None:
    parameters = LandauLevelParameters(
        magnetic_field_tesla=7.5,
        electric_field_v_per_m=-8_000.0,
        effective_mass_ratio=0.2,
        maximum_level=4,
        selected_level=2,
    )
    configuration = landau_live_configuration(parameters)

    assert configuration["magnetic_field_tesla"] == 7.5
    assert configuration["electric_field_v_per_m"] == -8_000.0
    assert configuration["effective_mass_ratio"] == 0.2
    assert configuration["selected_level"] == 2
    assert configuration["field_slider_max_tesla"] == 15.0
    assert configuration["electric_slider_max_v_per_m"] == 16_000.0
    assert configuration["initial_electron_count"] == 5


def test_live_material_is_self_contained_and_uses_browser_animation() -> None:
    parameters = LandauLevelParameters(
        magnetic_field_tesla=2.75,
        electric_field_v_per_m=1_250.0,
        angular_momentum=6,
    )
    html = landau_live_material_html(parameters)

    assert '<canvas id="landau-canvas"' in html
    assert 'id="toggle-b"' in html
    assert 'id="toggle-e"' in html
    assert 'value="packet"' in html
    assert 'value="landau"' in html
    assert 'value="symmetric"' in html
    assert "requestAnimationFrame(tick)" in html
    assert "ResizeObserver" in html
    assert "IntersectionObserver" in html
    assert "__CONFIG__" not in html
    assert "<script src=" not in html
    assert "NaN" not in html

    match = re.search(r"const cfg = (\{.*?\});", html)
    assert match is not None
    assert "Infinity" not in match.group(1)
    embedded_configuration = json.loads(match.group(1))
    assert embedded_configuration["magnetic_field_tesla"] == 2.75
    assert embedded_configuration["electric_field_v_per_m"] == 1_250.0
    assert embedded_configuration["angular_momentum"] == 6


def test_live_material_explains_eigenstates_and_measurement_samples() -> None:
    html = landau_live_material_html(LandauLevelParameters())

    assert "kohärentes Paket" in html
    assert "Energieeigenzustands ist stationär" in html
    assert "Born-Messproben" in html
    assert "keine verborgenen klassischen Bahnen" in html
    assert "E×B-Drift" in html


def test_live_material_contains_a_guided_physics_tutorial() -> None:
    html = landau_live_material_html(LandauLevelParameters())

    assert "Geführtes Physik-Tutorial" in html
    assert 'id="tutorial-action"' in html
    assert 'id="tutorial-observe"' in html
    assert 'id="tutorial-physics"' in html
    assert 'id="tutorial-expect"' in html
    assert 'id="tutorial-setup"' in html
    assert "const tutorialSteps" in html
    assert html.count('setup: "') == 9
    assert "prepareTutorialStep" in html
    assert "highlightTutorialTargets" in html
    assert "Messpunkte sind Born-Proben" in html
    assert "ω_c = eB/m*" in html
    assert "v_D = E×B/B²" in html
    assert "r = √(2m) l_B" in html
