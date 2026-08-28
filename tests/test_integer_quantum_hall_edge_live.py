from __future__ import annotations

import json
import re

from toposc_lab.app.integer_quantum_hall_edge_live import (
    integer_quantum_hall_edge_configuration,
    integer_quantum_hall_edge_live_html,
)
from toposc_lab.quantum_hall.integer_quantum_hall import IQHEParameters


def test_edge_live_configuration_contains_finite_scientific_data() -> None:
    parameters = IQHEParameters(
        magnetic_field_tesla=7.0,
        hall_voltage_microvolt=-120.0,
        skipping_orbit_radius_l_b=2.1,
    )
    configuration = integer_quantum_hall_edge_configuration(parameters)

    assert configuration["magnetic_field_tesla"] == 7.0
    assert configuration["hall_voltage_microvolt"] == -120.0
    assert configuration["skipping_orbit_radius_l_b"] == 2.1
    assert configuration["mode_count_per_edge"] >= 1
    assert configuration["left_velocity_km_s"] > 0.0
    assert configuration["right_velocity_km_s"] < 0.0
    assert len(configuration["x_over_l_b"]) == 161
    assert len(configuration["energies_mev"]) >= 1


def test_edge_live_html_is_self_contained_and_explains_semiclassical_status() -> None:
    html = integer_quantum_hall_edge_live_html(IQHEParameters())

    assert '<canvas id="edge-canvas"' in html
    assert 'id="bias"' in html
    assert 'id="radius"' in html
    assert "requestAnimationFrame(tick)" in html
    assert "ResizeObserver" in html
    assert "IntersectionObserver" in html
    assert "semiklassische Skipping-Orbit-Interpretation" in html
    assert "keine verborgene klassische Bahn" in html
    assert "I_y=N(e^2/h)V_H" in html
    assert "__CONFIG__" not in html
    assert "<script src=" not in html
    assert "NaN" not in html

    match = re.search(r"const cfg=(\{.*?\});", html)
    assert match is not None
    configuration = json.loads(match.group(1))
    assert configuration["mode_count_per_edge"] >= 1

