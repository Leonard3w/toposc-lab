"""Exploratory spatial audit of archived SIZE-001 markers; no new physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.search._research_storage import encode_record, load_record
from toposc_lab.search.phase_10_campaign import _verify_completion
from toposc_lab.search.phase_10_size_methods import MASK_NAMES, SIZES
from toposc_lab.search.phase_10_size_methods_campaign import (
    _load_bound_sealed,
    size_methods_summary,
)

FloatArray = NDArray[np.float64]


def edge_distances(points: FloatArray, pairs: list[tuple[int, int]]) -> FloatArray:
    """Euclidean distance to the nearest final changed segment, measured on X0."""
    distances = np.full(len(points), np.inf)
    for source, target in pairs:
        direction = points[target] - points[source]
        fraction = np.clip((points - points[source]) @ direction / (direction @ direction), 0, 1)
        projection = points[source] + fraction[:, None] * direction
        distances = np.minimum(distances, np.linalg.norm(points - projection, axis=1))
    return distances


def site_data(record: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    """Undo unique-coordinate sorting, retaining every physical site's identity."""
    geometry = cell['genome'].to_geometry()
    order = np.asarray(cell['topology_input_evidence']['unique_coordinate_order'])
    inverse = np.argsort(order)
    maps = record['scientific']['marker_maps']
    assert len(maps) == 5
    for name, item, stored in zip(
        MASK_NAMES, maps, record['scientific']['topology_grid']['local_chern'], strict=True
    ):
        assert item['name'] == name
        np.testing.assert_array_equal(item['positions'], geometry.coordinates[order])
        np.testing.assert_array_equal(item['local_marker'], maps[0]['local_marker'])
        selected = item['bulk_mask']
        value = np.sum(item['local_marker'][selected] * item['position_areas'][selected])
        value /= np.sum(item['position_areas'][selected])
        assert abs(value - stored['bulk_chern_estimate']) < 1e-12
    return {
        'marker': maps[0]['local_marker'][inverse],
        'areas': maps[0]['position_areas'][inverse],
        'masks': [item['bulk_mask'][inverse] for item in maps],
        'edges': {(e.source, e.target) for e in geometry.edges},
    }


def analyze(stage: Path) -> tuple[dict[str, Any], dict[tuple[int, str], list[FloatArray]]]:
    """Check all archives, then compute equal-trajectory spatial descriptions."""
    _verify_completion(stage)
    inputs = load_record(stage / 'input_plan.json')
    archived = load_record(stage / 'analysis.json')
    records = []
    data = []
    for index, cell in enumerate(inputs):
        record = _load_bound_sealed(stage / 'evaluations' / f'cell_{index:04d}', cell)
        run = record['run']
        assert run.is_valid and run.failure is None
        assert run.reproducibility.seed is None
        assert run.reproducibility.geometry_id == exact_geometry_id(cell['genome'].to_geometry())
        records.append(record)
        data.append(site_data(record, cell))
    rebuilt = size_methods_summary(tuple(records), preflight=False)
    assert all(encode_record(rebuilt[key]) == encode_record(archived[key]) for key in rebuilt)
    assert len(records) == 150 and rebuilt['control_valid']
    rows = []
    heatmaps: dict[tuple[int, str], list[FloatArray]] = {}
    for n in SIZES:
        reference_index = next(
            i for i, r in enumerate(records) if r['n'] == n and r['block'] == 'control_start'
        )
        reference = data[reference_index]
        points = np.array([(x, y) for x in range(n) for y in range(n)], dtype=float)
        depth = np.min(np.column_stack((points, n - 1 - points)), axis=1)
        for index, record in enumerate(records):
            if record['n'] != n or record['block'].startswith('control'):
                continue
            current = data[index]
            delta = current['marker'] - reference['marker']
            changes = sorted(current['edges'] ^ reference['edges'])
            distance = edge_distances(points, changes)
            row: dict[str, Any] = {
                'n': n, 'block': record['block'], 'root': record['root'],
                'changed_segments': len(changes), 'rings': [], 'mask_decomposition': [],
            }
            for ring in range(n // 2):
                mask = depth == ring
                near = mask & (distance <= 1.0)
                far = mask & (distance > 2.0)
                row['rings'].append({
                    'depth': ring, 'sites': int(mask.sum()),
                    'mean_abs_delta': float(np.mean(np.abs(delta[mask]))),
                    'mean_signed_delta': float(np.mean(delta[mask])),
                    'near_sites': int(near.sum()), 'far_sites': int(far.sum()),
                    'near_mean_abs_delta': float(np.mean(np.abs(delta[near]))) if near.any() else None,
                    'far_mean_abs_delta': float(np.mean(np.abs(delta[far]))) if far.any() else None,
                })
            for name, mask, control_mask in zip(
                MASK_NAMES, current['masks'], reference['masks'], strict=True
            ):
                def average(marker: FloatArray, area: FloatArray, selected: Any) -> float:
                    return float(np.sum(marker[selected] * area[selected]) / np.sum(area[selected]))

                actual = average(current['marker'], current['areas'], mask)
                same_region = average(reference['marker'], reference['areas'], mask)
                original_region = average(reference['marker'], reference['areas'], control_mask)
                row['mask_decomposition'].append({
                    'name': name, 'candidate_C': actual, 'square_same_mask_C': same_region,
                    'square_original_mask_C': original_region,
                    'field_and_area_effect': actual - same_region,
                    'mask_selection_effect': same_region - original_region,
                    'mask_changed': bool(np.any(mask != control_mask)),
                })
            rows.append(row)
            heatmaps.setdefault((n, record['block']), []).append(np.abs(delta))
    aggregates = []
    for n in SIZES:
        for block in ('positions', 'connectivity', 'combined'):
            selected_rows = [r for r in rows if r['n'] == n and r['block'] == block]
            rings = []
            for ring_depth in range(n // 2):
                selected = [r['rings'][ring_depth] for r in selected_rows]
                paired = [r for r in selected if r['near_sites'] and r['far_sites']]
                differences = [r['near_mean_abs_delta'] - r['far_mean_abs_delta'] for r in paired]
                rings.append({
                    'depth': ring_depth,
                    'median_trial_mean_abs_delta': float(np.median([r['mean_abs_delta'] for r in selected])),
                    'near_far_paired_trials': len(paired),
                    'near_greater_count': sum(v > 0 for v in differences),
                    'median_near_minus_far': float(np.median(differences)) if paired else None,
                })
            aggregates.append({'n': n, 'block': block, 'rings': rings})
    result = {
        'status': 'exploratory post-hoc spatial analysis; no new screening criteria',
        'source_stage': stage.as_posix(),
        'source_code_commit': archived['environment']['code_commit'],
        'source_complete_sha256': hashlib.sha256((stage / 'complete.json').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'definitions': {
            'delta': 'candidate local marker density minus same-size square at identical site label',
            'distance': 'Euclidean distance on original coordinates to final symmetric-difference segments',
            'near': '<=1 lattice spacing', 'far': '>2 lattice spacings; (1,2] omitted',
            'unit': 'one trajectory; sites and masks are correlated, not independent replicates',
        },
        'rows': rows, 'aggregates': aggregates,
    }
    _verify_completion(stage)
    return result, heatmaps


def plot_maps(heatmaps: dict[tuple[int, str], list[FloatArray]], output: Path) -> None:
    """Show every condition with equal-trajectory means and one common color scale."""
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    means = {key: np.mean(values, axis=0) for key, values in heatmaps.items()}
    lower = min(float(values.min()) for values in means.values())
    upper = max(float(values.max()) for values in means.values())
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), layout='constrained')
    for row, n in enumerate(SIZES):
        for col, (block, label) in enumerate((
            ('positions', 'Positions only'), ('connectivity', 'Edges only'), ('combined', 'Combined')
        )):
            ax = axes[row, col]
            panel = ax.imshow(means[(n, block)].reshape(n, n).T, origin='lower',
                              norm=LogNorm(vmin=lower, vmax=upper), cmap='magma',
                              interpolation='nearest')
            ax.set_title(f'{n} x {n} | {label}')
            ax.set_xlabel('Original x site coordinate')
            ax.set_ylabel('Original y site coordinate')
    fig.colorbar(panel, ax=axes, shrink=0.75, label='Mean |marker - square marker| (16 trajectories)')
    fig.suptitle('Exploratory spatial marker changes\nSame-size reference subtracted; logarithmic color scale')
    fig.savefig(output / 'marker_changes.png', dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=Path('results/phase_10_size_methods_v1/full'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory to preserve previous analyses.')
    result, maps = analyze(args.stage)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'spatial_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8'
    )
    plot_maps(maps, args.output)
    print(json.dumps(result['aggregates'], indent=2), flush=True)


if __name__ == '__main__':
    main()
