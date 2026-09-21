import numpy as np
import pytest
from shocktest import ShockFinder, ShockSamples
from shocktest.spatial import nearest_points, connected_components
from examples.galaxy import classify_galaxy_shock_crossing
from test_maps import grid_cell


def test_compact_survives_dense_clear_and_preserves_indices():
    finder = ShockFinder()
    analysis = finder.analyze(grid_cell())
    compact = analysis.to_compact()
    rows = np.flatnonzero(analysis.result.shock)
    expected = analysis.result.selected_indices[analysis.result.upstream_index[rows]]
    np.testing.assert_array_equal(compact['upstream_index'], expected)
    np.testing.assert_allclose(compact['dissipation_flux'], analysis.dissipation.flux[rows])
    for group in compact.groups:
        assert np.all(np.isin(group.center_indices, compact['input_row']))
    assert not np.shares_memory(compact['mach'], analysis.result.mach)
    analysis.clear()
    assert len(compact) == len(rows) and compact['mach'].max() > 1
    direct = finder.analyze(grid_cell(), compact=True)
    assert isinstance(direct, ShockSamples)
    np.testing.assert_array_equal(direct['mach'], compact['mach'])
    assert direct.counts == compact.counts


def test_tiled_neighbors_and_components_match_dense_reference():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(35, 3)), rng.normal(size=(43, 3))
    index, distance = nearest_points(a, b, memory_budget_bytes=4096)
    full = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    np.testing.assert_array_equal(index, full.argmin(axis=1))
    np.testing.assert_allclose(distance, full.min(axis=1))
    left = connected_components(a, .7, memory_budget_bytes=4096, use_scipy=False)
    right = connected_components(a, .7)
    np.testing.assert_array_equal(left[:, None] == left, right[:, None] == right)


def test_crossing_uses_intersection_and_searches_entire_segment():
    catalog = dict(rows=np.array([10]), pos=np.zeros((1, 3)), dx=np.ones(1),
        mach=np.array([3.]), flux=np.ones(1), zone_width=np.zeros(1),
        normal=np.array([[1., 0., 0.]]), valid_normal=np.array([True]))
    # Plane intersection is at the patch center; midpoint is far outside it.
    result = classify_galaxy_shock_crossing(np.array([[-1., -10., 0.]]),
        np.array([[9., 90., 0.]]), catalog, search_radius_km=1., memory_budget_bytes=4096)
    assert result['crossed'][0]
    assert result['transverse_distance'][0] == pytest.approx(0.)
