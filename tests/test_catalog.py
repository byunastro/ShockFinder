import numpy as np
import pytest

import shocktest
from shocktest import pyShockFinder


def result_from_centers(pos, mach, normal=None, dx=None, shock=None):
    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]
    if dx is None:
        dx = np.ones(n)
    if shock is None:
        shock = np.asarray(mach) > 1.0
    if normal is None:
        normal = np.tile([1.0, 0.0, 0.0], (n, 1))
    indices = np.arange(n, dtype=np.int64)
    return shocktest.ShockResult(
        mach=np.asarray(mach, dtype=np.float64),
        shock=np.asarray(shock, dtype=bool),
        center_index=np.where(shock, indices, -1),
        upstream_index=np.full(n, -1, dtype=np.int64),
        downstream_index=np.full(n, -1, dtype=np.int64),
        selected_indices=indices,
        pos=pos,
        dx=np.asarray(dx, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        level=np.zeros(n, dtype=np.int32),
    )


def test_face_connected_centers_form_one_surface():
    result = result_from_centers(
        [[0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 2.5, 0.5]],
        [3.0, 3.1, 2.9],
    )

    catalog = shocktest.build_shock_catalog(result)

    assert len(catalog.groups) == 1
    assert catalog.groups[0].n_centers == 3
    assert catalog.groups[0].mach_peak == pytest.approx(3.1)
    assert catalog.groups[0].mach_mean == pytest.approx(3.0)
    np.testing.assert_array_equal(catalog.group_id, np.zeros(3, dtype=np.int64))


def test_separated_surfaces_remain_distinct():
    result = result_from_centers(
        [[0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [5.5, 0.5, 0.5]],
        [3.0, 3.0, 3.0],
    )

    catalog = shocktest.build_shock_catalog(result)

    assert len(catalog.groups) == 2
    assert sorted(group.n_centers for group in catalog.groups) == [1, 2]


def test_different_normals_prevent_false_merge():
    result = result_from_centers(
        [[0.5, 0.5, 0.5], [0.5, 1.5, 0.5]],
        [3.0, 3.0],
        normal=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    catalog = shocktest.build_shock_catalog(result, normal_cosine=0.7)

    assert len(catalog.groups) == 2


def test_catalog_uses_dissipation_area_and_total():
    result = result_from_centers(
        [[0.5, 0.5, 0.5], [0.5, 1.5, 0.5]],
        [2.0, 4.0],
    )
    dissipation = pyShockFinder.DissipationResult(
        flux=np.array([1.0, 1.0]),
        total=np.array([10.0, 30.0]),
        area=np.array([1.0, 3.0]),
        efficiency=np.zeros(2),
        sound_speed=np.zeros(2),
    )

    group = shocktest.build_shock_catalog(
        result, dissipation=dissipation, mach_tolerance=1.0
    ).groups[0]

    assert group.area == pytest.approx(4.0)
    assert group.area_unit == "kpc2"
    assert group.mach_mean == pytest.approx(3.5)
    assert group.dissipation_total == pytest.approx(40.0)
    np.testing.assert_allclose(group.centroid, [0.5, 1.25, 0.5])


def test_real_finder_returns_oriented_unit_normals():
    from test_maps import grid_cell

    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    result = finder.find(grid_cell())

    lengths = np.linalg.norm(result.normal[result.shock], axis=1)
    np.testing.assert_allclose(lengths, 1.0)
    assert np.all(result.normal[result.shock, 0] > 0.0)


def test_catalog_rejects_invalid_thresholds():
    result = result_from_centers([[0.5, 0.5, 0.5]], [2.0])

    with pytest.raises(ValueError, match="mach_tolerance"):
        shocktest.build_shock_catalog(result, mach_tolerance=-0.1)
    with pytest.raises(ValueError, match="normal_cosine"):
        shocktest.build_shock_catalog(result, normal_cosine=1.1)
    with pytest.raises(ValueError, match="min_mach"):
        shocktest.build_shock_catalog(result, min_mach=0.9)
    with pytest.raises(ValueError, match="boundary"):
        shocktest.build_shock_catalog(result, boundary="periodic")


def test_grouping_connects_coarse_and_fine_face_centers():
    result = result_from_centers(
        [[1.0, 1.0, 1.0], [2.5, 0.5, 0.5]],
        [3.0, 3.0],
        dx=[2.0, 1.0],
    )
    result.level = np.array([0, 1], dtype=np.int32)

    catalog = shocktest.build_shock_catalog(result)

    assert len(catalog.groups) == 1


def test_deduplication_collapses_centers_stacked_along_shock_normal():
    result = result_from_centers(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 0.5]],
        [2.5, 3.0, 3.1],
    )

    catalog = shocktest.build_shock_catalog(result, deduplicate=True)

    assert catalog.center_representative[0] == 1
    assert catalog.center_representative[1] == 1
    assert catalog.center_representative[2] == 2
    assert catalog.group_id[0] == -1
    assert len(catalog.groups) == 1
    np.testing.assert_array_equal(catalog.groups[0].center_indices, [1, 2])


def test_deduplication_preserves_tangential_surface_centers():
    result = result_from_centers(
        [[0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 2.5, 0.5]],
        [3.0, 3.0, 3.0],
    )

    catalog = shocktest.build_shock_catalog(result, deduplicate=True)

    np.testing.assert_array_equal(catalog.center_representative, [0, 1, 2])
    assert catalog.groups[0].n_centers == 3


def test_real_finder_reports_positive_zone_span_only_at_shocks():
    from test_maps import grid_cell

    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    result = finder.find(grid_cell())

    assert np.all(result.zone_width[result.shock] > 0.0)
    assert np.all(result.zone_width[~result.shock] == 0.0)


def test_open_boundary_does_not_connect_opposite_region_edges():
    result = result_from_centers(
        [[0.5, 0.5, 0.5], [9.5, 0.5, 0.5]],
        [3.0, 3.0],
    )

    catalog = shocktest.build_shock_catalog(result, boundary="open")

    assert len(catalog.groups) == 2


def test_catalog_statistics_are_invariant_to_cell_order():
    result = result_from_centers(
        [
            [0.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
            [5.5, 0.5, 0.5],
            [5.5, 1.5, 0.5],
        ],
        [2.9, 3.0, 5.0, 5.1],
    )
    order = np.array([2, 0, 3, 1])
    shuffled = result_from_centers(
        result.pos[order],
        result.mach[order],
        normal=result.normal[order],
        dx=result.dx[order],
    )

    first = shocktest.build_shock_catalog(result)
    second = shocktest.build_shock_catalog(shuffled)

    first_stats = sorted((g.n_centers, g.mach_peak, g.area) for g in first.groups)
    second_stats = sorted((g.n_centers, g.mach_peak, g.area) for g in second.groups)
    assert first_stats == second_stats


def test_sensitivity_sweep_reports_parameter_effects(monkeypatch):
    result = result_from_centers(
        [[0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [4.5, 0.5, 0.5]],
        [1.4, 2.0, 4.0],
    )
    original = shocktest.ShockFinder._build_neighbor_tables
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        shocktest.ShockFinder, "_build_neighbor_tables", staticmethod(counted)
    )

    rows = shocktest.analyze_catalog_sensitivity(
        result,
        mach_tolerances=(0.2, 0.5),
        normal_cosines=(0.7,),
        duplicate_normal_cosines=(0.8,),
        min_machs=(1.3, 1.5),
        deduplicate=False,
    )

    assert len(rows) == 4
    assert calls == 1
    assert {row.n_input_centers for row in rows if row.min_mach == 1.3} == {3}
    assert {row.n_input_centers for row in rows if row.min_mach == 1.5} == {2}
    loose = next(
        row for row in rows
        if row.min_mach == 1.3 and row.mach_tolerance == 0.5
    )
    strict = next(
        row for row in rows
        if row.min_mach == 1.3 and row.mach_tolerance == 0.2
    )
    assert loose.n_groups < strict.n_groups
