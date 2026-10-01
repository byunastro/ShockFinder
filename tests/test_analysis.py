import numpy as np

import shocktest
from shocktest import pyShockFinder

from test_maps import grid_cell


def test_single_pass_analysis_matches_separate_workflow():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    cell = grid_cell()
    separate_result = finder.find(cell)
    separate_dissipation = pyShockFinder.compute_dissipation(cell, separate_result)
    separate_catalog, labels = shocktest.shock_front_catalog(
        separate_result, separate_dissipation, return_labels=True)
    analysis = finder.analyze(cell, compute_dissipation=True, build_catalog=True)
    np.testing.assert_array_equal(analysis.result.shock, separate_result.shock)
    for name in shocktest.front_dtype.names:
        np.testing.assert_array_equal(analysis.catalog[name], separate_catalog[name])
    np.testing.assert_array_equal(analysis.labels, labels)
    assert analysis.catalog.dtype.itemsize == 78
    for front in analysis.catalog:
        rows = np.flatnonzero(analysis.labels == front['front_id'])
        assert len(rows) == front['ncell']
        assert analysis.result.shock[rows].all()


def test_single_pass_builds_neighbor_tables_once(monkeypatch):
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    original = shocktest.ShockFinder._build_neighbor_tables
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        shocktest.ShockFinder, "_build_neighbor_tables", staticmethod(counted)
    )

    analysis = finder.analyze(grid_cell())

    assert calls == 1
    assert analysis.catalog is not None


def test_analysis_reports_timings_and_counts():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0

    analysis = finder.analyze(grid_cell())

    assert {"input", "neighbors", "scan", "detection_total", "dissipation", "catalog", "total"} <= analysis.timings.keys()
    assert all(value >= 0.0 for value in analysis.timings.values())
    assert analysis.timings["total"] >= analysis.timings["detection_total"]
    assert analysis.counts["retained"] == analysis.result.mach.size
    assert analysis.counts["shock"] == np.count_nonzero(analysis.result.shock)
    assert analysis.counts["groups"] == len(analysis.catalog)


def test_analysis_optional_products_and_clear():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    analysis = finder.analyze(
        grid_cell(), compute_dissipation=False, build_catalog=False
    )

    assert analysis.dissipation is None
    assert analysis.catalog is None
    assert "dissipation" not in analysis.timings
    assert "catalog" not in analysis.timings

    analysis.clear()

    assert analysis.result is None
    assert analysis.timings == {}
    assert analysis.counts == {
        "retained": 0,
        "shock": 0,
        "representative": 0,
        "groups": 0,
    }


def test_analysis_handles_empty_extracted_region():
    cell = {key: np.asarray(value)[:0] for key, value in grid_cell().items()}
    finder = shocktest.ShockFinder()
    finder.minlevel = 0

    analysis = finder.analyze(cell)

    assert analysis.result.mach.size == 0
    assert analysis.dissipation.total.size == 0
    assert len(analysis.catalog) == 0
    assert analysis.counts["retained"] == 0


def test_analysis_uses_finder_gamma_for_dissipation():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.gamma = 1.4
    cell = grid_cell()

    analysis = finder.analyze(cell)
    expected = pyShockFinder.compute_dissipation(
        cell, analysis.result, gamma=finder.gamma
    )

    np.testing.assert_allclose(analysis.dissipation.efficiency, expected.efficiency)
    np.testing.assert_allclose(analysis.dissipation.sound_speed, expected.sound_speed)


def test_analysis_rejects_inconsistent_dissipation_gamma():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.gamma = 1.4

    with np.testing.assert_raises_regex(ValueError, "must match finder.gamma"):
        finder.analyze(grid_cell(), dissipation_options={"gamma": 5.0 / 3.0})


def test_analysis_forwards_grouping_options_and_uses_single_builder():
    finder = shocktest.ShockFinder()
    options = dict(min_group_size=1, mach_tolerance=0.1, connectivity='face', gap_factor=0.25)
    analysis = finder.analyze(grid_cell(), catalog_options=options)
    expected, labels = shocktest.shock_front_catalog(analysis.result, analysis.dissipation,
                                                   return_labels=True, **options)
    for name in expected.dtype.names:
        np.testing.assert_array_equal(analysis.catalog[name], expected[name])
    np.testing.assert_array_equal(analysis.labels, labels)
    analysis.clear()
    assert analysis.labels is None


def test_compact_grouping_uses_saved_consistency_diagnostics():
    finder = shocktest.ShockFinder()
    options = dict(require_mach_consistent=True, min_group_size=1)
    expected = finder.analyze(grid_cell(), catalog_options=options).to_compact()
    actual = finder.analyze(grid_cell(), compact=True, catalog_options=options)
    np.testing.assert_array_equal(actual['group_id'], expected['group_id'])
    for name in actual.groups.dtype.names:
        np.testing.assert_array_equal(actual.groups[name], expected.groups[name])
