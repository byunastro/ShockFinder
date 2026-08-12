import numpy as np

import shocktest
from shocktest import pyShockFinder

from test_maps import grid_cell


def group_signature(catalog):
    return sorted(
        (group.n_centers, group.mach_peak, group.mach_mean, group.area)
        for group in catalog.groups
    )


def test_single_pass_analysis_matches_separate_workflow():
    cell = grid_cell()
    finder = shocktest.ShockFinder()
    finder.minlevel = 0

    separate_result = finder.find(cell)
    separate_dissipation = pyShockFinder.compute_dissipation(cell, separate_result)
    separate_catalog = shocktest.build_shock_catalog(
        separate_result,
        dissipation=separate_dissipation,
        deduplicate=True,
        min_mach=finder.min_mach,
    )
    analysis = finder.analyze(cell)

    np.testing.assert_allclose(analysis.result.mach, separate_result.mach)
    np.testing.assert_array_equal(analysis.result.shock, separate_result.shock)
    np.testing.assert_array_equal(
        analysis.result.upstream_index, separate_result.upstream_index
    )
    np.testing.assert_allclose(analysis.dissipation.total, separate_dissipation.total)
    assert group_signature(analysis.catalog) == group_signature(separate_catalog)


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
    assert analysis.counts["groups"] == len(analysis.catalog.groups)


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
    assert analysis.catalog.groups == []
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
