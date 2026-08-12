import csv

import numpy as np
import pytest

import shocktest

from test_maps import grid_cell


def assert_group_equal(actual, expected):
    scalar_fields = (
        "group_id", "n_centers", "mach_peak", "mach_mean", "area", "area_unit",
        "dissipation_total", "level_min", "level_max", "upstream_temperature",
        "upstream_density", "external_fraction", "classification",
    )
    for field in scalar_fields:
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if isinstance(expected_value, float) and np.isnan(expected_value):
            assert np.isnan(actual_value)
        else:
            assert actual_value == pytest.approx(expected_value) if isinstance(expected_value, float) else actual_value == expected_value
    np.testing.assert_array_equal(actual.center_indices, expected.center_indices)
    np.testing.assert_allclose(actual.centroid, expected.centroid)
    np.testing.assert_allclose(actual.mean_normal, expected.mean_normal)
    np.testing.assert_allclose(actual.bounds, expected.bounds)


def test_catalog_npz_round_trip_preserves_complete_catalog(tmp_path):
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    analysis = finder.analyze(grid_cell(), external_temperature=2.0e7)
    path = tmp_path / "shock_catalog.npz"

    returned = shocktest.save_shock_catalog(path, analysis.catalog)
    loaded = shocktest.load_shock_catalog(path)

    assert returned == path
    np.testing.assert_array_equal(loaded.group_id, analysis.catalog.group_id)
    np.testing.assert_array_equal(
        loaded.center_representative, analysis.catalog.center_representative
    )
    assert len(loaded.groups) == len(analysis.catalog.groups)
    for actual, expected in zip(loaded.groups, analysis.catalog.groups):
        assert_group_equal(actual, expected)


def test_empty_catalog_round_trip(tmp_path):
    catalog = shocktest.ShockCatalog(
        group_id=np.full(3, -1, dtype=np.int64),
        center_representative=np.full(3, -1, dtype=np.int64),
        groups=[],
    )
    path = tmp_path / "empty.npz"

    shocktest.save_shock_catalog(path, catalog, compressed=False)
    loaded = shocktest.load_shock_catalog(path)

    assert loaded.groups == []
    np.testing.assert_array_equal(loaded.group_id, catalog.group_id)
    np.testing.assert_array_equal(
        loaded.center_representative, catalog.center_representative
    )


def test_catalog_csv_contains_group_summary(tmp_path):
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    catalog = finder.analyze(grid_cell(), external_temperature=2.0e7).catalog
    path = tmp_path / "shock_groups.csv"

    shocktest.save_shock_catalog_csv(path, catalog)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(catalog.groups)
    assert rows[0]["classification"] == "external"
    assert float(rows[0]["mach_peak"]) == pytest.approx(catalog.groups[0].mach_peak)
    assert {"centroid_x", "normal_z", "external_fraction", "area_unit"} <= rows[0].keys()


def test_catalog_loader_rejects_missing_fields(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, schema_version=np.array(1))

    with pytest.raises(ValueError, match="missing fields"):
        shocktest.load_shock_catalog(path)


def test_catalog_loader_rejects_unknown_schema(tmp_path):
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    catalog = finder.analyze(grid_cell()).catalog
    valid_path = tmp_path / "valid.npz"
    invalid_path = tmp_path / "invalid.npz"
    shocktest.save_shock_catalog(valid_path, catalog)
    with np.load(valid_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["schema_version"] = np.asarray(999, dtype=np.int32)
    np.savez(invalid_path, **arrays)

    with pytest.raises(ValueError, match="unsupported catalog schema"):
        shocktest.load_shock_catalog(invalid_path)


def test_catalog_loader_rejects_inconsistent_center_counts(tmp_path):
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    catalog = finder.analyze(grid_cell()).catalog
    valid_path = tmp_path / "valid.npz"
    invalid_path = tmp_path / "invalid-counts.npz"
    shocktest.save_shock_catalog(valid_path, catalog)
    with np.load(valid_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["n_centers"] = arrays["n_centers"] + 1
    np.savez(invalid_path, **arrays)

    with pytest.raises(ValueError, match="center counts"):
        shocktest.load_shock_catalog(invalid_path)
