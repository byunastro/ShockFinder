"""Exact-value equivalence for memory-saving paths; no relaxed tolerances."""
from dataclasses import fields
import numpy as np
import pytest

from shocktest import ShockFinder, ShockSamples, ShockResult
from shocktest.catalog import _region_bounds
from test_maps import grid_cell


def assert_samples_equal(left, right):
    assert left.columns.keys() == right.columns.keys()
    for name in left.columns:
        assert left[name].dtype == right[name].dtype
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)
    assert left.counts == right.counts
    assert len(left.groups) == len(right.groups)
    for a, b in zip(left.groups, right.groups):
        for field in fields(a):
            np.testing.assert_equal(getattr(a, field.name), getattr(b, field.name))


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('filtered', [False, True])
@pytest.mark.parametrize('pressure', [False, True])
def test_direct_compact_matches_dense_columns_and_catalog(dtype, filtered, pressure):
    cell = grid_cell(nx=12, ny=6)
    # Test reading float32 fields without a full-size float64 conversion.
    cell = {k: v.astype(np.float32) if v.dtype.kind == 'f' else v for k, v in cell.items()}
    if pressure:
        cell['thermal_pressure'] = cell['rho', 'Msol/kpc3'] * cell['T', 'K']
        cell['thermal_pressure'][::3] *= .1
    finder = ShockFinder()
    finder.mach_validation_dtype = dtype
    finder.filter_inconsistent = filtered
    expected = finder.analyze(cell).to_compact()
    actual = finder.analyze(cell, compact=True)
    assert_samples_equal(expected, actual)


@pytest.mark.parametrize('kind', ['empty', 'no_shocks', 'no_validation', 'level_selection'])
def test_compact_optional_empty_and_selected_meshes(kind):
    cell = grid_cell()
    finder = ShockFinder()
    if kind == 'empty':
        cell = {k: v[:0] for k, v in cell.items()}
    elif kind == 'no_shocks':
        cell['vx', 'km/s'].fill(1.)
    elif kind == 'no_validation':
        finder.validate_mach = False
    else:
        cell['level'][::2] = 18
        finder.minlevel = 19
    dense = finder.find(cell)
    samples = finder.find(cell, compact=True)
    assert_samples_equal(dense.to_compact(), samples)


def test_compact_validation_allocates_only_centers(monkeypatch):
    original = ShockFinder._populate_mach_validation
    sizes = []
    def record(self, cell, arrays, result):
        sizes.append((result.mach.size, arrays['temp'].size))
        return original(self, cell, arrays, result)
    monkeypatch.setattr(ShockFinder, '_populate_mach_validation', record)
    samples = ShockFinder().find(grid_cell(nx=100, ny=4), compact=True)
    assert sizes == [(len(samples), 400)]
    assert sizes[0][0] < sizes[0][1]


def test_science_profile_preserves_stored_values_and_float_precision():
    finder = ShockFinder()
    full = finder.analyze(grid_cell(), compact=True)
    small = finder.analyze(grid_cell(), compact=True,
                           compact_options={'profile': 'science', 'index_dtype': 'auto'})
    assert small.nbytes < full.nbytes
    for name, values in small.columns.items():
        np.testing.assert_array_equal(values, full[name])
        if values.dtype.kind == 'f':
            assert values.dtype == full[name].dtype
    assert small['input_row'].dtype == np.int32
    assert 'mach_pressure' not in small.columns
    assert 'upstream_pos' not in small.columns
    assert small['mach'].dtype == np.float64
    assert small.counts == full.counts


def test_auto_indices_never_truncate_large_input_rows():
    result = ShockResult(mach=np.array([3.]), shock=np.array([True]),
        center_index=np.array([0]), upstream_index=np.array([0]), downstream_index=np.array([0]),
        selected_indices=np.array([2**31 + 19], dtype=np.int64))
    compact = result.to_compact(index_dtype='auto')
    assert compact['retained_row'].dtype == np.int32
    for name in ('input_row', 'center_index', 'upstream_index', 'downstream_index'):
        assert compact[name].dtype == np.int64
        assert compact[name][0] == 2**31 + 19


@pytest.mark.parametrize('profile', ['full', 'science'])
@pytest.mark.parametrize('compressed', [False, True])
def test_compact_archive_round_trip_preserves_all_products(tmp_path, profile, compressed):
    expected = ShockFinder().analyze(grid_cell(), compact=True,
        compact_options={'profile': profile, 'index_dtype': 'auto'})
    path = expected.save(tmp_path / 'shocks.npz', compressed=compressed)
    actual = ShockSamples.load(path)
    assert_samples_equal(expected, actual)
    assert actual.metadata == expected.metadata
    assert actual.timings == expected.timings
    for name in expected.columns:
        if expected[name].dtype == np.float64:
            np.testing.assert_array_equal(expected[name].view(np.uint64), actual[name].view(np.uint64))


def test_chunked_bounds_are_exact():
    rng = np.random.default_rng(12)
    pos, dx = rng.normal(size=(47, 3))*1.e20, rng.uniform(1.e15, 1.e17, 47)
    lo, hi = _region_bounds(pos, dx, chunk_size=7)
    np.testing.assert_array_equal(lo, np.min(pos-.5*dx[:, None], axis=0))
    np.testing.assert_array_equal(hi, np.max(pos+.5*dx[:, None], axis=0))


def test_dense_index_narrowing_is_opt_in_and_lossless():
    finder = ShockFinder()
    expected = finder.find(grid_cell())
    finder.index_dtype = 'auto'
    actual = finder.find(grid_cell())
    for field in fields(expected):
        a, b = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=field.name)
    for name in ('selected_indices', 'center_index', 'upstream_index', 'downstream_index'):
        assert getattr(expected, name).dtype == np.int64
        assert getattr(actual, name).dtype == np.int32
    assert actual.mach.dtype == expected.mach.dtype == np.float64
    wider = ShockFinder._to_python_indices(np.array([2**31+4], np.int64), mode='auto')
    assert wider[0] == 2**31+3 and wider.dtype == np.int64
    finder.index_dtype = 'float32'
    with pytest.raises(ValueError, match='index_dtype'):
        finder.find(grid_cell())
