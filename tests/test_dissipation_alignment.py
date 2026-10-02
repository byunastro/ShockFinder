"""Row-order and serialization contract, including pre-contract pickle loading."""
from dataclasses import fields, make_dataclass
import pickle
import numpy as np
import pytest
from shocktest import pyShockFinder as module
from shocktest import shock_front_catalog
from test_shock_front_catalog import saved


def test_nonmonotonic_input_ids_preserve_dense_rows_and_compact_order():
    r, _ = saved(ids=[9, 4, 7, 8])
    r.shock[1] = False
    r.upstream_index[:] = [2, 1, 3, 0]
    r.mach[:] = [2., 3., 4., 5.]
    cell = {('T', 'K'): np.arange(10.)*1.e6+1.e7,
            ('rho', 'Msol/kpc3'): np.arange(10.)*1.e5+1.e6,
            ('dx', 'km'): (np.arange(10.)+1.)*module.KPC/1.e5}
    dense = module.compute_dissipation(cell, r)
    compact = module.compute_dissipation(cell, r, _compact=True)
    for field in fields(dense):
        array = getattr(dense, field.name)
        assert array.shape == (4,)
        assert array[1] == 0.
        np.testing.assert_array_equal(getattr(compact, field.name), array[r.shock])
    rows = np.flatnonzero(r.shock)
    np.testing.assert_allclose(dense.area[rows], (r.selected_indices[rows]+1.)**2)
    temp = cell['T', 'K'][r.selected_indices[r.upstream_index[rows]]]
    expected_cs = np.sqrt(r.gamma*module.K_BOLTZMANN*temp/(.59*module.PROTON_MASS))/1.e5
    np.testing.assert_allclose(dense.sound_speed[rows], expected_cs)
    np.testing.assert_array_equal(dense.total, dense.flux*dense.area)
    for obj in (dense, compact):
        assert not hasattr(obj, 'selected_indices')
        payload = pickle.dumps(obj, protocol=4)
        assert b'selected_indices' not in payload
        restored = pickle.loads(payload)
        for field in fields(obj):
            np.testing.assert_array_equal(getattr(restored, field.name), getattr(obj, field.name))


@pytest.mark.parametrize('with_ids', [False, True])
def test_historical_slotted_pickles_load_without_dropping_ids(monkeypatch, with_ids):
    r, d = saved()
    expected = shock_front_catalog(r, d)
    names = [field.name for field in fields(d)]
    old_class = make_dataclass('DissipationResult', names+(['selected_indices'] if with_ids else []), slots=True)
    old_class.__module__ = module.__name__
    order = np.array([3, 1, 0, 2]) if with_ids else np.arange(4)
    args = [getattr(d, name)[order] for name in names]
    if with_ids:
        args.append(r.selected_indices[order])
    with monkeypatch.context() as patch:
        patch.setattr(module, 'DissipationResult', old_class)
        payload = pickle.dumps(old_class(*args), protocol=4)
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(shock_front_catalog(r, restored), expected)
    again = pickle.loads(pickle.dumps(restored, protocol=4))
    np.testing.assert_array_equal(shock_front_catalog(r, again), expected)
    assert hasattr(again, 'selected_indices') == with_ids


def test_idless_fast_path_never_joins_and_checks_all_supplied_column_lengths(monkeypatch):
    r, d = saved()
    def forbidden(*args, **kwargs):
        pytest.fail('ID-less dissipation must not need searchsorted')
    monkeypatch.setattr(np, 'searchsorted', forbidden)
    # Direct helper avoids the unrelated spatial-tree batching searchsorted.
    from shocktest.fronts import _dissipation
    area, total = _dissipation(r, r.selected_indices, np.arange(4), d, False, 2)
    np.testing.assert_array_equal(area, d.area)
    np.testing.assert_array_equal(total, d.total)
    d.flux = d.flux[:2]
    with pytest.raises(ValueError, match='flux must have shape'):
        _dissipation(r, r.selected_indices, np.arange(4), d, False, 2)
