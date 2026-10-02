"""Analytic spatial-front tests; no simulation files, clusters, or detector runs."""
from dataclasses import fields
import pickle

import numpy as np
import pytest

from shocktest import (
    shock_front_catalog, front_dtype, QUALITY_NO_AREA, QUALITY_NO_DISS_RATE,
    QUALITY_UNDEFINED_NORMAL, QUALITY_APPROX_CONNECTIVITY, QUALITY_GAP_BRIDGED,
    QUALITY_PARTIAL_SUMMARY,
)
from shocktest.core import ShockResult
from shocktest.pyShockFinder import DissipationResult
from shocktest.compact import compact_shocks
from shocktest import fronts


def saved(pos=None, *, dx=1., normal=None, ids=None):
    if pos is None:
        pos = np.column_stack((np.zeros(4), np.arange(4.), np.zeros(4)))
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    rows = np.arange(n)
    result = ShockResult(np.full(n, 3.), np.ones(n, bool), rows.copy(), rows.copy(), rows.copy(),
        rows+100 if ids is None else np.asarray(ids), pos=pos,
        dx=np.broadcast_to(dx, (n,)).copy(), normal=np.tile([1., 0., 0.], (n, 1)) if normal is None else np.asarray(normal, dtype=float),
        position_unit='kpc')
    diss = DissipationResult(np.full(n, 10.), np.arange(n, dtype=float)+10.,
        np.arange(n, dtype=float)+1., np.zeros(n), np.zeros(n))
    return result, diss


def as_dict(obj):
    return {field.name: getattr(obj, field.name) for field in fields(obj)}




def test_schema_units_weighted_summaries_and_unique_centers():
    r, d = saved()
    r.mach[:] = [2.5, 3., 4., 5.]
    catalog, labels = shock_front_catalog(r, d, return_labels=True)
    assert catalog.dtype == front_dtype and catalog.dtype.itemsize == 78
    assert not catalog.dtype.hasobject and not catalog.dtype.isalignedstruct
    assert catalog.shape == (1,)
    f = catalog[0]
    assert f['front_id'] == 0 and f['ncell'] == 4
    np.testing.assert_allclose(f['center'], [0., 2., 0.])
    np.testing.assert_allclose(f['normal'], [1., 0., 0.])
    np.testing.assert_allclose(f['extent'], [1., 4., 1.])
    assert f['area'] == 10. and f['mach'] == np.float32(4.05) and f['diss_rate'] == 46.
    assert f['quality'] == QUALITY_APPROX_CONNECTIVITY
    np.testing.assert_array_equal(labels, np.zeros(4, np.int32))
    # A shock-zone alias points to an accepted center; never double-count it.
    r.center_index[0] = 1
    cat, lab = shock_front_catalog(r, d, return_labels=True)
    assert cat['ncell'][0] == 3 and cat['area'][0] == 9.
    np.testing.assert_array_equal(lab, [-1, 0, 0, 0])


def test_geometry_converts_km_to_kpc_without_changing_dissipation_units():
    r, d = saved()
    expected = shock_front_catalog(r, d)
    r.pos *= 3.0856775814913673e16
    r.dx *= 3.0856775814913673e16
    r.position_unit = 'km'
    actual = shock_front_catalog(r, d)
    for name in front_dtype.names:
        np.testing.assert_allclose(actual[name], expected[name])


def test_legacy_id_join_permutations_missing_ids_and_aligned_outputs():
    r, d = saved()
    expected = shock_front_catalog(r, d)
    permutation = [3, 0, 2, 1]
    shuffled = {key: value[permutation] for key, value in as_dict(d).items()}
    shuffled['selected_indices'] = r.selected_indices[permutation].copy()
    np.testing.assert_array_equal(shock_front_catalog(r, shuffled), expected)
    shuffled['selected_indices'][1] = 987654
    actual = shock_front_catalog(r, shuffled)[0]
    assert np.isnan(actual['area']) and np.isnan(actual['diss_rate'])
    assert actual['quality'] & QUALITY_PARTIAL_SUMMARY
    shuffled['selected_indices'][:] = 100
    with pytest.raises(ValueError, match='duplicate dissipation'):
        shock_front_catalog(r, shuffled)
    assert not hasattr(d, 'selected_indices')
    np.testing.assert_array_equal(shock_front_catalog(r, d), expected)
    d.total = d.total[:2]
    with pytest.raises(ValueError, match='total must have shape'):
        shock_front_catalog(r, d)


def test_dissipation_missing_values_never_become_incomplete_totals():
    r, d = saved()
    d.area[1] = np.nan
    d.total[2] = np.nan
    d.flux[:] = 1.e100
    r.normal[0] = np.nan
    cat = shock_front_catalog(r, d)[0]
    assert np.isnan(cat['area']) and np.isnan(cat['diss_rate'])
    np.testing.assert_allclose(cat['center'], [0., 1.5, 0.])
    np.testing.assert_allclose(cat['normal'], [1., 0., 0.])
    assert cat['mach'] == 3.
    assert cat['quality'] & QUALITY_NO_AREA
    assert cat['quality'] & QUALITY_NO_DISS_RATE
    assert cat['quality'] & QUALITY_PARTIAL_SUMMARY
    assert not cat['quality'] & QUALITY_UNDEFINED_NORMAL
    missing = shock_front_catalog(r, {'selected_indices': r.selected_indices, 'flux': d.flux})[0]
    assert np.isnan(missing['diss_rate'])
    r.normal = None
    no_normal = shock_front_catalog(r, None)[0]
    assert np.isnan(no_normal['normal']).all()
    assert no_normal['quality'] & QUALITY_UNDEFINED_NORMAL


def test_validity_masks_and_diagnostic_mach_consistency():
    r, d = saved(np.c_[np.zeros(8), np.arange(8.), np.zeros(8)])
    r.shock[0] = False
    r.upstream_index[1] = -1
    r.mach_validation_status = np.zeros(8, np.uint16)
    r.mach_validation_status[2] = 1 << 8
    r.mach_consistent = np.ones(8, bool)
    r.mach_consistent[3] = False
    r.mach[4] = 1.
    r.pos[5] = np.nan
    cat, labels = shock_front_catalog(r, d, min_group_size=1, return_labels=True)
    np.testing.assert_array_equal(labels, [-1, -1, -1, 0, -1, -1, 1, 1])
    assert cat['ncell'].sum() == 3
    _, strict = shock_front_catalog(r, d, min_group_size=1, return_labels=True, require_mach_consistent=True)
    assert strict[3] == -1


def test_parallel_nearby_surfaces_stay_separate():
    pos = np.array([[x, y, 0.] for x in (0., 1.) for y in range(5)])
    r, d = saved(pos)
    cat, labels = shock_front_catalog(r, d, return_labels=True)
    np.testing.assert_array_equal(cat['ncell'], [5, 5])
    np.testing.assert_array_equal(labels, [0]*5+[1]*5)
    # Oppositely oriented adjacent samples do not use abs(normal dot normal).
    r, d = saved()
    r.normal[2:] *= -1
    cat = shock_front_catalog(r, d, min_group_size=2)
    np.testing.assert_array_equal(cat['ncell'], [2, 2])


def test_curved_front_connects_by_local_normals_and_cancellation_is_not_rejection():
    theta = np.linspace(0., 2*np.pi, 32, endpoint=False)
    normals = np.c_[np.cos(theta), np.sin(theta), np.zeros(32)]
    r, d = saved(normals*5., dx=1.1, normal=normals)
    d.area[:] = 1.
    cat, labels = shock_front_catalog(r, d, return_labels=True)
    assert len(cat) == 1 and cat['ncell'][0] == 32
    assert np.isnan(cat['normal']).all()
    assert cat['quality'][0] & QUALITY_UNDEFINED_NORMAL
    assert not cat['quality'][0] & QUALITY_PARTIAL_SUMMARY
    np.testing.assert_array_equal(labels, np.zeros(32))


def test_mixed_amr_widths_preserve_contact_and_extent():
    r, d = saved([[0, 0, 0], [0, 1.5, 0], [0, 2.5, 0], [0, 3.5, 0]], dx=[2., 1., 1., 1.])
    cat = shock_front_catalog(r, d, connectivity='face')
    assert cat['ncell'][0] == 4
    np.testing.assert_allclose(cat['extent'][0], [2., 5., 2.])


def test_gap_bridging_is_disabled_by_default_and_flagged_when_enabled():
    r, d = saved([[0, 0, 0], [0, 1, 0], [0, 3, 0], [0, 4, 0]])
    cat = shock_front_catalog(r, d, min_group_size=2)
    np.testing.assert_array_equal(cat['ncell'], [2, 2])
    assert not np.any(cat['quality'] & QUALITY_GAP_BRIDGED)
    bridged = shock_front_catalog(r, d, min_group_size=2, gap_factor=1.)
    assert bridged['ncell'][0] == 4
    assert bridged['quality'][0] & QUALITY_GAP_BRIDGED








def test_determinism_reordering_threads_and_irrelevant_cluster_metadata(monkeypatch):
    import shocktest
    monkeypatch.setattr(shocktest.ShockFinder, 'find', lambda *a, **kw: pytest.fail('detector must not run'))
    r, d = saved([[0, 0, 0], [0, 1, 0], [0, 2, 0], [20, 0, 0], [20, 1, 0], [20, 2, 0]],
                 ids=[800, 700, 900, 100, 200, 300])
    before = pickle.dumps((r, d))
    cat, labels = shock_front_catalog(r, d, return_labels=True)
    again, more_threads = shock_front_catalog(r, d, return_labels=True, thread=4, max_neighbor_pairs=2)
    assert again.tobytes() == cat.tobytes()
    np.testing.assert_array_equal(labels, more_threads)
    assert pickle.dumps((r, d)) == before
    np.testing.assert_array_equal(labels, [1, 1, 1, 0, 0, 0])
    mapping = as_dict(r)
    mapping.update(cluster_info={'ccen1': 'unused'}, merger_axis='unused')
    assert shock_front_catalog(mapping, d).tobytes() == cat.tobytes()
    with pytest.raises(TypeError):
        shock_front_catalog(r, d, cluster_info={})
    permutation = np.array([5, 2, 0, 4, 1, 3])
    reordered = {key: val[permutation] if isinstance(val, np.ndarray) else val for key, val in mapping.items()}
    for key in ('center_index', 'upstream_index', 'downstream_index'):
        reordered[key] = np.arange(6)
    reordered_diss = {key: value[permutation] for key, value in as_dict(d).items()}
    out, lab = shock_front_catalog(reordered, reordered_diss, return_labels=True)
    assert out.tobytes() == cat.tobytes()
    np.testing.assert_array_equal(lab, labels[permutation])


def test_point_fallback_missing_widths_minimum_size_and_precision():
    r, d = saved()
    r.dx = None
    with pytest.raises(ValueError, match='dx must have shape'):
        shock_front_catalog(r, d)
    r, d = saved()
    r.pos += 1.e9
    r.pos[:, 1] += .125
    out = shock_front_catalog(r, d)
    assert out['center'][0, 1] == 1.e9+2.125
    cat, labels = shock_front_catalog(r, d, min_group_size=5, return_labels=True)
    assert len(cat) == 0 and np.all(labels == -1) and labels.dtype == np.int32
    with pytest.raises(OverflowError):
        shock_front_catalog(r, d, min_group_size=2**31)


def test_compact_samples_keep_their_record_order_and_embedded_measurements():
    r, d = saved()
    samples = compact_shocks(r, d)
    dense = shock_front_catalog(r, d)
    compact, labels = shock_front_catalog(samples, return_labels=True)
    np.testing.assert_array_equal(dense, compact)
    assert labels.shape == (len(samples),)
    assert 'dissipation_selected_indices' not in samples.columns
    # Dense originals can be released without invalidating results or samples.
    r.clear()
    d.clear()
    np.testing.assert_array_equal(shock_front_catalog(samples), compact)


def test_float_summary_overflow_and_missing_units():
    r, d = saved()
    r.mach[:] = 1.e100
    d.total[:] = 1.e308
    out = shock_front_catalog(r, d)[0]
    assert np.isnan(out['mach']) and np.isnan(out['diss_rate'])
    assert out['quality'] & QUALITY_NO_DISS_RATE and out['quality'] & QUALITY_PARTIAL_SUMMARY
    r.position_unit = None
    with pytest.raises(ValueError, match='position_unit'):
        shock_front_catalog(r, d)


def test_compiled_and_scipy_union_produce_identical_results(monkeypatch):
    r, d = saved()
    expected = shock_front_catalog(r, d, max_neighbor_pairs=2)
    original = fronts._Components.__init__
    def without_kernel(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.kernel = None
    monkeypatch.setattr(fronts._Components, '__init__', without_kernel)
    actual = shock_front_catalog(r, d, max_neighbor_pairs=3)
    np.testing.assert_array_equal(expected, actual)


def test_dissipation_changes_summaries_only_not_group_membership():
    r, d = saved([[x, y, 0.] for x in (0., 1.) for y in range(5)])
    expected, labels = shock_front_catalog(r, d, return_labels=True)
    d.area[:] = np.nan
    d.total[:] = -1.
    changed, changed_labels = shock_front_catalog(r, d, return_labels=True)
    np.testing.assert_array_equal(labels, changed_labels)
    np.testing.assert_array_equal(expected[['front_id', 'ncell']], changed[['front_id', 'ncell']])




def test_spatial_pair_batches_are_bounded():
    from scipy.spatial import cKDTree
    points = np.zeros((8, 3))
    pairs = list(fronts._tree_pairs(cKDTree(points), np.arange(8), np.arange(8),
                                   points, np.ones(8), 3, 1))
    assert sum(len(left) for left, _ in pairs) == 64
    assert all(len(left) <= 3 for left, _ in pairs)


def test_mixed_signed_unsigned_large_ids_are_never_rounded():
    ids = np.arange(4, dtype=np.int64)+2**60
    r, d = saved(ids=ids)
    expected = shock_front_catalog(r, d)
    shuffled = {key: value[::-1].copy() for key, value in as_dict(d).items()}
    shuffled['selected_indices'] = r.selected_indices[::-1].astype(np.uint64)
    np.testing.assert_array_equal(shock_front_catalog(r, shuffled), expected)


def test_missing_mean_members_keep_their_small_but_valid_weights():
    r, d = saved()
    d.area[:] = [1.e-300, 1.e300, 1.e300, 1.e300]
    r.normal[1:] = np.nan
    cat = shock_front_catalog(r, d, chunk_size=1, min_group_size=1)[0]
    assert cat['mach'] == 3.
    np.testing.assert_allclose(cat['normal'], [1, 0, 0])
    assert cat['quality'] & QUALITY_PARTIAL_SUMMARY
    assert not cat['quality'] & QUALITY_NO_AREA


def test_summary_chunking_and_legacy_pickle_slot_absence():
    r, d = saved()
    baseline = shock_front_catalog(r, d)
    small = shock_front_catalog(r, d, chunk_size=1)
    for name in front_dtype.names:
        np.testing.assert_allclose(small[name], baseline[name])
    d = pickle.loads(pickle.dumps(d))
    assert not hasattr(d, 'selected_indices')
    np.testing.assert_array_equal(shock_front_catalog(r, d), baseline)
    np.testing.assert_array_equal(shock_front_catalog(r, d), baseline)


def test_dissipation_producer_preserves_dense_and_compact_row_order_without_ids():
    from shocktest.pyShockFinder import compute_dissipation
    r, _ = saved(ids=[4, 7, 8, 9])
    # Supply existing results, never rerun the detector even in this check.
    r.shock[1] = False
    cell = {('T', 'K'): np.full(10, 1.e7), ('rho', 'Msol/kpc3'): np.full(10, 1.e6),
            ('dx', 'km'): np.full(10, 3.0856775814913673e16)}
    dense = compute_dissipation(cell, r)
    compact = compute_dissipation(cell, r, _compact=True)
    assert not hasattr(dense, 'selected_indices')
    assert not hasattr(compact, 'selected_indices')
    np.testing.assert_array_equal(compact.total, dense.total[r.shock])
    # Compact outputs must not masquerade as dense row-aligned products.
    with pytest.raises(ValueError, match='must have shape'):
        shock_front_catalog(r, compact, min_group_size=1)


def test_missing_mach_column_keeps_accepted_centers_and_flags_summary():
    r, d = saved()
    r.mach = None
    out = shock_front_catalog(r, d, min_group_size=1)[0]
    assert out['ncell'] == 1 and np.isnan(out['mach'])
    assert out['quality'] & QUALITY_PARTIAL_SUMMARY




def test_compact_center_aliases_and_unsigned_id_overflow():
    r, d = saved()
    r.center_index[0] = 1
    samples = compact_shocks(r, d)
    out, labels = shock_front_catalog(samples, return_labels=True)
    assert out['ncell'][0] == 3 and out['area'][0] == 9.
    np.testing.assert_array_equal(labels, [-1, 0, 0, 0])
    r, d = saved()
    d.selected_indices = np.array([2**63, 101, 102, 103], dtype=np.uint64)
    with pytest.raises(OverflowError, match='int64'):
        shock_front_catalog(r, d)


def test_mach_tolerance_splits_large_local_jumps_but_allows_gradual_change():
    r, d = saved(np.c_[np.zeros(6), np.arange(6), np.zeros(6)])
    r.mach[:] = [2., 2.1, 2.2, 5., 5.1, 5.2]
    cat, labels = shock_front_catalog(r, d, return_labels=True)
    np.testing.assert_array_equal(cat['ncell'], [3, 3])
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
    assert len(shock_front_catalog(r, d, mach_tolerance=1.)) == 1
    r.mach[:] = [2., 2.5, 3., 3.5, 4., 4.5]
    assert len(shock_front_catalog(r, d)) == 1
    assert not len(shock_front_catalog(r, d, mach_tolerance=0.))


@pytest.mark.parametrize('value', [-.1, 1.1, np.nan, np.inf])
def test_invalid_mach_tolerance_rejected(value):
    r, d = saved()
    with pytest.raises(ValueError, match='mach_tolerance'):
        shock_front_catalog(r, d, mach_tolerance=value)


@pytest.mark.parametrize('option', ['box_size', 'base_cell_size', 'linking_length', 'neighbor_tables', 'assume_aligned'])
def test_removed_geometry_options_rejected(option):
    r, d = saved()
    with pytest.raises(TypeError):
        shock_front_catalog(r, d, **{option: 1.})


def test_crop_does_not_connect_opposite_boundaries():
    r, d = saved([[0, 9.5, 0], [0, .5, 0], [0, 1.5, 0]])
    cat, labels = shock_front_catalog(r, d, return_labels=True, min_group_size=1)
    np.testing.assert_array_equal(cat['ncell'], [1, 2])
    np.testing.assert_array_equal(labels, [0, 1, 1])


def test_edge_touch_can_join_otherwise_separate_groups():
    r, d = saved([[0,0,0], [0,1,0], [0,2,0], [0,3,1], [0,4,1], [0,5,1]])
    face = shock_front_catalog(r, d, connectivity='face')
    touch = shock_front_catalog(r, d, connectivity='touch')
    np.testing.assert_array_equal(face['ncell'], [3,3])
    np.testing.assert_array_equal(touch['ncell'], [6])


def test_single_public_catalog_name():
    import shocktest
    from examples import galaxy
    assert shocktest.shock_front_catalog is fronts.shock_front_catalog
    assert not hasattr(shocktest, 'build_shock_catalog')
    assert not hasattr(galaxy, 'shock_front_catalog')
    assert callable(galaxy.shock_front_samples)
