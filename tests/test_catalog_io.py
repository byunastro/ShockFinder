import csv
import numpy as np
import pytest
import shocktest
from test_maps import grid_cell


def test_catalog_membership_round_trip_and_retrieval(tmp_path):
    analysis = shocktest.ShockFinder().analyze(grid_cell())
    path = tmp_path/'fronts.npz'
    shocktest.save_shock_catalog(path, analysis.catalog, labels=analysis.labels)
    catalog, labels = shocktest.load_shock_catalog(path, return_labels=True)
    assert catalog.tobytes() == analysis.catalog.tobytes()
    np.testing.assert_array_equal(labels, analysis.labels)
    for front in catalog:
        assert np.count_nonzero(labels == front['front_id']) == front['ncell']
        assert analysis.result.shock[labels == front['front_id']].all()


def test_empty_catalog_round_trip(tmp_path):
    cat = np.empty(0, shocktest.front_dtype)
    labels = np.full(3, -1, np.int32)
    path = shocktest.save_shock_catalog(tmp_path/'empty.npz', cat, labels=labels)
    out, membership = shocktest.load_shock_catalog(path, return_labels=True)
    assert out.dtype == cat.dtype and len(out) == 0
    np.testing.assert_array_equal(membership, labels)


def test_catalog_without_labels_does_not_invent_membership(tmp_path):
    cat = shocktest.ShockFinder().analyze(grid_cell()).catalog
    path = shocktest.save_shock_catalog(tmp_path/'summary.npz', cat)
    with pytest.raises(ValueError, match='no membership'):
        shocktest.load_shock_catalog(path, return_labels=True)


def test_catalog_csv_contains_numeric_fields(tmp_path):
    cat = shocktest.ShockFinder().analyze(grid_cell()).catalog
    path = shocktest.save_shock_catalog_csv(tmp_path/'fronts.csv', cat)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(cat) > 0
    assert float(rows[0]['mach']) == pytest.approx(cat['mach'][0])
    assert {'center_x', 'normal_z', 'diss_rate', 'quality'} <= rows[0].keys()


@pytest.mark.parametrize('version', [1, 2, 999])
def test_obsolete_schemas_rejected(tmp_path, version):
    path = tmp_path/'old.npz'
    np.savez(path, schema_version=version)
    with pytest.raises(ValueError, match='unsupported'):
        shocktest.load_shock_catalog(path)


def test_membership_counts_validated(tmp_path):
    a = shocktest.ShockFinder().analyze(grid_cell())
    labels = np.full_like(a.labels, -1)
    with pytest.raises(ValueError, match='counts'):
        shocktest.save_shock_catalog(tmp_path/'bad.npz', a.catalog, labels=labels)
