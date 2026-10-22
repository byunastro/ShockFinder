"""Saved-output discovery and physical front-property selection."""

import numpy as np
import pytest

from examples.galaxy import filter_shock_front_catalog, match_front_tracks
from examples.run_galaxy_stripping import _select_fronts, discover_saved_shock_snapshots


def _catalog():
    dtype = [("front_id", "i4"), ("ncell", "i4"), ("mach", "f4"),
             ("diss_rate", "f8"), ("area", "f8"), ("quality", "u2"),
             ("center", "f8", (3,)), ("extent", "f4", (3,)),
             ("normal", "f4", (3,))]
    catalog = np.zeros(4, dtype=dtype)
    catalog["front_id"] = [101, 3163, 7000, 9000]
    catalog["ncell"] = [20, 12000, 15000, 16000]
    catalog["mach"] = [4., 3.1, 2.4, 3.5]
    catalog["diss_rate"] = [2.e40, 1.2e44, 2.e44, 2.e44]
    catalog["area"] = [10., 100., 100., 100.]
    catalog["quality"] = [0, 0, 0, 3]  # Last row has incomplete area and rate.
    catalog["center"][:, 0] = [0., 10., 20., 30.]
    catalog["extent"][:] = 2.
    catalog["normal"][:, 0] = 1.
    return catalog


def test_front_cuts_are_combined_inclusive_and_reject_incomplete_flux():
    catalog = _catalog()
    filtered = filter_shock_front_catalog(
        catalog, min_mach=3., min_cells=10000, min_flux=1.e42,
        min_diss_rate=1.e44)
    assert filtered["front_id"].tolist() == [3163]
    assert filter_shock_front_catalog(catalog, min_cells=15000)["front_id"].tolist() == [7000, 9000]
    assert filter_shock_front_catalog(catalog, min_flux=1.e42)["front_id"].tolist() == [3163, 7000]
    with pytest.raises(ValueError, match="min_mach"):
        filter_shock_front_catalog(catalog, min_mach=np.nan)
    with pytest.raises(ValueError, match="min_cells"):
        filter_shock_front_catalog(catalog, min_cells=1.5)


def test_forced_front_must_pass_property_cuts_but_need_not_be_nearby():
    catalog = _catalog()
    positions = np.array([[10., 0., 0.]])
    radii = np.array([1.])
    selected, eligible, nearby = _select_fronts(
        catalog, positions, radii, snapshot=785, padding_kpc=0.,
        forced_ids={3163}, min_front_mach=3., min_front_cells=10000,
        min_front_flux=1.e42, min_front_diss_rate=1.e44)
    assert selected.tolist() == [3163]
    assert eligible == 1
    assert nearby == {3163}
    with pytest.raises(ValueError, match="fail the front-property cuts"):
        _select_fronts(catalog, positions, radii, snapshot=785, padding_kpc=0.,
                       forced_ids={7000}, min_front_mach=3., min_front_cells=None,
                       min_front_flux=None, min_front_diss_rate=None)
    selected, _, nearby = _select_fronts(
        catalog, np.array([[0., 0., 0.]]), radii, snapshot=785, padding_kpc=0.,
        forced_ids={3163}, min_front_mach=3., min_front_cells=10000,
        min_front_flux=None, min_front_diss_rate=None)
    assert selected.tolist() == [3163] and nearby == set()


def test_saved_snapshot_range_discovers_only_complete_available_outputs(tmp_path):
    for snapshot in (699, 783, 785, 901):
        for prefix in ("catalog", "labels", "result", "dissipation"):
            (tmp_path/f"{prefix}_{snapshot:05d}.pkl").touch()
    assert discover_saved_shock_snapshots(tmp_path, 700, 900) == (783, 785)
    (tmp_path/"catalog_00787.pkl").touch()
    with pytest.raises(FileNotFoundError, match="incomplete saved products"):
        discover_saved_shock_snapshots(tmp_path, 700, 900)
    with pytest.raises(ValueError, match="0 <= start <= end"):
        discover_saved_shock_snapshots(tmp_path, 900, 700)


def test_track_matching_uses_catalog_ids_not_array_offsets():
    previous = _catalog()[:2].copy()
    current = previous.copy()
    current["front_id"] = [9001, 9002]
    current["center"][:, 0] += 1.
    matched = match_front_tracks(previous, current, [3163], [9002],
                                 track_prefix="783")
    assert matched == {9002: "783:3163"}
