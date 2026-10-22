"""Read-only mapping of large protocol-4 shock pickles."""

import pickle

import numpy as np

from examples.galaxy import shock_product_from_catalog
from examples.saved_shock_pickle import open_saved_shock_snapshot
from shocktest import shock_front_catalog
from shocktest.core import ShockResult
from shocktest.pyShockFinder import DissipationResult


def test_mapped_fortran_coordinates_match_front_catalog(tmp_path):
    n = 5
    pos = np.asfortranarray(np.column_stack((np.arange(n, dtype=float),
                                             np.zeros(n), np.ones(n))))
    normal = np.asfortranarray(np.tile([1., 0., 0.], (n, 1)))
    rows = np.arange(n, dtype=np.int32)
    result = ShockResult(
        mach=np.full(n, 2.5), shock=np.ones(n, bool),
        center_index=rows.copy(), upstream_index=rows.copy(), downstream_index=rows.copy(),
        selected_indices=rows.copy(), pos=pos, dx=np.ones(n), normal=normal,
        level=np.full(n, 12, np.int32), zone_width=np.ones(n),
        position_unit="kpc")
    dissipation = DissipationResult(np.ones(n), np.ones(n), np.ones(n),
                                    np.zeros(n), np.zeros(n))
    catalog, labels = shock_front_catalog(result, dissipation,
                                          return_labels=True, min_group_size=1)
    for name, value in (("result", result), ("dissipation", dissipation), ("labels", labels)):
        with (tmp_path/f"{name}_00785.pkl").open("wb") as stream:
            pickle.dump(value, stream, protocol=4)
    mapped_result, mapped_dissipation, mapped_labels = open_saved_shock_snapshot(
        tmp_path, 785, position_unit="kpc")
    np.testing.assert_array_equal(mapped_result.pos, pos)
    np.testing.assert_array_equal(mapped_result.normal, normal)
    np.testing.assert_array_equal(mapped_labels, labels)
    product = shock_product_from_catalog(
        catalog, mapped_labels, mapped_result, mapped_dissipation,
        785, front_ids=catalog["front_id"])
    assert sum(len(member["shock_id"]) for member in product["members"]) == n
