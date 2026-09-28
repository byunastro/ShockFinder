"""Regression checks for the public object-only example functions."""

import copy
import json
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from examples.shock_catalog import (
    finalize_merger_shock_catalogs, get_merger_shock_members, merger_shock_catalog,
    _MergerOptions, _merger_geometry, _merger_group_cells, _merger_verify_epochs,
)
from shocktest.core import ShockResult
from shocktest.pyShockFinder import DissipationResult


def cluster_history():
    snapshots = np.array([600, 605, 710, 740, 785, 800, 875, 876, 877])
    separations = np.array([1300., 1000., 100., 400., 700., 650., 600., np.nan, np.nan])
    first = np.zeros((len(snapshots), 3))
    second = first.copy()
    first[:, 0], second[:, 0] = -separations / 2, separations / 2
    first[-2:, 0] = [100., 110.]
    second[-2:] = np.nan
    return {"iout": snapshots, "ccen1": first, "ccen2": second,
            "rvir1": np.full(len(snapshots), 600.),
            "rvir2": np.array([600.] * 7 + [np.nan, np.nan]),
            "t_BB": 1 + np.arange(len(snapshots)) * .1,
            "redshift": np.array([.86, .85, .67, .64, .58, .57, .55, .54, .53]),
            "mvir1": np.full(len(snapshots), 1.e14),
            "merger_shock_options": {"thresholds_calibrated": True}}


def saved_output(x):
    n = 8
    rows = np.array([1, 2, 3, 5, 6, 7])
    shock = np.zeros(n, bool)
    shock[rows] = True
    center = np.full(n, -1, np.int32)
    center[rows] = rows
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[1:4] = [[x, -10., 0.], [x, 0., 0.], [x, 10., 0.]]
    pos[5:] = [[0., 1490., 0.], [0., 1500., 0.], [0., 1510., 0.]]
    normal = np.zeros_like(pos)
    normal[1:4] = [1., .1, 0.]
    normal[5:] = [0., 1., 0.]
    consistent = shock.copy()
    consistent[5] = False
    result = ShockResult(np.where(shock, 3., 0.).astype(np.float32), shock,
        center, np.where(shock, 0, -1).astype(np.int32), np.where(shock, 4, -1).astype(np.int32),
        np.arange(n, dtype=np.int64) + 1000, pos=pos, dx=np.full(n, 10., dtype=np.float32),
        normal=normal, mach_consistent=consistent, position_unit="kpc")
    flux = np.zeros(n)
    flux[1:4], flux[5:] = 1.e42, 1.e38
    diss = DissipationResult(flux, flux * 100., np.where(shock, 100., 0.),
                             np.zeros(n), np.zeros(n))
    return result, diss


class ObjectApiTests(unittest.TestCase):
    def test_native_center_ids_and_member_recovery(self):
        info = cluster_history()
        result, diss = saved_output(100.)
        info_before = pickle.dumps(info)
        result_before = pickle.dumps(result)
        with patch("shocktest.ShockFinder.find", side_effect=AssertionError("must not rerun detector")):
            catalog = merger_shock_catalog(710, result, diss, info)
        np.testing.assert_array_equal(catalog["shock_id"], [1, 2, 3])
        np.testing.assert_array_equal(result.center_index[catalog["shock_id"]], catalog["shock_id"])
        np.testing.assert_array_equal(result.mach[catalog["shock_id"]], [3., 3., 3.])
        members = get_merger_shock_members(catalog, 2, result, diss)
        np.testing.assert_array_equal(members["shock_id"], [1, 2, 3])
        np.testing.assert_array_equal(members["input_cell_id"], [1001, 1002, 1003])
        np.testing.assert_array_equal(members["upstream_index"], [0, 0, 0])
        np.testing.assert_array_equal(members["pos"], result.pos[[1, 2, 3]])
        self.assertEqual(members["assessment"]["shock_id"], 1)
        self.assertEqual(catalog["membership"]["front_index"][3], -1)
        self.assertIn("uncertain", {f["classification"] for f in catalog["fronts"]})
        self.assertTrue(catalog["uncertain"].all())
        self.assertEqual(pickle.dumps(info), info_before)
        self.assertEqual(pickle.dumps(result), result_before)
        wrong = copy.deepcopy(result)
        wrong.pos[2, 0] += 1
        with self.assertRaisesRegex(ValueError, "measurements do not match"):
            get_merger_shock_members(catalog, 2, wrong)

    def test_temporal_evidence_and_terminal_secondary_absence(self):
        info = cluster_history()
        catalogs, outputs = [], []
        for snapshot, x in [(710, 100.), (740, 140.), (785, 180.), (875, 220.), (876, 260.), (877, 300.)]:
            result, diss = saved_output(x)
            outputs.append(result)
            previous_bytes = pickle.dumps(catalogs[-1]) if catalogs else None
            catalog = merger_shock_catalog(snapshot, result, diss, info)
            if catalogs:
                self.assertEqual(pickle.dumps(catalogs[-1]), previous_bytes)
            catalogs.append(catalog)
            info["previous_catalog"] = catalog
        first_score = catalogs[0]["evidence"].copy()
        finalized = finalize_merger_shock_catalogs(catalogs)
        self.assertTrue(np.all(finalized[0]["evidence"] > first_score))
        self.assertEqual(finalized[0]["confidence"].tolist(), ["high"] * 3)
        self.assertNotIn("short_track", finalized[0]["quality_flags"][0])
        track_origins = [c["fronts"][0]["track_origin"] for c in finalized]
        self.assertEqual(track_origins, ["710:1"] * 6)
        self.assertTrue(finalized[-1]["uncertain"].all())
        self.assertIn("secondary_center_unavailable", finalized[-1]["quality_flags"][0])
        transition = finalized[-2]["fronts"][0]
        self.assertTrue(np.isnan(transition["outward_axis_speed_kpc_gyr"]))
        self.assertIsNone(finalized[-1]["epoch_verification"]["separation_history"][-1]["separation_kpc"])
        restored = pickle.loads(pickle.dumps(finalized[0]))
        np.testing.assert_array_equal(get_merger_shock_members(restored, 1, outputs[0])["shock_id"], [1, 2, 3])
        for frame in restored["tracking_state"]["frames"]:
            self.assertTrue(all("rows" not in front for front in frame["fronts"]))
        self.assertIs(catalogs[-1]["tracking_state"]["frames"][0], catalogs[0]["tracking_state"]["frames"][0])
        with self.assertRaisesRegex(ValueError, "advance in snapshot order"):
            merger_shock_catalog(877, outputs[-1], saved_output(300.)[1], info)
        info["merger_shock_options"]["candidate_score"] = .7
        with self.assertRaisesRegex(ValueError, "options changed"):
            merger_shock_catalog(877, outputs[-1], saved_output(300.)[1], info)

    def test_empty_output_and_missing_history_fields(self):
        info = cluster_history()
        result, diss = saved_output(100.)
        result.shock[:] = False
        catalog = merger_shock_catalog(710, result, diss, info)
        self.assertEqual(catalog["shock_id"].size, 0)
        self.assertEqual(catalog["fronts"], [])
        self.assertEqual(finalize_merger_shock_catalogs([]), [])
        del info["t_BB"]
        with self.assertRaisesRegex(ValueError, "time_gyr"):
            merger_shock_catalog(710, result, diss, info)

    def test_partial_virial_radii_do_not_invent_overlap(self):
        info = cluster_history()
        del info["rvir2"]
        result, diss = saved_output(100.)
        catalog = merger_shock_catalog(710, result, diss, info)
        self.assertFalse(catalog["epoch_verification"]["overlap"]["verified"])
        self.assertTrue(catalog["epoch_verification"]["pericenter"]["verified"])


class MergerGeometryTests(unittest.TestCase):
    def test_amr_contact_and_orientation(self):
        config = _MergerOptions()
        data = {
            "valid": np.ones(3, bool),
            "pos": np.array([[0., 0., 0.], [1.5, 0., 0.], [3., 0., 0.]]),
            "dx": np.array([2., 1., 1.]),
            "normal": np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        }
        groups = _merger_group_cells(data, config)
        self.assertEqual(sorted(map(len, groups)), [1, 2])

    def test_epoch_geometry(self):
        config = _MergerOptions()
        snaps = [600, 605, 710, 740, 785, 800]
        separations = [1300, 1000, 100, 400, 700, 650]
        centers = {snap: (np.array([-sep / 2, 0., 0.]), np.array([sep / 2, 0., 0.]), (600., 600.))
                   for snap, sep in zip(snaps, separations)}
        meta = {snap: {"time_gyr": i * 0.1, "redshift": z} for i, (snap, z) in enumerate(zip(snaps, [.86, .85, .67, .64, .58, .57]))}
        epochs = _merger_verify_epochs(config, centers, meta, snaps)
        self.assertEqual([epochs[name]["snapshot"] for name in ("overlap", "pericenter", "apocenter")], [605, 710, 785])
        self.assertTrue(all(epochs[name]["verified"] for name in ("overlap", "pericenter", "apocenter")))
        self.assertAlmostEqual(_merger_geometry(centers[710], config)["separation_kpc"], 100.)

    def test_terminal_absence_is_not_zero_separation_or_core_passage(self):
        config = _MergerOptions()
        snaps = [874, 875, 876]
        centers = {874: (np.zeros(3), np.array([200., 0., 0.]), (600., 600.)),
                   875: (np.zeros(3), np.array([100., 0., 0.]), (600., 600.)),
                   876: (np.zeros(3), None, (600., None))}
        meta = {s: {"time_gyr": 1 + i * .1, "redshift": .5 - i * .01} for i, s in enumerate(snaps)}
        config.reference_epochs["pericenter"] = {"snapshot": 875, "redshift": .49}
        epochs = _merger_verify_epochs(config, centers, meta, snaps)
        self.assertEqual(epochs["pericenter"]["snapshot"], 875)
        self.assertFalse(epochs["pericenter"]["verified"])
        self.assertFalse(epochs["pericenter"]["local_extremum_bracketed_by_outputs"])
        self.assertIsNone(epochs["separation_history"][-1]["separation_kpc"])
        self.assertIsNone(epochs["separation_history"][-1]["instantaneous_axis"])
        json.dumps(epochs, allow_nan=False)

if __name__ == "__main__":
    unittest.main()
