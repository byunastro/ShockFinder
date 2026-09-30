"""Regression checks for the public object-only example functions."""

import copy
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import gc
import weakref
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from threading import Timer
from unittest.mock import Mock, patch

import numpy as np

from examples.shock_catalog import (
    get_merger_shock_members, merger_shock_catalog,
    _MergerOptions, _merger_geometry, _merger_group_cells, _merger_verify_epochs,
    cache_merger_shock_inputs, load_merger_shock_inputs, _merger_compact_results,
    _merger_group_fingerprint, _merger_source_fingerprint, _merger_summarize_front,
    merger_neighbor_pool, _merger_extraction_bounds,
    prepare_merger_shock_inputs,
)
from shocktest.core import ShockResult
from shocktest.pyShockFinder import DissipationResult
import examples.shock_catalog as merger_module


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


class SnapshotCatalogTests(unittest.TestCase):
    def test_default_is_independent_final_snapshot_with_measured_evidence_only(self):
        result, diss = saved_output(100.)
        info = cluster_history()
        # Even malformed/irrelevant previous outputs must have no influence.
        info["previous_catalog"] = {"tracking_state": "unused"}
        before = pickle.dumps(info)
        catalog = merger_shock_catalog(710, result, diss, info, thread=20)
        merger_shock_catalog(605, result, diss, info, thread=1)
        self.assertFalse(hasattr(merger_module, "_merger_link_fronts"))
        self.assertFalse(hasattr(merger_module, "finalize_merger_shock_catalogs"))
        self.assertNotIn("tracking_state", catalog)
        self.assertEqual(catalog["analysis_mode"], "snapshot")
        self.assertEqual(catalog["metadata"]["time_gyr"], info["t_BB"][2])
        self.assertEqual(pickle.dumps(info), before)
        self.assertEqual(catalog["neighbor_execution"]["workers_requested"], 20)
        front = catalog["fronts"][0]
        components = front["evidence_components"]
        expected = sum(components[k]*w for k, w in
                       (("epoch", .12), ("extent", .14), ("normal", .14), ("axis", .18), ("dissipation", .10)))/.68
        self.assertAlmostEqual(front["evidence"], expected)
        for key in ("continuity", "outward_motion", "origin"):
            self.assertNotIn(key, components)
        self.assertTrue(front["uncertain"])
        self.assertIn("snapshot_only", front["quality_flags"])
        self.assertNotIn("short_track", front["quality_flags"])
        for key in ("track_origin", "previous_front", "next_front", "propagation_speed_km_s"):
            self.assertNotIn(key, front)
        self.assertEqual(front["front_id"], "710:1")
        with self.assertRaises(TypeError):
            merger_shock_catalog(710, result, diss, info, track=True)
        for option in ("minimum_track_length", "max_link_interval_gyr", "max_front_speed_kpc_gyr", "origin_scale_kpc"):
            with self.assertRaisesRegex(ValueError, "unknown merger_shock_options"):
                merger_shock_catalog(710, result, diss, dict(cluster_history(), merger_shock_options={option: 1}))
        np.testing.assert_array_equal(get_merger_shock_members(catalog, 2, result, diss)["shock_id"], [1, 2, 3])

    def test_compact_inputs_release_all_dense_arrays_and_match_direct_call(self):
        result, diss = saved_output(100.)
        expected = merger_shock_catalog(710, result, diss, cluster_history())
        arrays = [getattr(obj, f.name) for obj in (result, diss) for f in fields(obj)
                  if isinstance(getattr(obj, f.name), np.ndarray)]
        references = [weakref.ref(v) for v in arrays]
        compact = prepare_merger_shock_inputs(710, result, diss, chunk_size=2)
        for value in compact.data.values():
            if isinstance(value, np.ndarray):
                self.assertFalse(value.flags.writeable)
                self.assertTrue(value.flags.owndata)
                self.assertTrue(all(not np.shares_memory(value, source) for source in arrays))
        del result, diss, arrays
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        actual = merger_shock_catalog(710, compact, None, cluster_history())
        for key in ("shock_id", "evidence", "confidence", "uncertain"):
            np.testing.assert_array_equal(actual[key], expected[key])
        self.assertEqual(pickle.dumps(actual["fronts"]), pickle.dumps(expected["fronts"]))
        restored = pickle.loads(pickle.dumps(actual))
        self.assertEqual(get_merger_shock_members(restored, 1, compact)["pos"].shape, (3, 3))

    def test_thread_validation_and_override_do_not_change_options(self):
        result, diss = saved_output(100.)
        for value in (True, np.bool_(True), 0, -1, 1.5, "2"):
            with self.assertRaisesRegex(ValueError, "thread must"):
                merger_shock_catalog(710, result, diss, cluster_history(), thread=value)
        info = cluster_history()
        info["merger_shock_options"]["neighbor_workers"] = 7
        output = merger_shock_catalog(710, result, diss, info, thread=np.int64(1))
        self.assertEqual(output["neighbor_execution"]["workers_requested"], 1)
        self.assertEqual(info["merger_shock_options"]["neighbor_workers"], 7)

    def test_empty_compact_input_and_snapshot_mismatch(self):
        result, diss = saved_output(100.)
        result.shock[:] = False
        compact = prepare_merger_shock_inputs(710, result, diss)
        self.assertEqual(merger_shock_catalog(710, compact, None, cluster_history())["fronts"], [])
        with self.assertRaisesRegex(ValueError, "snapshot"):
            merger_shock_catalog(740, compact, None, cluster_history())
        with self.assertRaisesRegex(ValueError, "pass None"):
            merger_shock_catalog(710, compact, diss, cluster_history())

    def test_fragmented_groups_use_two_arrays_and_yield_views(self):
        n = 1000
        data = {"pos": np.column_stack((np.arange(n)*10., np.zeros((n, 2)))),
                "dx": np.ones(n), "normal": np.tile([1., 0., 0.], (n, 1)), "valid": np.ones(n, bool)}
        groups = _merger_group_cells(data, _MergerOptions(neighbor_backend="scipy"))
        self.assertEqual(len(groups), n)
        self.assertLessEqual(groups.members.nbytes+groups.offsets.nbytes, 8*n+4)
        self.assertTrue(np.shares_memory(groups[0], groups.members))
        self.assertEqual([int(g[0]) for g in groups], list(range(n)))


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
        self.assertIn("excluded_boundary", {f["classification"] for f in catalog["fronts"]})
        self.assertTrue(catalog["uncertain"].all())
        self.assertEqual(pickle.dumps(info), info_before)
        self.assertEqual(pickle.dumps(result), result_before)
        wrong = copy.deepcopy(result)
        wrong.pos[2, 0] += 1
        with self.assertRaisesRegex(ValueError, "measurements do not match"):
            get_merger_shock_members(catalog, 2, wrong)

    def test_independent_outputs_and_terminal_secondary_absence(self):
        info = cluster_history()
        catalogs = []
        for snapshot, x in [(710, 100.), (740, 140.), (785, 180.), (875, 220.), (876, 260.), (877, 300.)]:
            result, diss = saved_output(x)
            catalog = merger_shock_catalog(snapshot, result, diss, info)
            self.assertEqual(catalog["fronts"][0]["front_id"], f"{snapshot}:1")
            self.assertNotIn("tracking_state", catalog)
            self.assertTrue(catalog["uncertain"].all())
            restored = pickle.loads(pickle.dumps(catalog))
            np.testing.assert_array_equal(get_merger_shock_members(restored, 1, result)["shock_id"], [1, 2, 3])
            catalogs.append(catalog)
        self.assertIn("secondary_center_unavailable", catalogs[-1]["fronts"][0]["quality_flags"])
        self.assertIsNone(catalogs[-1]["epoch_verification"]["separation_history"][-1]["separation_kpc"])
        repeat = merger_shock_catalog(710, *saved_output(100.), info)
        self.assertEqual(pickle.dumps(repeat["fronts"]), pickle.dumps(catalogs[0]["fronts"]))

    def test_empty_output_and_missing_history_fields(self):
        info = cluster_history()
        result, diss = saved_output(100.)
        result.shock[:] = False
        catalog = merger_shock_catalog(710, result, diss, info)
        self.assertEqual(catalog["shock_id"].size, 0)
        self.assertEqual(catalog["fronts"], [])
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

    def test_early_centroid_oscillations_do_not_replace_core_passage(self):
        snapshots = [605, 618, 619, 708, 710, 789, 791]
        separation = [1100., 970., 980., 116., 125., 477., 470.]
        centers = {s: (np.zeros(3), np.array([d, 0., 0.]), (600., 600.))
                   for s, d in zip(snapshots, separation)}
        meta = {s: {"time_gyr": 1 + i*.1, "redshift": z}
                for i, (s, z) in enumerate(zip(snapshots, [.85, .82, .81, .676, .67, .579, .57]))}
        epochs = _merger_verify_epochs(_MergerOptions(), centers, meta, snapshots)
        self.assertEqual(epochs["pericenter"]["snapshot"], 708)
        self.assertEqual(epochs["apocenter"]["snapshot"], 789)
        self.assertTrue(epochs["pericenter"]["verified"])
        self.assertFalse(epochs["overlap"]["verified"])
        self.assertFalse(epochs["overlap"]["crossing_bracketed_by_outputs"])


class OpenExtractionTests(unittest.TestCase):
    def test_cube_center_switch_precedes_secondary_tree_termination(self):
        c1, c2 = np.array([100., 200., 300.]), np.array([500., 600., 700.])
        centers = {869: (c1, c2, (600., 600.)), 870: (c1, c2, (600., 600.)),
                   876: (c1, None, (600., None))}
        for snapshot, center in ((869, .5*(c1+c2)), (870, c1), (876, c1)):
            bounds = _merger_extraction_bounds(snapshot, centers, _MergerOptions())
            np.testing.assert_array_equal(bounds["lower_kpc"], center-1500.)
            np.testing.assert_array_equal(bounds["upper_kpc"], center+1500.)
        with self.assertRaisesRegex(ValueError, "both measured centers"):
            _merger_extraction_bounds(869, {869: (c1, None, (600., None))}, _MergerOptions())

    def test_opposite_exterior_faces_never_connect_or_wrap(self):
        data = {"pos": np.array([[-1499., 0., 0.], [1499., 0., 0.]]),
                "dx": np.full(2, 4.), "normal": np.tile([1., 0., 0.], (2, 1)),
                "valid": np.ones(2, bool)}
        self.assertEqual(len(_merger_group_cells(data, _MergerOptions())), 2)
        info = cluster_history()
        info["merger_shock_options"]["box_size_kpc"] = 3000.
        with self.assertRaisesRegex(ValueError, "open boundaries"):
            merger_shock_catalog(710, *saved_output(100.), info)

    def test_entire_cut_component_is_excluded_but_recoverable(self):
        result, diss = saved_output(100.)
        result.pos[5:, 1] = [1475., 1485., 1495.]
        result.mach_consistent[5] = True
        diss.flux[5:], diss.total[5:] = 1.e48, 1.e50
        info = cluster_history()
        info["merger_shock_options"]["boundary_margin_cells"] = 0.
        catalog = merger_shock_catalog(710, result, diss, info)
        excluded = catalog["fronts"][1]
        self.assertEqual(excluded["classification"], "excluded_boundary")
        self.assertTrue(excluded["boundary_excluded"])
        self.assertEqual(excluded["boundary_faces"], ("ymax",))
        self.assertIn("extraction_boundary_intersection", excluded["quality_flags"])
        self.assertTrue(np.isnan(excluded["evidence"]))
        np.testing.assert_array_equal(get_merger_shock_members(catalog, 5, result)["shock_id"], [5, 6, 7])
        np.testing.assert_array_equal(catalog["shock_id"], [1, 2, 3])
        self.assertEqual(catalog["dissipation_reference_p75_erg_s"], diss.total[1:4].sum())

    def test_guard_uses_each_amr_cell_width(self):
        result, diss = saved_output(100.)
        result.pos[5:, 1] = [1450., 1460., 1470.]
        result.dx[5:] = 24.
        for margin, expected in ((0., False), (1., True)):
            info = cluster_history()
            info["merger_shock_options"]["boundary_margin_cells"] = margin
            catalog = merger_shock_catalog(710, result, diss, info)
            self.assertEqual(catalog["fronts"][1]["boundary_excluded"], expected)
            if expected:
                self.assertIn("extraction_boundary_guard", catalog["fronts"][1]["quality_flags"])

    def test_boundary_exclusion_is_independent_per_snapshot(self):
        info, catalogs = cluster_history(), []
        for snapshot, x in ((710, 1400.), (740, 1495.), (785, 1400.)):
            result, diss = saved_output(x)
            result.shock[5:] = False
            catalogs.append(merger_shock_catalog(snapshot, result, diss, info))
        self.assertEqual(catalogs[1]["fronts"][0]["classification"], "excluded_boundary")
        self.assertFalse(catalogs[0]["fronts"][0]["boundary_excluded"])
        self.assertFalse(catalogs[2]["fronts"][0]["boundary_excluded"])
        self.assertEqual([c["fronts"][0]["front_id"] for c in catalogs], ["710:1", "740:1", "785:1"])


class BoundedMergerTests(unittest.TestCase):
    def test_amr_partitions_match_brute_force_with_small_pair_budgets(self):
        rng = np.random.default_rng(4021)
        n = 70
        normal = rng.normal(size=(n, 3))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        data = {"pos": rng.uniform(0., 10., size=(n, 3)),
                "normal": normal, "dx": rng.choice([.5, 1., 2., 4.], n),
                "valid": rng.random(n) > .15}
        for box in (None, 10.):
            config = _MergerOptions(box_size_kpc=box, cell_chunk_size=3,
                                    spatial_query_chunk=9, max_neighbor_pairs=5,
                                    neighbor_backend="scipy")
            # Independent all-pairs graph for this small fixture. The optimized
            # tree search must preserve these AMR contact/normal relations.
            adjacency = {int(i): set() for i in np.flatnonzero(data["valid"])}
            for i in adjacency:
                for j in adjacency:
                    delta = data["pos"][i]-data["pos"][j]
                    if box is not None:
                        delta -= box*np.round(delta/box)
                    reach = .5*(data["dx"][i]+data["dx"][j])+.25*max(data["dx"][i], data["dx"][j])
                    if np.all(np.abs(delta) <= reach) and abs(normal[i]@normal[j]) >= .6:
                        adjacency[i].add(j)
            expected, unseen = set(), set(adjacency)
            while unseen:
                todo, component = [unseen.pop()], set()
                while todo:
                    i = todo.pop()
                    if i not in component:
                        component.add(i)
                        todo.extend(adjacency[i]-component)
                unseen -= component
                expected.add(frozenset(component))
            for budget in (5, 200):
                config.max_neighbor_pairs = budget
                actual = {frozenset(g) for g in _merger_group_cells(data, config)}
                self.assertEqual(actual, expected)

    def test_periodic_front_contact_and_extent(self):
        result, diss = saved_output(0.)
        result.shock[5:] = False
        result.pos[1:4] = [[9.8, 0., 0.], [0., 0., 0.], [.2, 0., 0.]]
        result.dx[:] = 1.
        data = _merger_compact_results(result, diss, chunk_size=2)
        config = _MergerOptions(box_size_kpc=10., cell_chunk_size=1, max_neighbor_pairs=2)
        groups = _merger_group_cells(data, config)
        self.assertEqual(len(groups), 1)
        geom = _merger_geometry((np.zeros(3), np.array([2., 0., 0.]), (1., 1.)), config)
        front = _merger_summarize_front(groups[0], data, geom, 710, 7., .67, config)
        self.assertAlmostEqual(front["extent_x_kpc"], 1.4, places=6)
        self.assertAlmostEqual(front["center_x_kpc"] % 10., 0., places=6)
        self.assertIn("periodic_front_unwrapped", front["quality_flags"])

    def test_streamed_hash_matches_native_member_validation(self):
        result, diss = saved_output(100.)
        data = _merger_compact_results(result, diss, chunk_size=2)
        rows = np.array([0, 1, 2, 4, 5])
        expected = _merger_source_fingerprint(*(data[k][rows] for k in
            ("retained_row", "cell_id", "pos", "mach", "dx", "normal")))
        for size in (1, 2, 100):
            self.assertEqual(_merger_group_fingerprint(rows, data, size), expected)

    def test_compact_selection_and_performance_changes_preserve_assessments(self):
        result, diss = saved_output(100.)
        info = cluster_history()
        full = merger_shock_catalog(710, result, diss, info)
        info["merger_shock_options"].update(expand_candidate_cells=False,
            cell_chunk_size=2, spatial_query_chunk=1, max_neighbor_pairs=2)
        compact = merger_shock_catalog(710, result, diss, info)
        self.assertEqual(full["fronts"], compact["fronts"])
        np.testing.assert_array_equal(compact["shock_id"], [1])
        self.assertEqual(compact["selection_mode"], "front_representatives")
        members = get_merger_shock_members(compact, 1, result, diss)
        np.testing.assert_array_equal(members["shock_id"], full["shock_id"])
        before = pickle.dumps(full)
        next_catalog = merger_shock_catalog(740, *saved_output(140.), info)
        self.assertEqual(next_catalog["selection_mode"], "front_representatives")
        self.assertEqual(pickle.dumps(full), before)
        self.assertEqual(len(full["shock_id"]), 3)


class MergerInputCacheTests(unittest.TestCase):
    def test_cached_snapshots_match_dense_and_restore_original_ids(self):
        info, cached_info = cluster_history(), cluster_history()
        dense_catalogs, cached_catalogs = [], []
        with tempfile.TemporaryDirectory() as directory:
            for snapshot, x in [(710, 100.), (740, 140.), (785, 180.)]:
                result, diss = saved_output(x)
                # Original unit conversion and non-unit normals must survive.
                result.pos = result.pos.astype(float)*3.0856775814913673e16
                result.dx = result.dx.astype(float)*3.0856775814913673e16
                result.position_unit = "km"
                result.normal *= 3.
                before = pickle.dumps((result, diss))
                path = Path(directory)/str(snapshot)
                with patch("shocktest.ShockFinder.find", side_effect=AssertionError("must not rerun detector")):
                    inputs = cache_merger_shock_inputs(snapshot, result, diss, path, chunk_size=2)
                    dense = merger_shock_catalog(snapshot, result, diss, info)
                    cached = merger_shock_catalog(snapshot, inputs, None, cached_info)
                self.assertEqual(pickle.dumps((result, diss)), before)
                self.assertTrue(all(not a.flags.writeable for a in inputs.data.values() if isinstance(a, np.ndarray)))
                self.assertIsInstance(inputs.data["pos"], np.memmap)
                self.assertEqual(dense["membership"]["source_fingerprint"], cached["membership"]["source_fingerprint"])
                np.testing.assert_array_equal(cached["membership"]["front_index"], dense["membership"]["front_index"])
                self.assertEqual(cached["membership"]["front_index"][3], -1)
                members = get_merger_shock_members(cached, 2, inputs)
                np.testing.assert_array_equal(members["shock_id"], [1, 2, 3])
                np.testing.assert_array_equal(members["input_cell_id"], [1001, 1002, 1003])
                np.testing.assert_array_equal(members["center_index"], [1, 2, 3])
                np.testing.assert_array_equal(members["upstream_index"], [0, 0, 0])
                np.testing.assert_allclose(members["pos"], result.pos[[1, 2, 3]]/3.0856775814913673e16)
                np.testing.assert_allclose(members["normal"], result.normal[[1, 2, 3]]/np.linalg.norm(result.normal[[1, 2, 3]], axis=1)[:, None])
                np.testing.assert_allclose(members["upstream_pos"], result.pos[[0, 0, 0]]/3.0856775814913673e16)
                np.testing.assert_array_equal(members["total"], diss.total[[1, 2, 3]])
                self.assertEqual(members["position_unit"], "kpc")
                self.assertEqual(members["original_position_unit"], "km")
                with self.assertRaises(FileExistsError):
                    cache_merger_shock_inputs(snapshot, result, diss, path)
                with self.assertRaisesRegex(ValueError, "snapshot"):
                    merger_shock_catalog(800, inputs, None, cluster_history())
                dense_catalogs.append(dense)
                cached_catalogs.append(cached)
            for dense, cached in zip(dense_catalogs, cached_catalogs):
                np.testing.assert_array_equal(dense["shock_id"], cached["shock_id"])
                np.testing.assert_allclose(dense["evidence"], cached["evidence"], rtol=1.e-12)
                self.assertEqual(dense["quality_flags"], cached["quality_flags"])
                self.assertEqual([f["front_id"] for f in dense["fronts"]], [f["front_id"] for f in cached["fronts"]])

    def test_empty_cache_and_corrupted_source_detection(self):
        result, diss = saved_output(100.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"valid"
            inputs = cache_merger_shock_inputs(710, result, diss, path)
            catalog = merger_shock_catalog(710, inputs, None, cluster_history())
            modified = np.load(path/"pos.npy", mmap_mode="r+")
            modified[0, 0] += 1.
            modified.flush()
            del modified
            with self.assertRaisesRegex(ValueError, "measurements do not match"):
                get_merger_shock_members(catalog, 1, load_merger_shock_inputs(path))
            np.save(path/"dx.npy", np.ones(1), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_merger_shock_inputs(path)
            result.shock[:] = False
            empty = cache_merger_shock_inputs(710, result, diss, Path(directory)/"empty", chunk_size=1)
            self.assertEqual(merger_shock_catalog(710, empty, None, cluster_history())["fronts"], [])


class MergerBackendFallbackTests(unittest.TestCase):
    def test_linux_workers_do_not_spawn_the_calling_script(self):
        with patch.object(merger_module.sys, "platform", "linux"):
            self.assertEqual(merger_module._merger_process_context().get_start_method(), "fork")
        with patch.object(merger_module.sys, "platform", "darwin"):
            self.assertEqual(merger_module._merger_process_context().get_start_method(), "spawn")

    def test_default_parallel_threshold_limits_worker_count(self):
        n = 1_000_000
        config = _MergerOptions(neighbor_workers=60)
        # Inspect the worker cap before allocating million-cell geometry.
        self.assertEqual(merger_module._merger_worker_count(n, config), 2)
        self.assertEqual(merger_module._merger_worker_count(5_000_000, config), 10)
        self.assertEqual(merger_module._merger_worker_count(30, _MergerOptions(
            neighbor_workers=60, neighbor_min_parallel_cells=1)), 30)

    def test_unguarded_script_runs_once_with_linux_process_context(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory)/"unguarded.py"
            marker = Path(directory)/"executions.txt"
            script.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
                "from examples.shock_catalog import merger_shock_catalog\n"
                "from tests.test_merger_shock_catalog import cluster_history, saved_output\n"
                f"with Path({str(marker)!r}).open('a') as stream: stream.write('run\\n')\n"
                "sys.platform = 'linux'\n"
                "info = cluster_history()\n"
                "info['merger_shock_options'].update(neighbor_backend='scipy',\n"
                "    neighbor_min_parallel_cells=1, neighbor_max_halo_ratio=4.)\n"
                "for snapshot in (710, 740):\n"
                "    result, dissipation = saved_output(100.)\n"
                "    catalog = merger_shock_catalog(snapshot, result, dissipation, info, thread=2)\n"
                "    assert catalog['neighbor_execution']['reason'] == 'parallel'\n"
                "    del result, dissipation, catalog\n"
            )
            completed = subprocess.run([sys.executable, str(script)], timeout=30,
                                       cwd=Path(__file__).resolve().parents[1], capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(marker.read_text().splitlines(), ["run"])

    def test_missing_extension_auto_fallback_and_explicit_error(self):
        result, diss = saved_output(100.)
        info = cluster_history()
        with patch.object(merger_module, "_merger_neighbors", None):
            catalog = merger_shock_catalog(710, result, diss, info)
            self.assertEqual(catalog["neighbor_backend"], "scipy")
            np.testing.assert_array_equal(catalog["shock_id"], [1, 2, 3])
            info["merger_shock_options"]["neighbor_backend"] = "fortran"
            with self.assertRaisesRegex(RuntimeError, "Build it"):
                merger_shock_catalog(710, result, diss, info)

    def test_parallel_falls_back_for_small_inputs_or_excessive_overlap(self):
        data = {"pos": np.zeros((4, 3)), "dx": np.ones(4),
                "normal": np.tile([1., 0., 0.], (4, 1)), "valid": np.ones(4, bool)}
        for backend, minimum, reason in (("auto", 100, "below_parallel_size_threshold"),
                                         ("scipy", 1, "excessive_ghost_overlap")):
            diagnostics = {}
            actual = _merger_group_cells(data, _MergerOptions(neighbor_workers=4,
                neighbor_backend=backend, neighbor_min_parallel_cells=minimum), diagnostics=diagnostics)
            self.assertEqual(diagnostics["reason"], reason)
            self.assertEqual(diagnostics["workers_used"], 1)
            self.assertEqual(len(actual), 1)

    def test_scipy_workers_preserve_connections_across_slab_boundaries(self):
        data = {"pos": np.column_stack((np.arange(30.)*.8, np.zeros((30, 2)))),
                "dx": np.ones(30), "normal": np.tile([1., 0., 0.], (30, 1)), "valid": np.ones(30, bool)}
        data["normal"][17:] = [0., 1., 0.]
        serial = _merger_group_cells(data, _MergerOptions(neighbor_backend="scipy"))
        diagnostics = {}
        actual = _merger_group_cells(data, _MergerOptions(neighbor_backend="scipy", neighbor_workers=2,
            neighbor_min_parallel_cells=1, max_neighbor_pairs=3), diagnostics=diagnostics)
        self.assertEqual(diagnostics["workers_used"], 2)
        self.assertEqual(len(actual), len(serial))
        for a, b in zip(actual, serial):
            np.testing.assert_array_equal(a, b)


@unittest.skipUnless(merger_module._merger_neighbors is not None and
                    hasattr(merger_module._merger_neighbors.merger_neighbor_kernel, "merge_component_labels"),
                    "optional Fortran extension must be rebuilt for multiprocessing")
class MergerParallelTests(unittest.TestCase):
    @staticmethod
    def mapped_geometry(directory, data, suffix):
        mapped = {"valid": data["valid"]}
        for key in ("pos", "dx", "normal"):
            path = Path(directory)/f"{key}_{suffix}.npy"
            np.save(path, data[key], allow_pickle=False)
            mapped[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return mapped

    def test_partitions_and_ghost_roots_preserve_amr_and_normal_components(self):
        rng = np.random.default_rng(98731)
        chain = {"pos": np.column_stack((np.arange(24.)*.8, np.zeros((24, 2)))),
                 "dx": np.r_[np.ones(12), np.full(12, 2.)],
                 "normal": np.tile([1., 0., 0.], (24, 1)), "valid": np.ones(24, bool)}
        # All four regions have different smallest ghost members, but this
        # chain must be one global component. Transverse normals split it.
        split = {k: v.copy() for k, v in chain.items()}
        split["normal"][12:] = [0., 1., 0.]
        normal = rng.normal(size=(71, 3))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        random = {"pos": rng.uniform(-2., 10., (71, 3)), "normal": normal,
                  "dx": rng.choice([.25, 1., 2., 4.], 71), "valid": rng.random(71) > .1}
        random["pos"][~random["valid"]] = np.nan
        with tempfile.TemporaryDirectory() as directory, merger_neighbor_pool(4):
            for j, data in enumerate((chain, split, random)):
                data = self.mapped_geometry(directory, data, j)
                serial = _merger_group_cells(data, _MergerOptions(neighbor_backend="fortran"))
                diagnostics = {}
                parallel = _merger_group_cells(data, _MergerOptions(neighbor_backend="fortran",
                    neighbor_workers=4, neighbor_min_parallel_cells=1, neighbor_max_halo_ratio=5.),
                    diagnostics=diagnostics)
                self.assertEqual(len(serial), len(parallel))
                for left, right in zip(serial, parallel):
                    np.testing.assert_array_equal(left, right)
                self.assertEqual(diagnostics["reason"], "parallel")
                self.assertEqual(diagnostics["input_transport"], "memory_map")
                self.assertTrue(diagnostics["reused_pool"])
                self.assertEqual(len(parallel), (1, 2, len(serial))[j])

    def test_memory_map_views_use_their_actual_file_offset(self):
        data = {"pos": np.column_stack((np.arange(20.)*.8, np.zeros((20, 2)))),
                "dx": np.ones(20), "normal": np.tile([1., 0., 0.], (20, 1)),
                "valid": np.ones(20, bool)}
        with tempfile.TemporaryDirectory() as directory, merger_neighbor_pool(2):
            mapped = self.mapped_geometry(directory, data, "offset")
            subset = {k: v[7:] for k, v in mapped.items()}
            expected = _merger_group_cells(subset, _MergerOptions())
            actual = _merger_group_cells(subset, _MergerOptions(neighbor_workers=2,
                neighbor_min_parallel_cells=1, neighbor_max_halo_ratio=4.))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)

    def test_excessive_ghost_overlap_falls_back_to_serial(self):
        data = {"pos": np.zeros((4, 3)), "dx": np.full(4, 10.),
                "normal": np.tile([1., 0., 0.], (4, 1)), "valid": np.ones(4, bool)}
        diagnostics = {}
        groups = _merger_group_cells(data, _MergerOptions(neighbor_workers=4,
            neighbor_min_parallel_cells=1, neighbor_max_halo_ratio=2.), diagnostics=diagnostics)
        self.assertEqual(len(groups), 1)
        self.assertEqual(diagnostics["reason"], "excessive_ghost_overlap")
        self.assertEqual(diagnostics["workers_used"], 1)

    def test_partial_submission_failure_drains_workers_before_releasing_geometry(self):
        data = {"pos": np.zeros((4, 3)), "dx": np.ones(4),
                "normal": np.tile([1., 0., 0.], (4, 1)), "valid": np.ones(4, bool)}
        rows = np.arange(4, dtype=np.int32)
        running = Future()
        running.set_running_or_notify_cancel()
        pool = Mock()
        pool.submit.side_effect = [running, RuntimeError("submission failed")]

        @contextmanager
        def shared_geometry(_):
            try:
                yield {}
            finally:
                self.assertTrue(running.done(), "geometry released while the borrowed worker is still running")

        # The first worker is already running when submitting the next one
        # fails. A borrowed pool will remain alive after the exception.
        timer = Timer(.03, running.set_result, args=((rows, rows),))
        timer.start()
        try:
            with patch.object(merger_module, "_MERGER_NEIGHBOR_POOL", (pool, 2)), \
                    patch.object(merger_module, "_merger_shared_geometry", shared_geometry):
                with self.assertRaisesRegex(RuntimeError, "submission failed"):
                    _merger_group_cells(data, _MergerOptions(neighbor_workers=2,
                        neighbor_min_parallel_cells=1), diagnostics={})
            self.assertTrue(running.done())
        finally:
            timer.join()

    def test_complete_catalog_matches_serial_and_performance_options_may_change(self):
        catalogs = {}
        with tempfile.TemporaryDirectory() as directory, merger_neighbor_pool(2):
            inputs = {}
            for snapshot, x in ((710, 100.), (740, 140.), (785, 180.)):
                inputs[snapshot] = cache_merger_shock_inputs(snapshot, *saved_output(x), Path(directory)/str(snapshot))
            for workers in (1, 2):
                info, outputs = cluster_history(), []
                info["merger_shock_options"].update(neighbor_workers=workers,
                    neighbor_min_parallel_cells=1, neighbor_max_halo_ratio=4.)
                for snapshot, cache in inputs.items():
                    catalog = merger_shock_catalog(snapshot, cache, None, info)
                    outputs.append(catalog)
                catalogs[workers] = outputs
            for left, right in zip(catalogs[1], catalogs[2]):
                for key in ("shock_id", "evidence", "confidence", "uncertain"):
                    np.testing.assert_array_equal(left[key], right[key])
                # NaN evidence on excluded fronts needs a NaN-aware comparison.
                self.assertEqual(pickle.dumps(left["fronts"]), pickle.dumps(right["fronts"]))
                np.testing.assert_array_equal(left["membership"]["front_index"], right["membership"]["front_index"])
                self.assertEqual(left["membership"]["source_fingerprint"], right["membership"]["source_fingerprint"])
            info["merger_shock_options"]["neighbor_workers"] = 1
            catalog = merger_shock_catalog(800, *saved_output(220.), info)
            self.assertEqual(catalog["fronts"][0]["front_id"], "800:1")


@unittest.skipIf(merger_module._merger_neighbors is None, "optional merger Fortran extension is not built")
class MergerFortranTests(unittest.TestCase):
    def test_streamed_geometry_matches_numpy_with_mixed_normals_and_area(self):
        if not hasattr(merger_module._merger_neighbors.merger_neighbor_kernel, "measure_geometry"):
            self.skipTest("rebuild extension for streamed geometry")
        rng = np.random.default_rng(7621)
        n = 113
        normals = rng.normal(size=(n, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        data = {"pos": rng.normal(size=(n, 3))*30., "dx": rng.choice([.5, 1., 2.], n),
                "normal": normals, "area": rng.uniform(.2, 10., n), "mach": rng.uniform(2., 4., n),
                "total": rng.uniform(1., 10., n), "flux": rng.uniform(1., 10., n), "validation_unknown": False}
        rows = np.arange(n, dtype=np.int32)[::2]
        geom = _merger_geometry((np.zeros(3), np.array([200., 0., 0.]), (100., 100.)), _MergerOptions())
        numpy_front = _merger_summarize_front(rows, data, geom, 710, 7., .67,
            _MergerOptions(neighbor_backend="scipy", cell_chunk_size=7))
        fortran_front = _merger_summarize_front(rows, data, geom, 710, 7., .67, _MergerOptions())
        for key, value in numpy_front.items():
            if isinstance(value, float):
                np.testing.assert_allclose(fortran_front[key], value, rtol=2.e-13, atol=2.e-13, err_msg=key)
        self.assertEqual(fortran_front["quality_flags"], numpy_front["quality_flags"])

    def assert_partitions_equal(self, data, box=None, cosine=.6):
        options = dict(box_size_kpc=box, minimum_neighbor_normal_cosine=cosine,
                       cell_chunk_size=3, spatial_query_chunk=5, max_neighbor_pairs=7)
        scipy = _merger_group_cells(data, _MergerOptions(neighbor_backend="scipy", **options))
        fortran = _merger_group_cells(data, _MergerOptions(neighbor_backend="fortran", **options))
        self.assertEqual({frozenset(g) for g in scipy}, {frozenset(g) for g in fortran})

    def test_generic_amr_bins_and_periodic_one_two_three_bin_cases(self):
        rng = np.random.default_rng(9823)
        n = 90
        normal = rng.normal(size=(n, 3))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        normal[:20] = [1., 0., 0.]
        normal[::11] *= -1.
        data = {"pos": rng.uniform(-12., 28., (n, 3)), "normal": normal,
                "dx": rng.choice([.25, 1., 2., 8.], n), "valid": rng.random(n) > .1}
        # Includes wrapped coordinates, signed normals, invalid cells, multiple
        # refinement jumps and periodic target buckets with only 1/2/3 bins.
        data["pos"][~data["valid"]] = np.nan
        for box in (None, 9., 21., 32.):
            for cosine in (0., .6, 1.):
                self.assert_partitions_equal(data, box, cosine)

    def test_exact_contact_and_normal_boundaries(self):
        for distance in (np.nextafter(1.25, 0.), 1.25, np.nextafter(1.25, np.inf)):
            for cosine in (np.nextafter(.6, 0.), .6, np.nextafter(.6, 1.)):
                data = {"pos": np.array([[0., 0., 0.], [distance, 0., 0.]]),
                        "dx": np.ones(2), "valid": np.ones(2, bool),
                        "normal": np.array([[1., 0., 0.], [.6, .8, 0.]])}
                self.assert_partitions_equal(data, cosine=cosine)

    def test_fine_coarse_interface_outside_target_bounding_box(self):
        # The fine centers occupy the bin just above the coarse bounding box;
        # they must still reach the coarse cell at the upper end of that bin.
        data = {"pos": np.array([[100.5, .5, .5], [100.5, .5, 1.5],
                                  [101., -1., 1.], [101., -3., 1.]]),
                "dx": np.array([1., 1., 2., 2.]),
                "normal": np.tile([1., 0., 0.], (4, 1)), "valid": np.ones(4, bool)}
        self.assert_partitions_equal(data)
        groups = _merger_group_cells(data, _MergerOptions(neighbor_backend="fortran"))
        self.assertEqual(len(groups), 1)

    def test_packed_bin_overflow_falls_back_without_changing_parent(self):
        data = {"pos": np.array([[0., 0., 0.], [1.25, 1.25, 1.25], [1.e8, 1.e8, 1.e8]]),
                "dx": np.ones(3), "normal": np.tile([1., 0., 0.], (3, 1)), "valid": np.ones(3, bool)}
        rows = np.arange(3, dtype=np.int32)
        parent = rows.copy()
        status = merger_module._merger_neighbors.merger_neighbor_kernel.connect_bucket(
            data["pos"].T, data["dx"], data["normal"].T, rows, rows, rows, parent, .25, .6, 0., 1)
        self.assertEqual(status, 1)
        np.testing.assert_array_equal(parent, rows)
        self.assert_partitions_equal(data)

    def test_catalog_scores_membership_and_cache_match_scipy(self):
        catalogs = {}
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("scipy", "fortran"):
                info = cluster_history()
                info["merger_shock_options"]["neighbor_backend"] = backend
                outputs = []
                for snapshot, x in [(710, 100.), (740, 140.), (785, 180.)]:
                    result, diss = saved_output(x)
                    if backend == "fortran":
                        result = cache_merger_shock_inputs(snapshot, result, diss, Path(directory)/str(snapshot), chunk_size=2)
                        diss = None
                    output = merger_shock_catalog(snapshot, result, diss, info)
                    self.assertEqual(output["neighbor_backend"], backend)
                    outputs.append(output)
                catalogs[backend] = outputs
            for a, b in zip(catalogs["scipy"], catalogs["fortran"]):
                for key in ("shock_id", "evidence", "confidence", "uncertain"):
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(a["quality_flags"], b["quality_flags"])
                self.assertEqual(a["fronts"], b["fronts"])
                self.assertEqual(a["membership"]["source_fingerprint"], b["membership"]["source_fingerprint"])
                np.testing.assert_array_equal(a["membership"]["front_index"], b["membership"]["front_index"])
            # Independent calls may change their backend.
            info["merger_shock_options"]["neighbor_backend"] = "scipy"
            output = merger_shock_catalog(800, *saved_output(220.), info)
            self.assertEqual(output["fronts"][0]["front_id"], "800:1")

if __name__ == "__main__":
    unittest.main()
