"""Regression checks for the public object-only example functions."""

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.shock_catalog import (
    finalize_merger_shock_catalogs, get_merger_shock_members, merger_shock_catalog,
    _MergerOptions, _merger_geometry, _merger_group_cells, _merger_verify_epochs,
    cache_merger_shock_inputs, load_merger_shock_inputs, _merger_compact_results,
    _merger_group_fingerprint, _merger_source_fingerprint, _merger_summarize_front,
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
        # Performance settings can change between outputs without resetting
        # the scientific history. Scientific thresholds still cannot change.
        info["previous_catalog"] = full
        next_result, next_diss = saved_output(140.)
        next_catalog = merger_shock_catalog(740, next_result, next_diss, info)
        refreshed = finalize_merger_shock_catalogs([full, next_catalog])
        self.assertEqual(refreshed[0]["selection_mode"], "front_representatives")
        self.assertIs(refreshed[0]["membership"], full["membership"])
        self.assertEqual(len(full["shock_id"]), 3)


class MergerInputCacheTests(unittest.TestCase):
    def test_cached_sequence_matches_dense_and_restores_original_ids(self):
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
                info["previous_catalog"], cached_info["previous_catalog"] = dense, cached
            for dense, cached in zip(finalize_merger_shock_catalogs(dense_catalogs), finalize_merger_shock_catalogs(cached_catalogs)):
                np.testing.assert_array_equal(dense["shock_id"], cached["shock_id"])
                np.testing.assert_allclose(dense["evidence"], cached["evidence"], rtol=1.e-12)
                self.assertEqual(dense["quality_flags"], cached["quality_flags"])
                self.assertEqual([f["track_origin"] for f in dense["fronts"]], [f["track_origin"] for f in cached["fronts"]])

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


@unittest.skipIf(merger_module._merger_neighbors is None, "optional merger Fortran extension is not built")
class MergerFortranTests(unittest.TestCase):
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

    def test_catalog_scores_membership_tracks_and_cache_match_scipy(self):
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
                    info["previous_catalog"] = output
                catalogs[backend] = finalize_merger_shock_catalogs(outputs)
            for a, b in zip(catalogs["scipy"], catalogs["fortran"]):
                for key in ("shock_id", "evidence", "confidence", "uncertain"):
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(a["quality_flags"], b["quality_flags"])
                self.assertEqual(a["fronts"], b["fronts"])
                self.assertEqual(a["membership"]["source_fingerprint"], b["membership"]["source_fingerprint"])
                np.testing.assert_array_equal(a["membership"]["front_index"], b["membership"]["front_index"])
            # Changing only the backend is allowed within the same sequence.
            info["merger_shock_options"]["neighbor_backend"] = "scipy"
            output = merger_shock_catalog(800, *saved_output(220.), info)
            self.assertEqual(output["fronts"][0]["track_origin"], "710:1")

if __name__ == "__main__":
    unittest.main()
