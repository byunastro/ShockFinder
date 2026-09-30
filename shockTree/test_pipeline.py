"""End-to-end synthetic saved-file tests and corruption/invariant checks."""

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .config import MatchOptions, PipelineConfig, shock_tree_dtype
from .export_metadata import snapshot_row
from .inputs import load_snapshot, read_metadata_table, selected_files
from .model import Metadata, decode_key, node_key
from .pipeline import build, save_compact
from .tree import validate_tree


def write_snapshot(directory, timestep, positions, normals=None, mach=None, ids=None, catalog_ids=None):
    """Write a tiny, explicitly synthetic dense producer schema; no detector."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    count = len(positions)
    ids = np.arange(count) if ids is None else np.asarray(ids)
    n = int(ids.max()) + 1 if len(ids) else 0
    mask = np.zeros(n, dtype=bool)
    mask[ids] = True
    center = np.full(n, -1, dtype=np.int32)
    center[ids] = ids
    pos = np.zeros((n, 3))
    pos[ids] = positions
    normal = np.zeros((n, 3))
    normal[ids] = np.tile([1., 0., 0.], (count, 1)) if normals is None else normals
    mach_array = np.zeros(n)
    mach_array[ids] = 2.0 if mach is None else mach
    result = {"shock": mask, "center_index": center, "selected_indices": np.arange(n, dtype=np.int32),
              "mach": mach_array, "pos": np.asfortranarray(pos), "normal": np.asfortranarray(normal),
              "dx": np.full(n, .01), "level": np.full(n, 20, dtype=np.int32), "position_unit": "kpc"}
    flux = np.zeros(n)
    flux[ids] = 3.0
    area = np.full(n, .0001)
    diss = {"flux": flux, "total": flux * area, "area": area, "efficiency": np.full(n, .1), "sound_speed": np.full(n, 100.)}
    catalog = None if catalog_ids is None else {"shock_id": np.asarray(catalog_ids, dtype=np.int64)}
    for kind, value in (("result", result), ("dissipation", diss), ("catalog", catalog)):
        (directory / f"NC_{kind}_{timestep:05d}.pkl").write_bytes(pickle.dumps(value, protocol=4))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.cfg = PipelineConfig(input_dir=str(self.directory), output_path=str(self.directory / "tree.npz"),
                                  snapshot_start=605, snapshot_end=610, metadata_path=str(self.directory / "metadata.csv"),
                                  coordinate_frame="physical", coordinate_origin="same_simulation_origin", periodic=False,
                                  max_plot_pairs=0, matching=MatchOptions(max_speed_kms=200), chunk_rows=2)

    def tearDown(self):
        self.temp.cleanup()

    def metadata(self, rows):
        with Path(self.cfg.metadata_path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["timestep", "aexp", "time_gyr"])
            writer.writeheader()
            writer.writerows({"timestep": t, "aexp": a, "time_gyr": age} for t, a, age in rows)

    def load(self):
        with np.load(self.cfg.output_path, allow_pickle=False) as archive:
            self.assertEqual(archive.files, ["shock_tree"])
            return archive["shock_tree"]

    def test_end_to_end_sparse_snapshots_original_ids_and_branch_selection(self):
        write_snapshot(self.directory, 605, [[0, 0, 0], [10, 0, 0]], ids=[1, 4])
        write_snapshot(self.directory, 607, [[.5, 0, 0], [10, 0, 0]], ids=[2, 3])
        write_snapshot(self.directory, 610, [[1, 0, 0], [50, 0, 0]], ids=[0, 4])
        self.metadata([(605, 1, 0), (607, 1, .01), (610, 1, .02)])
        report = build(self.cfg)
        tree = self.load()
        self.assertEqual(tree.dtype, shock_tree_dtype)
        np.testing.assert_array_equal(tree["shock_id"], [1, 4, 2, 3, 0, 4])
        branch = np.sort(tree[tree["last"] == tree[0]["last"]], order="timestep")
        np.testing.assert_array_equal(branch["timestep"], [605, 607, 610])
        self.assertEqual(branch[0]["fat"], -1)
        self.assertTrue(np.isnan(branch[0]["score_fat"]))
        self.assertEqual(branch[-1]["son"], -1)
        self.assertTrue(np.isnan(branch[-1]["score_son"]))
        self.assertEqual(branch[0]["son"], node_key(607, 2))
        self.assertEqual(branch[1]["fat"], node_key(605, 1))
        self.assertEqual(branch[0]["score_son"], branch[1]["score_fat"])
        self.assertEqual(report["validation"]["primary_links"], 3)
        self.assertTrue(report["validation"]["all_12_invariants_passed"])

    def test_repeated_local_terminal_ids_do_not_mix_isolated_branches(self):
        write_snapshot(self.directory, 605, [[0, 0, 0]])
        write_snapshot(self.directory, 610, [[100, 0, 0]])
        self.metadata([(605, 1, 0), (610, 1, .01)])
        build(self.cfg)
        tree = self.load()
        self.assertEqual(list(tree["shock_id"]), [0, 0])
        self.assertNotEqual(tree[0]["last"], tree[1]["last"])
        self.assertEqual(len(tree[tree["last"] == tree[0]["last"]]), 1)
        np.testing.assert_array_equal(tree["first"], node_key(tree["timestep"], tree["shock_id"]))

    def test_cosmological_expansion_does_not_create_motion(self):
        # Same comoving position; raw physical position expands with a.
        write_snapshot(self.directory, 605, [[50, 0, 0]])
        write_snapshot(self.directory, 610, [[60, 0, 0]])
        self.metadata([(605, .5, 0), (610, .6, .01)])
        report = build(self.cfg)
        tree = self.load()
        self.assertEqual(report["validation"]["primary_links"], 1)
        np.testing.assert_array_equal(tree["x"], [50, 60])
        self.assertAlmostEqual(report["pairs"][0]["prediction_residual_comoving_kpc"]["max"], 0)

    def test_comoving_input_has_physical_output_positions(self):
        self.cfg.coordinate_frame = "comoving"
        write_snapshot(self.directory, 605, [[100, 0, 0]])
        write_snapshot(self.directory, 610, [[100, 0, 0]])
        self.metadata([(605, .5, 0), (610, .6, .01)])
        build(self.cfg)
        np.testing.assert_array_equal(self.load()["x"], [50, 60])

    def test_catalog_ids_join_safely_when_reversed(self):
        self.cfg.catalog_mode = "identified"
        write_snapshot(self.directory, 605, [[0, 0, 0], [10, 0, 0]], ids=[1, 4], catalog_ids=[4, 1])
        self.metadata([(605, 1, 0)])
        build(self.cfg)
        np.testing.assert_array_equal(self.load()["shock_id"], [1, 4])

    def test_invalid_normal_is_recorded_and_excluded(self):
        write_snapshot(self.directory, 605, [[0, 0, 0], [10, 0, 0]], normals=[[2, 0, 0], [0, 0, 0]])
        self.metadata([(605, 1, 0)])
        report = build(self.cfg)
        tree = self.load()
        self.assertEqual(len(tree), 1)
        np.testing.assert_array_equal(tree[0]["n"], [1, 0, 0])
        self.assertEqual(report["snapshots"][0]["invalid_reasons"]["normal_invalid"], 1)
        self.assertIn("normal_invalid", (self.directory / "tree_diagnostics/invalid_records.csv").read_text())

    def test_join_refuses_wrong_center_identifier(self):
        write_snapshot(self.directory, 605, [[0, 0, 0]])
        path = self.directory / "NC_result_00605.pkl"
        result = pickle.loads(path.read_bytes())
        result["center_index"][0] = 7
        path.write_bytes(pickle.dumps(result, protocol=4))
        files, _ = selected_files(self.cfg)
        with self.assertRaisesRegex(ValueError, "dense join refused"):
            load_snapshot(files[605], Metadata(605, 1, 0), self.cfg)

    def test_missing_physics_fails_without_fabricating_tree(self):
        write_snapshot(self.directory, 605, [[0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "exact snapshot metadata"):
            build(self.cfg)
        self.assertFalse(Path(self.cfg.output_path).exists())

    def test_validation_rejects_cycle_wrong_reciprocity_and_bad_score(self):
        write_snapshot(self.directory, 605, [[0, 0, 0]])
        write_snapshot(self.directory, 610, [[0, 0, 0]])
        self.metadata([(605, 1, 0), (610, 1, .01)])
        build(self.cfg)
        tree = self.load()
        metadata = read_metadata_table(self.cfg.metadata_path, [605, 610], self.cfg)
        segments = {605: (0, 1), 610: (1, 2)}
        corrupt = tree.copy()
        corrupt[1]["son"] = node_key(605, 0)
        corrupt[1]["score_son"] = .5
        with self.assertRaises(ValueError):
            validate_tree(corrupt, segments, metadata, 1)
        corrupt = tree.copy()
        corrupt[1]["fat"] = -1
        corrupt[1]["score_fat"] = np.nan
        with self.assertRaisesRegex(ValueError, "reciprocal"):
            validate_tree(corrupt, segments, metadata, 1)
        corrupt = tree.copy()
        corrupt[0]["score_son"] = 1.1
        with self.assertRaisesRegex(ValueError, "score"):
            validate_tree(corrupt, segments, metadata, 1)

    def test_all_invalid_produces_loadable_empty_tree(self):
        write_snapshot(self.directory, 605, [[0, 0, 0]], normals=[[0, 0, 0]])
        self.metadata([(605, 1, 0)])
        report = build(self.cfg)
        self.assertEqual(len(self.load()), 0)
        self.assertEqual(report["validation"]["branches"], 0)

    def test_reference_encoding_bounds_and_round_trip(self):
        encoded = node_key([0, 785], [0, 2**32 - 1])
        t, ids = decode_key(encoded)
        np.testing.assert_array_equal(t, [0, 785])
        np.testing.assert_array_equal(ids, [0, 2**32 - 1])
        for timestep, identity in ((-1, 0), (2**31, 0), (785, 2**32), (785, -1)):
            with self.assertRaises(ValueError):
                node_key(timestep, identity)

    def test_disconnected_chains_cannot_share_branch_labels(self):
        write_snapshot(self.directory, 605, [[0, 0, 0], [10, 0, 0]])
        write_snapshot(self.directory, 610, [[0, 0, 0], [10, 0, 0]])
        self.metadata([(605, 1, 0), (610, 1, .01)])
        build(self.cfg)
        tree = self.load()
        metadata = read_metadata_table(self.cfg.metadata_path, [605, 610], self.cfg)
        tree["first"][[1, 3]] = tree[0]["first"]
        tree["last"][[1, 3]] = tree[0]["last"]
        with self.assertRaisesRegex(ValueError, "root|terminal|disconnected"):
            validate_tree(tree, {605: (0, 2), 610: (2, 4)}, metadata, 2)

    def test_explicit_manifest_resolves_arbitrary_names(self):
        write_snapshot(self.directory, 605, [[0, 0, 0]])
        row = {"timestep": 605}
        for kind in ("result", "dissipation", "catalog"):
            source = self.directory / f"NC_{kind}_00605.pkl"
            destination = self.directory / f"arbitrary_{kind}_2025_00605.pkl"
            source.rename(destination)
            row[f"{kind}_path"] = destination.name
        manifest = self.directory / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        self.cfg.manifest_path = str(manifest)
        self.metadata([(605, 1, 0)])
        build(self.cfg)
        self.assertEqual(len(self.load()), 1)

    def test_metadata_export_uses_exact_headers_and_box_units(self):
        row = snapshot_row(785, {"aexp": .5, "z": 1, "age": 5.8, "boxlen": 1}, .01, True)
        self.assertEqual(row["aexp"], .5)
        self.assertEqual(row["time_gyr"], 5.8)
        self.assertEqual(row["box_x_comoving_kpc"], 200)
        with self.assertRaisesRegex(ValueError, "disagree"):
            snapshot_row(785, {"aexp": .5, "z": 2, "age": 5.8}, .01, False)

    def test_metadata_box_cannot_silently_override_disagreeing_configuration(self):
        self.cfg.periodic = True
        self.cfg.box_comoving_kpc = [20., 20., 20.]
        Path(self.cfg.metadata_path).write_text(
            "timestep,aexp,time_gyr,box_x_comoving_kpc,box_y_comoving_kpc,box_z_comoving_kpc\n"
            "605,1,0,10,10,10\n")
        with self.assertRaisesRegex(ValueError, "box disagree"):
            read_metadata_table(self.cfg.metadata_path, [605], self.cfg)

    def test_resource_failure_preserves_previous_main_output_and_removes_staging(self):
        write_snapshot(self.directory, 605, [[0, 0, 0], [.1, 0, 0]])
        write_snapshot(self.directory, 610, [[0, 0, 0], [.1, 0, 0]])
        self.metadata([(605, 1, 0), (610, 1, .01)])
        self.cfg.matching.max_candidate_edges = 2
        previous = b"previous main output must be preserved"
        Path(self.cfg.output_path).write_bytes(previous)
        with self.assertRaisesRegex(ValueError, "memory guard"):
            build(self.cfg)
        self.assertEqual(Path(self.cfg.output_path).read_bytes(), previous)
        self.assertEqual(list(self.directory.glob(".shock_tree_work_*")), [])
        report = json.loads((self.directory / "tree_diagnostics/run.json").read_text())
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
