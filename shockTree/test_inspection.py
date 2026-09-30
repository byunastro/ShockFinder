"""Small synthetic tests for the bounded input inspection stage only."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .config import shock_tree_dtype
from .inspect_inputs import center_ids, discover
from .pickle_metadata import array_metadata, object_fields, read_metadata


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value, protocol=4):
        path = self.root / name
        path.write_bytes(pickle.dumps(value, protocol=protocol))
        return path

    def test_array_metadata_and_memmaps_preserve_shapes_dtypes_layout_and_alias(self):
        mach = np.arange(128, dtype="<f8")
        fields = {"mach": mach, "mach_temperature": mach,
                  "pos": np.asfortranarray(np.arange(384, dtype="<f8").reshape(128, 3)),
                  "level": np.arange(128, dtype="<i4"),
                  "status": np.arange(128, dtype="<u2"),
                  "big_endian": np.arange(128, dtype=">f8")}
        path = self.write("result_00605.pkl", fields)
        _, metadata = object_fields(read_metadata(path))
        for name, original in fields.items():
            item = array_metadata(metadata[name])
            mapped = item.map(path)
            self.assertEqual(item.dtype, original.dtype)
            self.assertEqual(item.shape, original.shape)
            np.testing.assert_array_equal(mapped, original)
            mapped._mmap.close()
        self.assertTrue(array_metadata(metadata["pos"]).fortran_order)
        self.assertIs(metadata["mach"], metadata["mach_temperature"])

    def test_empty_array_and_none_catalog(self):
        path = self.write("result_00605.pkl", {"mach": np.empty(0, dtype="<f8")})
        _, fields = object_fields(read_metadata(path))
        self.assertEqual(array_metadata(fields["mach"]).map(path).size, 0)
        path = self.write("catalog_00605.pkl", None)
        self.assertEqual(object_fields(read_metadata(path)), ("builtins.NoneType", {}))

    def test_chunked_center_audit_preserves_ids(self):
        fields = {"shock": np.array([False, True, False, True, True, False]),
                  "center_index": np.array([-1, 1, -1, 3, 4, -1], dtype=np.int32)}
        path = self.write("result_00605.pkl", fields)
        _, metadata = object_fields(read_metadata(path))
        ids, counts = center_ids(path, metadata, chunk_rows=2)
        np.testing.assert_array_equal(ids, [1, 3, 4])
        self.assertTrue(counts["center_ids_equal_retained_row"])
        self.assertEqual(counts["detected_shocks"], 3)

    def test_invalid_duplicate_and_negative_center_ids_reported(self):
        fields = {"shock": np.ones(3, dtype=bool), "center_index": np.array([-1, 1, 1], dtype=np.int64)}
        path = self.write("result_00605.pkl", fields)
        _, metadata = object_fields(read_metadata(path))
        _, counts = center_ids(path, metadata, chunk_rows=2)
        self.assertTrue(counts["negative_center_ids"])
        self.assertFalse(counts["center_ids_unique_within_snapshot"])

    def test_file_discovery_handles_flattened_paths_and_sparse_snapshots(self):
        for timestep in (605, 710, 785, 787):
            for kind in ("result", "dissipation", "catalog"):
                (self.root / f"NC_{kind}_{timestep:05d}.pkl").touch()
        discovered, excluded = discover(self.root, 605, 785)
        self.assertEqual(list(discovered), [605, 710, 785])
        self.assertEqual(excluded, [787])
        self.assertEqual(set(discovered[710]), {"result", "dissipation", "catalog"})

    def test_ambiguous_discovery_fails(self):
        (self.root / "result_00605.pkl").touch()
        (self.root / "NC_result_00605.pkl").touch()
        with self.assertRaisesRegex(ValueError, "multiple result files"):
            discover(self.root, 605, 785)

    def test_truncated_and_trailing_pickles_fail(self):
        path = self.write("result_00605.pkl", {"mach": np.arange(128, dtype=float)})
        raw = path.read_bytes()
        path.write_bytes(raw[:-10])
        with self.assertRaisesRegex(ValueError, "truncated|STOP"):
            read_metadata(path)
        path.write_bytes(raw + b"extra")
        with self.assertRaisesRegex(ValueError, "trailing"):
            read_metadata(path)

    def test_unsupported_globals_do_not_execute(self):
        path = self.write("catalog_00605.pkl", Path("/tmp/example"))
        with self.assertRaisesRegex(ValueError, "unsupported pickle global"):
            read_metadata(path)

    def test_required_output_dtype_is_exact(self):
        self.assertEqual(shock_tree_dtype.names, (
            "timestep", "aexp", "mach", "x", "y", "z", "n", "shock_id", "fat", "son",
            "score_fat", "score_son", "first", "last"))
        self.assertEqual(shock_tree_dtype["n"].shape, (3,))
        self.assertEqual(shock_tree_dtype.itemsize, 124)


if __name__ == "__main__":
    unittest.main()
