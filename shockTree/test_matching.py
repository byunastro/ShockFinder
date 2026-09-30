"""Synthetic physical association cases; no ShockFinder detection is run."""

import unittest

import numpy as np

from .config import MatchOptions
from .matching import EDGE_DTYPE, assign_primary, match_snapshots
from .model import Snapshot


def snapshot(t, positions, ids=None, normals=None, mach=None, dx=None, energy=None, age=None):
    position = np.asarray(positions, dtype=float).reshape(-1, 3)
    count = len(position)
    normal = np.tile([1., 0., 0.], (count, 1)) if normals is None else np.asarray(normals, dtype=float)
    normal = normal / np.linalg.norm(normal, axis=1)[:, None] if count else normal
    return Snapshot(t, 1.0, (t - 605) * 0.01 if age is None else age,
                    np.arange(count, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64),
                    position, normal, np.full(count, 2.) if mach is None else np.asarray(mach, dtype=float),
                    np.full(count, .01) if dx is None else np.asarray(dx, dtype=float),
                    np.full(count, 3.) if energy is None else np.asarray(energy, dtype=float))


def chosen(result):
    return set(zip(result.edges["parent"][result.accepted], result.edges["child"][result.accepted]))


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.options = MatchOptions(max_speed_kms=200.)

    def test_stationary_shock(self):
        parent = snapshot(605, [[0, 0, 0]])
        child = snapshot(606, [[0, 0, 0]])
        result = match_snapshots(parent, child, self.options)
        self.assertEqual(chosen(result), {(0, 0)})
        self.assertAlmostEqual(result.scores[result.accepted][0], 1)

    def test_translating_shock_uses_established_motion(self):
        a = snapshot(605, [[0, 0, 0]])
        b = snapshot(606, [[.5, 0, 0]])
        c = snapshot(607, [[1, 0, 0]])
        first = match_snapshots(a, b, self.options)
        second = match_snapshots(b, c, self.options)
        self.assertEqual(chosen(first), {(0, 0)})
        self.assertEqual(chosen(second), {(0, 0)})
        self.assertEqual(first.stats["history_predictions"], 0)
        self.assertEqual(second.stats["history_predictions"], 1)
        self.assertAlmostEqual(second.edges["distance"][second.accepted][0], 0)

    def test_curved_expanding_shock(self):
        theta = np.arange(8) * np.pi / 4
        normals = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(8)])
        shifted = np.column_stack([np.cos(theta + .03), np.sin(theta + .03), np.zeros(8)])
        a = snapshot(605, normals * 5, normals=normals)
        b = snapshot(606, shifted * 5.3, normals=shifted, mach=np.full(8, 2.1), energy=np.full(8, 3.2))
        result = match_snapshots(a, b, self.options)
        self.assertEqual(chosen(result), {(i, i) for i in range(8)})

    def test_appearance_and_disappearance_with_sparse_snapshots(self):
        a = snapshot(605, [[0, 0, 0], [10, 0, 0]], age=0)
        b = snapshot(610, [[.1, 0, 0], [50, 0, 0]], age=.01)
        result = match_snapshots(a, b, self.options)
        self.assertEqual(chosen(result), {(0, 0)})
        self.assertEqual(result.stats["matched_parent_fraction"], .5)
        self.assertEqual(result.stats["matched_child_fraction"], .5)

    def test_split_chooses_highest_confidence_primary(self):
        a = snapshot(605, [[0, 0, 0]])
        b = snapshot(606, [[0, 0, 0], [.4, 0, 0]])
        result = match_snapshots(a, b, self.options)
        self.assertEqual(chosen(result), {(0, 0)})
        self.assertEqual(result.stats["discarded_secondary_links"], 1)

    def test_merge_chooses_highest_confidence_primary(self):
        a = snapshot(605, [[0, 0, 0], [.4, 0, 0]])
        b = snapshot(606, [[0, 0, 0]])
        result = match_snapshots(a, b, self.options)
        self.assertEqual(chosen(result), {(0, 0)})
        self.assertEqual(result.stats["discarded_secondary_links"], 1)

    def test_nearby_shocks_with_different_normals(self):
        a = snapshot(605, [[0, 0, 0], [.1, 0, 0]], normals=[[1, 0, 0], [0, 1, 0]])
        b = snapshot(606, [[.1, 0, 0], [0, 0, 0]], normals=[[1, 0, 0], [0, 1, 0]])
        result = match_snapshots(a, b, self.options)
        self.assertEqual(chosen(result), {(0, 0), (1, 1)})
        self.assertGreater(result.stats["rejected"]["normal_orientation"], 0)

    def test_periodic_boundary_crossing(self):
        a = snapshot(605, [[9.95, 0, 0]])
        b = snapshot(606, [[.05, 0, 0]])
        result = match_snapshots(a, b, self.options, box=np.full(3, 10.))
        self.assertEqual(chosen(result), {(0, 0)})
        self.assertAlmostEqual(result.edges["distance"][result.accepted][0], .1)
        self.assertAlmostEqual(b.velocity[0, 0], 10)

    def test_amr_refinement_and_derefinement(self):
        a = snapshot(605, [[0, 0, 0]], ids=[99], dx=[1.0])
        b = snapshot(606, [[.8, 0, 0]], ids=[3], dx=[.25])
        c = snapshot(607, [[0, 0, 0]], ids=[178], dx=[1.0])
        self.assertEqual(chosen(match_snapshots(a, b, self.options)), {(0, 0)})
        self.assertEqual(chosen(match_snapshots(b, c, self.options)), {(0, 0)})

    def test_unmatched_snapshot_pair(self):
        result = match_snapshots(snapshot(605, [[0, 0, 0]]), snapshot(606, [[100, 0, 0]]), self.options)
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(len(result.edges), 0)

    def test_missing_dissipation_does_not_destroy_spatial_link(self):
        result = match_snapshots(snapshot(605, [[0, 0, 0]], energy=[0]), snapshot(606, [[0, 0, 0]], energy=[np.nan]), self.options)
        self.assertEqual(chosen(result), {(0, 0)})
        self.assertLess(result.scores[result.accepted][0], 1)

    def test_signed_normal_gate_rejects_opposite_orientation(self):
        result = match_snapshots(snapshot(605, [[0, 0, 0]]), snapshot(606, [[0, 0, 0]], normals=[[-1, 0, 0]]), self.options)
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(result.stats["rejected"]["normal_orientation"], 1)

    def test_ambiguity_margin_reduces_confidence(self):
        a = snapshot(605, [[0, 0, 0]])
        unique = match_snapshots(a, snapshot(606, [[0, 0, 0]]), self.options)
        ambiguous = match_snapshots(a, snapshot(606, [[0, 0, 0], [0, 0, 0]]), self.options)
        self.assertLess(ambiguous.scores.max(), unique.scores.max())

    def test_global_assignment_resolves_competing_parents(self):
        edge = np.zeros(3, dtype=EDGE_DTYPE)
        edge["parent"], edge["child"] = [0, 0, 1], [0, 1, 0]
        scores = np.array([.9, .8, .85])
        accepted = assign_primary(edge, scores, 2, 2, self.options)
        self.assertEqual(set(zip(edge["parent"][accepted], edge["child"][accepted])), {(0, 1), (1, 0)})

    def test_long_gap_and_empty_snapshot_do_not_link(self):
        a = snapshot(605, [[0, 0, 0]], age=0)
        b = snapshot(710, [[0, 0, 0]], age=1)
        self.assertEqual(len(match_snapshots(a, b, self.options).accepted), 0)
        self.assertEqual(len(match_snapshots(a, snapshot(606, []), self.options).accepted), 0)

    def test_candidate_memory_guard_fails_instead_of_truncating(self):
        self.options.max_candidate_edges = 2
        a = snapshot(605, [[0, 0, 0], [.1, 0, 0]])
        b = snapshot(606, [[0, 0, 0], [.1, 0, 0]])
        with self.assertRaisesRegex(ValueError, "memory guard"):
            match_snapshots(a, b, self.options)


if __name__ == "__main__":
    unittest.main()
