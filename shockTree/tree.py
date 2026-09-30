"""Disk-backed primary branches and exhaustive validation of the minimal dtype."""

import numpy as np

from .config import shock_tree_dtype
from .model import decode_key, node_key


BRANCH_STATE_DTYPE = np.dtype([("length", "<i4"), ("score_sum", "<f8"), ("min_score", "<f8")])


def append_snapshot(tree, start, snapshot):
    stop = start + len(snapshot)
    out = tree[start:stop]
    out[:] = np.zeros((), dtype=shock_tree_dtype)
    out["timestep"] = snapshot.timestep
    out["aexp"] = snapshot.aexp
    out["shock_id"] = snapshot.ids
    out["mach"] = snapshot.mach
    out["n"] = snapshot.normal
    # Output positions are physical kpc. Matching uses the comoving view.
    for axis, name in enumerate("xyz"):
        out[name] = snapshot.pos[:, axis] * snapshot.aexp
    out["fat"] = out["son"] = -1
    out["score_fat"] = out["score_son"] = np.nan
    out["first"] = out["last"] = snapshot.keys
    return start, stop


def link_pair(tree, parent_range, child_range, parent, child, result):
    edge = result.edges[result.accepted]
    p, c = edge["parent"], edge["child"]
    score = result.scores[result.accepted]
    tree["son"][parent_range[0] + p] = node_key(child.timestep, child.ids[c])
    tree["fat"][child_range[0] + c] = node_key(parent.timestep, parent.ids[p])
    tree["score_son"][parent_range[0] + p] = score
    tree["score_fat"][child_range[0] + c] = score


def reference_rows(tree, segments, keys, expected_timestep=None):
    """Resolve global references using sorted per-snapshot original IDs."""
    keys = np.asarray(keys, dtype=np.int64)
    timestep, ids = decode_key(keys)
    if expected_timestep is not None and np.any(timestep != expected_timestep):
        raise ValueError("reference does not point to the adjacent selected snapshot")
    output = np.empty(keys.size, dtype=np.int64)
    for t in np.unique(timestep):
        if int(t) not in segments:
            raise ValueError(f"reference points outside saved tree: snapshot {t}")
        lo, hi = segments[int(t)]
        rows = np.flatnonzero(timestep == t)
        source_ids = tree["shock_id"][lo:hi]
        index = np.searchsorted(source_ids, ids[rows])
        if np.any(index >= len(source_ids)) or not np.array_equal(source_ids[index], ids[rows]):
            raise ValueError("reference shock ID does not exist in its encoded snapshot")
        output[rows] = lo + index
    return output


def construct_branches(tree, segments, state, chunk_rows):
    timesteps = list(segments)
    for ordinal, t in enumerate(timesteps):
        lo, hi = segments[t]
        for start in range(lo, hi, chunk_rows):
            stop = min(hi, start + chunk_rows)
            rows = np.arange(start, stop)
            keys = node_key(t, tree["shock_id"][start:stop])
            tree["first"][start:stop] = keys
            state["length"][start:stop] = 1
            state["score_sum"][start:stop] = 0
            state["min_score"][start:stop] = 1
            linked = tree["fat"][start:stop] != -1
            if np.any(linked):
                if ordinal == 0:
                    raise ValueError("range-first snapshot has a progenitor")
                parents = reference_rows(tree, segments, tree["fat"][start:stop][linked], timesteps[ordinal - 1])
                target = rows[linked]
                tree["first"][target] = tree["first"][parents]
                state["length"][target] = state["length"][parents] + 1
                state["score_sum"][target] = state["score_sum"][parents] + tree["score_fat"][target]
                state["min_score"][target] = np.minimum(state["min_score"][parents], tree["score_fat"][target])
    for ordinal in range(len(timesteps) - 1, -1, -1):
        t = timesteps[ordinal]
        lo, hi = segments[t]
        for start in range(lo, hi, chunk_rows):
            stop = min(hi, start + chunk_rows)
            tree["last"][start:stop] = node_key(t, tree["shock_id"][start:stop])
            linked = tree["son"][start:stop] != -1
            if np.any(linked):
                if ordinal == len(timesteps) - 1:
                    raise ValueError("range-last snapshot has a descendant")
                children = reference_rows(tree, segments, tree["son"][start:stop][linked], timesteps[ordinal + 1])
                target = np.arange(start, stop)[linked]
                tree["last"][target] = tree["last"][children]


def validate_tree(tree, segments, metadata, chunk_rows=1_000_000):
    if tree.dtype != shock_tree_dtype:
        raise ValueError("tree dtype differs from the exact required dtype")
    timesteps = list(segments)
    if timesteps != sorted(set(timesteps)):
        raise ValueError("selected snapshot list is not strictly increasing")
    nodes = links = roots = terminals = 0
    prior_stop = 0
    for ordinal, t in enumerate(timesteps):
        lo, hi = segments[t]
        if lo != prior_stop or hi < lo:
            raise ValueError("snapshot segments do not partition the sorted tree")
        prior_stop = hi
        prior_id = -1
        for start in range(lo, hi, chunk_rows):
            stop = min(hi, start + chunk_rows)
            batch = tree[start:stop]
            ids = batch["shock_id"]
            keys = node_key(t, ids)
            if len(ids) and (ids[0] <= prior_id or np.any(ids[1:] <= ids[:-1])):
                raise ValueError("shock IDs repeat or are not sorted within a snapshot")
            if len(ids):
                prior_id = int(ids[-1])
            if np.any(batch["timestep"] != t) or np.any(batch["aexp"] != metadata[t].aexp):
                raise ValueError("tree snapshot/scale factor does not match metadata")
            if any(np.any(~np.isfinite(batch[name])) for name in ("aexp", "mach", "x", "y", "z")):
                raise ValueError("nonfinite tree measurement")
            if np.any(batch["mach"] <= 0):
                raise ValueError("nonpositive Mach number")
            norms = np.linalg.norm(batch["n"], axis=1)
            if np.any(~np.isfinite(norms)) or not np.allclose(norms, 1, rtol=0, atol=1e-8):
                raise ValueError("tree normals are not finite unit vectors")
            for reference, score in (("fat", "score_fat"), ("son", "score_son")):
                missing = batch[reference] == -1
                if np.any(batch[reference] < -1) or np.any(~np.isnan(batch[score][missing])):
                    raise ValueError("missing-link sentinel/score is invalid")
                values = batch[score][~missing]
                if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
                    raise ValueError("accepted link score is not finite in [0,1]")
                if not np.any(~missing):
                    continue
                adjacent_ordinal = ordinal + (-1 if reference == "fat" else 1)
                if not 0 <= adjacent_ordinal < len(timesteps):
                    raise ValueError("reference crosses the selected range boundary")
                adjacent = timesteps[adjacent_ordinal]
                target = reference_rows(tree, segments, batch[reference][~missing], adjacent)
                opposite = "son" if reference == "fat" else "fat"
                opposite_score = "score_son" if reference == "fat" else "score_fat"
                if not np.array_equal(tree[opposite][target], keys[~missing]):
                    raise ValueError("primary links are not reciprocal")
                if not np.array_equal(tree[opposite_score][target], batch[score][~missing]):
                    raise ValueError("reciprocal link scores differ")
                # Every directed edge strictly increases its topological layer.
                # A cycle would require that ordinal to increase and return to
                # its start; this explicit edge check excludes all such cycles.
                if reference == "son" and metadata[adjacent].time_gyr <= metadata[t].time_gyr:
                    raise ValueError("time fails to increase on an edge; cycle/time-order violation")
                for label in ("first", "last"):
                    if not np.array_equal(tree[label][target], batch[label][~missing]):
                        raise ValueError("linked nodes do not share primary branch endpoints")
            first_rows = reference_rows(tree, segments, batch["first"])
            last_rows = reference_rows(tree, segments, batch["last"])
            if np.any(tree["fat"][first_rows] != -1) or np.any(tree["son"][last_rows] != -1):
                raise ValueError("first/last does not reference a root/terminal")
            if not np.array_equal(tree["first"][first_rows], batch["first"]) or not np.array_equal(tree["last"][last_rows], batch["last"]):
                raise ValueError("branch endpoints do not identify themselves")
            if not np.array_equal(tree["first"][last_rows], batch["first"]) or not np.array_equal(tree["last"][first_rows], batch["last"]):
                raise ValueError("same last identifier labels disconnected primary branches")
            roots_here = batch["fat"] == -1
            terminals_here = batch["son"] == -1
            if not np.array_equal(batch["first"][roots_here], keys[roots_here]):
                raise ValueError("a branch root does not identify itself as first")
            if not np.array_equal(batch["last"][terminals_here], keys[terminals_here]):
                raise ValueError("a branch terminal does not identify itself as last")
            isolated = (batch["fat"] == -1) & (batch["son"] == -1)
            if not np.array_equal(batch["first"][isolated], keys[isolated]) or not np.array_equal(batch["last"][isolated], keys[isolated]):
                raise ValueError("isolated-node labels violate the approved node-key convention")
            nodes += len(batch)
            links += int(np.count_nonzero(batch["son"] != -1))
            roots += int(np.count_nonzero(batch["fat"] == -1))
            terminals += int(np.count_nonzero(batch["son"] == -1))
    if prior_stop != len(tree) or roots != terminals or links != nodes - roots:
        raise ValueError("primary graph does not partition into disjoint one-to-one branches")
    return {"nodes": nodes, "primary_links": links, "branches": roots,
            "all_12_invariants_passed": True, "temporal_cycles": 0,
            "cycle_check": "all reciprocal edges strictly increase selected-snapshot topological layer and physical time",
            "branch_connectivity_check": "globally unique terminal anchors, shared labels on every edge, unique one-to-one chains"}
